from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from transformers import get_linear_schedule_with_warmup

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import pearsonr
from tqdm import tqdm
import argparse
import logging

# importing custom LORA functions
from lora_layers import inject_lora_to_kq_attn
from roberta_lora import val

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v2-xxlarge")
    parser.add_argument("--task-name", type=str, default="sst2")
    parser.add_argument("--resume-training", type=str, default="None", help="If not 'None', contains path to .pth from which to resume training.")
    parser.add_argument("--resume-optimizer", type=str, default="None", help="If not 'None', contains path to optimizer state from which to resume training.")
    parser.add_argument("--save-dir", type=str, default="../results")

    return parser.parse_args()

# using batch sizes from Appendix D, Table 10
GLUE_TASK_BATCH = {
    "mrpc": 32,
    "mnli": 8,
    "sst2": 8,
    "cola": 4,
    "qnli": 6,
    "qqp": 8,
    "rte": 4
}

GLUE_TASK_LR = {"mnli": 1e-4, "stsb": 2e-4, "sst2": 6e-4, "mrpc": 2e-4, "cola": 1e-4, "qnli": 1e-4, "qqp": 1e-4, "rte": 2e-4}

# Same keys for tokenization
GLUE_TASK_TOKENIZE = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "cola": ("sentence", None),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis")
}

GLUE_SEQ_LEN = {"mnli": 256, "stsb": 128, "sst2": 128, "mrpc": 128, "cola": 64, "qnli": 512, "qqp": 320, "rte": 320}

# from Appendix D, Table 10
GLUE_NUM_EPOCHS = {"mnli": 5, "stsb": 10, "sst2": 16, "mrpc": 30, "cola": 10, "qnli": 8, "qqp": 11, "rte": 11}

GLUE_BINARY_TASKS = ["sst2", "mrpc", "cola", "qnli", "qqp", "rte"]

def create_deberta_dataloaders(args, task_name):
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_name)
    dataset = load_dataset("glue", task_name)

    def tokenize_function(example):
        # some GLUE tasks are single sentence, some are sentence pair cls
        k1, k2 = GLUE_TASK_TOKENIZE[task_name]
        if k2 is None:
            return tokenizer(
                example[k1],
                padding="max_length",
                truncation=True,
                max_length=GLUE_SEQ_LEN[task_name]
            )
        else:
            return tokenizer(
                example[k1],
                example[k2],
                padding="max_length",
                truncation=True,
                max_length=GLUE_SEQ_LEN[task_name]
            )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # rename 'label' to 'labels' if needed
    if 'label' in tokenized_dataset['train'].column_names and 'labels' not in tokenized_dataset['train'].column_names:
        tokenized_dataset = tokenized_dataset.map(
            lambda example: {'labels': example['label']},
            remove_columns=['label']
        )
  
    dataset_columns = ['input_ids', 'attention_mask', 'labels']
    # 'token_type_ids" exists if task requires multiple sentences
    if 'token_type_ids' in tokenized_dataset['train'].column_names:
        dataset_columns.append('token_type_ids')
  
    tokenized_dataset.set_format(type='torch', columns=dataset_columns)

    batch_size = GLUE_TASK_BATCH[task_name]
    train_loader = DataLoader(tokenized_dataset["train"], batch_size=batch_size, shuffle=True)

    if task_name == "mnli":
        val_loader = {
            "matched": DataLoader(tokenized_dataset["validation_matched"], batch_size=batch_size),
            "mismatched": DataLoader(tokenized_dataset["validation_mismatched"], batch_size=batch_size)
        }
    else:   
        val_loader = DataLoader(tokenized_dataset["validation"], batch_size=batch_size)

    return train_loader, val_loader


def train_deberta(args, model):
    """
    Trains LORA weights for DeBERTa-v2-xxlarge model on given GLUE task.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"running on {device}")

    task_name = args.task_name
    # create dataloaders for given GLUE task
    train_loader, val_loader = create_deberta_dataloaders(args, task_name)
    logging.info(f"created dataloaders")

    # based on Appendix D, use AdamW. with diff weight decay parameters
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    if task_name == "stsb":
       optimizer = torch.optim.AdamW(lora_params, lr=GLUE_TASK_LR[task_name], weight_decay=0.1)
    else:
       weight_decay = 0 if task_name in ["mnli", "cola"] else 0.01
       optimizer = torch.optim.AdamW(lora_params, lr=GLUE_TASK_LR[task_name], weight_decay=weight_decay)
    
    # use a warmup ratio 0.1 (Appendix D, Table 10)
    num_train_steps = len(train_loader) * GLUE_NUM_EPOCHS[task_name]
    warmup_steps = int(num_train_steps * 0.1) 

    # use linear schedule with warmup ratio
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_steps
    )

    # potentially reload optimizer and scheduler state
    start_epoch = 0
    if args.resume_training != "None" and args.resume_optimizer != "None":
        logging.info(f"Loading optimizer state from {args.resume_optimizer}")
        checkpoint = torch.load(args.resume_optimizer, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
    
    remaining_epochs = GLUE_NUM_EPOCHS[task_name] - start_epoch
    for e in tqdm(range(start_epoch, remaining_epochs), leave=True):
        model.train()
        train_running_loss = 0
        for batch in tqdm(train_loader, desc=f"train (epoch {e})", position=1, leave=False):
            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_running_loss += loss.item()
      
        # save model after every epoch
        torch.save(model.state_dict(), f"{args.save_dir}/{args.model_name.split('/')[-1]}-e{e}-{task_name}.pth")
        logging.info(f"Model saved to {args.save_dir}/{args.model_name.split('/')[-1]}-e{e}-{task_name}.pth")
      
        avg_train_loss = train_running_loss / len(train_loader)
        if task_name == "mnli":
            # MNLI has two validation sets
            print(f"epoch {e}")
            print(f"training loss: {avg_train_loss:.4f}")
            logging.info(f"epoch {e}")
            logging.info(f"training loss: {avg_train_loss:.4f}")
            
            metrics_matched, avg_val_loss_matched = val_deberta(model, val_loader["matched"], task_name, device)
            metrics_mismatched, avg_val_loss_mismatched = val_deberta(model, val_loader["mismatched"], task_name, device)
            
            matched_size = len(val_loader["matched"].dataset)
            mismatched_size = len(val_loader["mismatched"].dataset)
            avg_val_loss = (avg_val_loss_matched * matched_size + avg_val_loss_mismatched * mismatched_size) / (matched_size + mismatched_size)
            
            # calculate average accuracy
            matched_correct = metrics_matched["accuracy"] * matched_size
            mismatched_correct = metrics_mismatched["accuracy"] * mismatched_size
            overall_accuracy = (matched_correct + mismatched_correct) / (matched_size + mismatched_size)
            
            print(f"val loss: {avg_val_loss:.4f}")
            print(f"overall accuracy: {overall_accuracy:.4f}")
            logging.info(f"val loss: {avg_val_loss:.4f}")
            logging.info(f"overall accuracy: {overall_accuracy:.4f}")
        else:
            metrics, avg_val_loss = val(model, val_loader, task_name, device)
            print(f"\nepoch {e}\ntraining loss: {avg_train_loss:.4f}\nval loss: {avg_val_loss:.4f}")
            logging.info(f"\nepoch {e}\ntraining loss: {avg_train_loss:.4f}\nval loss: {avg_val_loss:.4f}")
            logging.info(f"val metrics: {metrics}\n")

def compute_metrics(y_true, y_pred, task_name):
    # use accuracy for most tasks
    if task_name in ["sst2", "mrpc", "qqp", "rte", "qnli", "mnli"]:
        acc = accuracy_score(y_true, y_pred)
        return {"accuracy": acc}
    elif task_name == "cola":
        # CoLA uses Matthews correlation coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        return {"mcc": mcc}
    elif task_name == "stsb":
        # STS-B uses Pearson correlation
        pearson_corr, _ = pearsonr(y_true, y_pred)
        return {"pearson_corr": pearson_corr}
    else:
        raise RuntimeError(f"{task_name} not supported.")

def main(args):
    logging.basicConfig(
        filename=f'{args.save_dir}/{args.model_name.split("/")[-1]}_{args.task_name}.log',
        level=logging.INFO,
        filemode='a', # append
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if args.task_name in GLUE_BINARY_TASKS:
        model = DebertaV2ForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    elif args.task_name == "mnli":
        model = DebertaV2ForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    else:
        raise RuntimeError(f"Running LoRA on {args.model_name} for GLUE {args.task_name} is not supported.")

    # add low-rank weight matrices and freeze everything else
    inject_lora_to_kq_attn(args, model, rank=8, alpha=8)
    if args.resume_training != "None":
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       logging.info(f"resuming training from {args.resume_training}")
       model.load_state_dict(torch.load(args.resume_training, map_location=device))
    
    logging.info(f"starting LoRA on {args.model_name} on GLUE {args.task_name}")
    # NOTE sanity check to ensure LoRA is only training small subset of parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of trainable parameters: {trainable_params}")
    logging.info(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.4f}%")

    train_deberta(args, model)

if __name__ == "__main__":
    args = get_args()
    if args.model_name != "microsoft/deberta-v2-xxlarge":
        raise RuntimeError(f"This script does not support LoRA for {args.model_name}.")

    main(args)