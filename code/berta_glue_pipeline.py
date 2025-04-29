"""
This script runs a roberta-base OR deberta-base-xxl model with and without LoRA implementation on a single binary GLUE task.

Intended to be run on diff set of hyperparameters from LoRA paper (for experimentation in a shorter time with limited compute).
"""

from transformers import AutoModelForSequenceClassification, RobertaTokenizer, DebertaV2Tokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging

from train_val_berta import val, GLUE_BINARY_TASKS
from lora_layers import inject_lora_to_kq_attn

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="roberta-base")
    parser.add_argument("--lora", type=bool, default=None, help="Add --lora=True if you want the model to injected with LoRA.")
    parser.add_argument("--task-name", type=str, default="rte")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=8, help="LoRA parameter rank")
    parser.add_argument("--alpha", type=int, default=8, help="LoRA parameter alpha")

    parser.add_argument("--resume-training", type=str, default="None", help="If not 'None', contains path to .pth from which to resume training.")
    parser.add_argument("--resume-optimizer", type=str, default="None", help="If not 'None', contains path to optimizer state from which to resume training.")
    parser.add_argument("--save-dir", type=str, default="../results")

    return parser.parse_args()

GLUE_TASK_TOKENIZE = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "cola": ("sentence", None),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis")
}

def create_dataloaders(args, task_name):
  if args.model_name == "roberta-base":
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
  else:
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_name)
  
  dataset = load_dataset("glue", task_name)

  def tokenize_function(example):
    k1, k2 = GLUE_TASK_TOKENIZE[task_name]
    if k2 is None:
        return tokenizer(
            example[k1],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_len
        )
    else:
        return tokenizer(
            example[k1],
            example[k2],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_len
        )

  tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
  # rename 'label' to 'labels' if needed
  if 'label' in tokenized_dataset['train'].column_names and 'labels' not in tokenized_dataset['train'].column_names:
      tokenized_dataset = tokenized_dataset.map(
          lambda example: {'labels': example['label']},
          remove_columns=['label']
      )
  
  dataset_columns = ['input_ids', 'attention_mask', 'labels']
  # 'token_type_ids" exists if task requres multiple sentences
  if 'token_type_ids' in tokenized_dataset['train'].column_names:
    dataset_columns.append('token_type_ids')
  
  tokenized_dataset.set_format(type='torch', columns=dataset_columns)

  # batch sizes set for Roberta-base classification (Appendix D, Table 9)
  train_loader = DataLoader(tokenized_dataset["train"], batch_size=args.batch_size, shuffle=True)  
  val_loader = DataLoader(tokenized_dataset["validation"], batch_size=args.batch_size)

  return train_loader, val_loader


def train(args, model):
    """
    Trains LORA weights for model on given GLUE task.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"running on {device}")
    if '/' in args.model_name:
      save_model_name = args.model_name.split('/')[-1]
    else:
      save_model_name = args.model_name

    task_name = args.task_name
    lora = "lora" if args.lora else "full"
    # create dataloaders for given GLUE task
    train_loader, val_loader = create_dataloaders(args, task_name)
    logging.info(f"Created dataloaders")

    # similar to LoRA paper, use AdamW and Linear LR scheduler (without warmup ratio however)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    # potentially reload optimizer and scheduler state
    start_epoch = 0
    if args.resume_training != "None" and args.resume_optimizer != "None":
        logging.info(f"Loading optimizer state from {args.resume_optimizer}")
        checkpoint = torch.load(args.resume_optimizer, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    
    for e in tqdm(range(start_epoch, args.num_epochs), leave=True):
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
      
      # deberta-xxl takes forever to train, so save every epoch
      if "deberta" in args.model_name:
        torch.save(model.state_dict(), f"{args.save_dir}/{lora}_{save_model_name}-e{e}-{task_name}.pth")
        logging.info(f"Model saved to {args.save_dir}/{lora}_{save_model_name}-e{e}-{task_name}.pth")

        # save optimizer and scheduler state
        optimizer_save_path = f"{args.save_dir}/{lora}_{save_model_name}-e{e}-{task_name}_optimizer.pth"
        torch.save({
            'epoch': e,
            'optimizer': optimizer.state_dict(),
        }, optimizer_save_path)
      # save model only at beg and end (to save memory)
      elif e == 0 or (e+1) >= args.num_epochs:
        torch.save(model.state_dict(), f"{args.save_dir}/{lora}_{save_model_name}-e{e}-{task_name}.pth")
        logging.info(f"Model saved to {args.save_dir}/{lora}_{save_model_name}-e{e}-{task_name}.pth")

        # save optimizer and scheduler state
        optimizer_save_path = f"{args.save_dir}/{lora}_{save_model_name}-e{e}-{task_name}_optimizer.pth"
        torch.save({
            'epoch': e,
            'optimizer': optimizer.state_dict(),
        }, optimizer_save_path)
      
      avg_train_loss = train_running_loss / len(train_loader)
      metrics, avg_val_loss = val(model, val_loader, task_name, device)
      print(f"\nepoch {e}\ntraining loss: {avg_train_loss:.4f}\nval loss: {avg_val_loss:.4f}")
      print(f"val metrics: {metrics}\n")
      logging.info(f"\nepoch {e}\ntraining loss: {avg_train_loss:.4f}\nval loss: {avg_val_loss:.4f}")
      logging.info(f"val metrics: {metrics}\n")

def main(args):
    lora = "lora" if args.lora else "full"
    if '/' in args.model_name:
      save_model_name = args.model_name.split('/')[-1]
    else:
      save_model_name = args.model_name

    logging.basicConfig(
      filename=f'{args.save_dir}/{lora}_{save_model_name}_{args.task_name}.log',
      level=logging.INFO,
      filemode='a', # append
      format='%(asctime)s - %(levelname)s - %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S'
    )
    
      # log all the hyperparameters use in this run
    logging.info(f"Starting {args.model_name} on {args.task_name}, {lora}")
    logging.info(f"Config: num_epochs: {args.num_epochs}, learning rate: {args.lr}, batch size {args.batch_size}, max_seq_len {args.max_seq_len}")

    # run this baseline model on a binary task
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    if args.lora:
      inject_lora_to_kq_attn(args, model, args.rank, args.alpha)
      logging.info(f"Applying LoRA with r={args.rank}, alpha={args.alpha}")

    # NOTE sanity check to ensure LoRA is only training small subset of parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of trainable parameters: {trainable_params}")
    logging.info(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.4f}%")
    
    train(args, model)

if __name__ == "__main__":
    args = get_args()

    # check to make sure correct model and task is being run
    if args.model_name != "roberta-base" and args.model_name != "microsoft/deberta-v2-xxlarge":
       raise RuntimeError(f"This script does not support LoRA for {args.model_name}.")

    if args.task_name not in GLUE_BINARY_TASKS:
       raise RuntimeError(f"This script does not support training on {args.task_name}.")

    main(args)