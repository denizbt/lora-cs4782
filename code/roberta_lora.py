from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

import sys
# from tqdm import tqdm
import argparse
import logging

# importing custom functions
from lora_layers import inject_lora_to_kq_attn
from train_val_berta import train, GLUE_BINARY_TASKS

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="roberta-base")
    parser.add_argument("--task-name", type=str, default="sst2")
    parser.add_argument("--resume-training", type=str, default="None", help="If not 'None', contains path to .pth from which to resume training.")
    parser.add_argument("--resume-optimizer", type=str, default="None", help="If not 'None', contains path to optimizer state from which to resume training.")
    parser.add_argument("--save-dir", type=str, default="../results")

    return parser.parse_args()

# define batch_size for GLUE tasks (if not in dictionary, batch_size is 16)
GLUE_TASK_BATCH = {
    "cola": 32,
    "qnli": 32,
    "rte": 32
}

# define LR for each GLUE task
GLUE_TASK_LR = {"mnli": 5e-4, "stsb": 4e-4, "sst2": 5e-4, "mrpc": 4e-4, "cola": 4e-4, "qnli": 4e-4, "qqp": 5e-4, "rte": 5e-4}

# define the dataset keys we need to tokenize for each GLUE task
# helper dictionary used in create_dataloaders() function
GLUE_TASK_TOKENIZE = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "cola": ("sentence", None),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis")
}

GLUE_NUM_EPOCHS ={"mnli": 30, "stsb": 40, "sst2": 60, "mrpc": 30, "cola": 80, "qnli": 25, "qqp": 25, "rte": 80}

def create_roberta_dataloaders(args, task_name, max_seq_length=512):
  tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
  dataset = load_dataset("glue", task_name)

  def tokenize_function(example):
    # some GLUE tasks are single sentence, some are sentence pair cls
    k1, k2 = GLUE_TASK_TOKENIZE[task_name]
    if k2 is None:
        return tokenizer(
            example[k1],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length
        )
    else:
        return tokenizer(
            example[k1],
            example[k2],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length
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
  batch_size = GLUE_TASK_BATCH.get(task_name, 16)
  train_loader = DataLoader(tokenized_dataset["train"], batch_size=batch_size, shuffle=True)

  if task_name == "mnli":
    val_loader = {
        "matched": DataLoader(tokenized_dataset["validation_matched"], batch_size=batch_size),
        "mismatched": DataLoader(tokenized_dataset["validation_mismatched"], batch_size=batch_size)
    }
  else:   
    val_loader = DataLoader(tokenized_dataset["validation"], batch_size=batch_size)

  return train_loader, val_loader


def train_roberta(args, model):
    """
    Trains LORA weights for roberta-base model on given GLUE task.
    
    Pre-condition: model have be roberta-base for sequence classification
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"running on {device}")

    task_name = args.task_name
    # create dataloaders for given GLUE task
    train_loader, val_loader = create_roberta_dataloaders(args, task_name)
    logging.info(f"created dataloaders")

    # based on Appendix D, use AdamW
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=GLUE_TASK_LR[task_name], weight_decay=0.01)

    # use a warmup ratio 0.06 (Appendix D, Table 9)
    num_train_steps = len(train_loader) * GLUE_NUM_EPOCHS[task_name]
    warmup_steps = int(num_train_steps * 0.06) 

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
    
    train(args, model, device, train_loader, val_loader, optimizer, scheduler, start_epoch, GLUE_NUM_EPOCHS[task_name])

def main(args):
    logging.basicConfig(
      filename=f'{args.save_dir}/{args.model_name}_{args.task_name}.log',
      level=logging.INFO,
      filemode='a', # append
      format='%(asctime)s - %(levelname)s - %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S',
      force=True
    )
    
    if args.task_name in GLUE_BINARY_TASKS:
      model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    elif args.task_name == "mnli":
      model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
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
    
    train_roberta(args, model)

if __name__ == "__main__":
    args = get_args()
    if args.model_name != "roberta-base":
       raise RuntimeError(f"This script does not support LoRA for {args.model_name}.")

    main(args)