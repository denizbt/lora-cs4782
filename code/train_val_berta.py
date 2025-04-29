""""
Contains general purpose train and val PyTorch loops to train Roberta-base and Deberta-XXL models on GLUE tasks.

Used in roberta_lora.py, deberta_lora.py etc.
"""
import torch
import sys
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import pearsonr
import logging
# from tqdm import tqdm

GLUE_BINARY_TASKS = ["sst2", "mrpc", "cola", "qnli", "qqp", "rte"]

def train(args, model, device, train_loader, val_loader, optimizer, scheduler, start_epoch, num_epochs):
    """
    Unified training loop for LORA weights for transformer models on GLUE tasks.
    """ 
    # model name (w/o / to avoid path complications)
    if '/' in args.model_name:
        save_model_name = args.model_name.split('/')[-1]
    else:
        save_model_name = args.model_name
    
    # Training loop
    for e in range(start_epoch, num_epochs):
        model.train()
        train_running_loss = 0
        
        i = 0
        # for batch in tqdm(train_loader, desc=f"train (e {e+1})", leave=True):
        for batch in train_loader:
            optimizer.zero_grad()
            
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_running_loss += loss.item()
            
            # print batch every 50 times
            if i % 50 == 0:
                print(f"Epoch {e+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                sys.stdout.flush()  # Force output display
            i += 1
        
        # Save model and optimizer after each epoch
        model_save_path = f"{args.save_dir}/{save_model_name}-e{e+1}-{args.task_name}.pth"
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Model saved to {model_save_path}")
        
        optimizer_save_path = f"{args.save_dir}/{save_model_name}-e{e+1}-{args.task_name}_optimizer.pth"
        torch.save({
            'epoch': e,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, optimizer_save_path)
        
        avg_train_loss = train_running_loss / len(train_loader)
        
        # Special handling for MNLI which has two validation sets
        if args.task_name == "mnli":
            print(f"Epoch {e+1}")
            print(f"Training loss: {avg_train_loss:.4f}")
            logging.info(f"Epoch {e+1}")
            logging.info(f"Training loss: {avg_train_loss:.4f}")
            
            # Evaluate on matched validation set
            metrics_matched, avg_val_loss_matched = val(
                model, val_loader["matched"], args.task_name, device
            )
            
            # Evaluate on mismatched validation set
            metrics_mismatched, avg_val_loss_mismatched = val(
                model, val_loader["mismatched"], args.task_name, device
            )
            
            # Calculate weighted average metrics
            matched_size = len(val_loader["matched"].dataset)
            mismatched_size = len(val_loader["mismatched"].dataset)
            
            avg_val_loss = (avg_val_loss_matched * matched_size + 
                           avg_val_loss_mismatched * mismatched_size) / (matched_size + mismatched_size)
            
            # Calculate overall accuracy
            matched_correct = metrics_matched["accuracy"] * matched_size
            mismatched_correct = metrics_mismatched["accuracy"] * mismatched_size
            overall_accuracy = (matched_correct + mismatched_correct) / (matched_size + mismatched_size)
            
            print(f"Val loss: {avg_val_loss:.4f}")
            print(f"Overall accuracy: {overall_accuracy:.4f}")
            logging.info(f"Val loss: {avg_val_loss:.4f}")
            logging.info(f"Overall accuracy: {overall_accuracy:.4f}")
        else:
            metrics, avg_val_loss = val(model, val_loader, args.task_name, device)
            print(f"\nEpoch {e+1}")
            print(f"Training loss: {avg_train_loss:.4f}")
            print(f"Val loss: {avg_val_loss:.4f}")
            logging.info(f"\nEpoch {e+1}")
            logging.info(f"Training loss: {avg_train_loss:.4f}")
            logging.info(f"Val loss: {avg_val_loss:.4f}")
            logging.info(f"Val metrics: {metrics}\n")
        
        # Force print statements to display
        sys.stdout.flush()

def val(model, val_loader, task_name, device):
    model.eval()
    val_running_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            logits = outputs.logits
            if task_name in GLUE_BINARY_TASKS or task_name == "mnli":
                preds = torch.argmax(logits, dim=-1)
            else:
                raise RuntimeError(f"{task_name} not supported in validation loop.")
            
            # Keep everything on GPU until the end
            all_preds.append(preds)
            all_labels.append(batch['labels'])
            
            loss = outputs.loss
            val_running_loss += loss.item()

            # print progress every 50 steps
            if i % 50 == 0:
              print(f"Validation batch {i}/{len(val_loader)}")
              sys.stdout.flush()
    
    # concatenate results at the end
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # move to CPU to compute_metrics
    metrics = compute_metrics(all_labels.cpu().numpy(), all_preds.cpu().numpy(), task_name)
    avg_val_loss = val_running_loss / len(val_loader)
    return metrics, avg_val_loss

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