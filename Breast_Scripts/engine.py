"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from timm.utils import ModelEma

from typing import Optional

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, \
    balanced_accuracy_score
    
import utils
    
def train_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                criterion: torch.nn.Module, 
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch: int,
                loss_scaler,
                max_norm: float=0.0,
                lr_scheduler=None,
                wandb=print,
                model_ema: Optional[ModelEma] = None,
                gradient_tracker=None,
                args = None):
        
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    train_stats = {}
    lr_num_updates = epoch * len(dataloader)
    preds = []; targs = []

    # Loop through data loader data batches
    for batch_idx, (input, target) in enumerate(dataloader):
        
        # Send data to device
        input, target = input.to(device), target.to(device)
        
        # 1. Clear gradients
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            bag_prob = model(input, None) # 2.Forward pass
            loss = criterion(bag_prob, target) # 3. Compute and accumulate loss
        
        train_loss += loss.item() 
        
        if loss_scaler is not None:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order # this attribute is added by timm on one optimizer (adahessian)
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward() # 3. Backward pass
            # if max_norm is not None:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) # 4. Clip gradients
            optimizer.step() # 5. Update weights
            
        if args.print_grad_stats:
            gradient_tracker.update_stats(model)

        # Update LR Scheduler
        if not args.cosine_one_cycle and lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=lr_num_updates)
            
        # Update Model Ema
        if model_ema is not None:
            if device == 'cuda:0' or device == 'cuda:1':
                torch.cuda.synchronize()
            model_ema.update(model)

        # Calculate and accumulate accuracy metric across all batches
        predictions = torch.argmax(bag_prob, dim=1)
        train_acc += (predictions == target).sum().item()/len(predictions)
        
        preds.append(predictions.cpu().numpy()); targs.append(target.cpu().numpy())
        
    if args.print_grad_stats:
        gradient_tracker.print_epoch_stats(epoch)
        gradient_tracker.reset_stats()  
            
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader); train_acc = train_acc / len(dataloader)

    train_stats['train_loss'] = train_loss
    train_stats['train_acc'] = train_acc
    train_stats['train_lr'] = optimizer.param_groups[0]['lr']
    
    if wandb!=print:
        wandb.log({"Train Loss":train_loss}, step=epoch)
        wandb.log({"Train Accuracy":train_acc},step=epoch)
        wandb.log({"Train LR":optimizer.param_groups[0]['lr']},step=epoch)
        
    # Compute Metrics
    preds=np.concatenate(preds); targs=np.concatenate(targs)
    train_stats['confusion_matrix'], train_stats['f1_score'] = confusion_matrix(targs, preds), f1_score(targs, preds, average=None) 
    train_stats['precision'], train_stats['recall'] = precision_score(targs, preds, average=None), recall_score(targs, preds, average=None)
    train_stats['bacc'] = balanced_accuracy_score(targs, preds)
    train_stats['acc1'], train_stats['loss'] = train_acc, train_loss
    
    return train_stats

@torch.no_grad()
def evaluation(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                criterion: torch.nn.Module, 
                device: torch.device,
                epoch: int,
                wandb=print,
                args=None):
    
    # Switch to evaluation mode
    model.eval()
    
    preds = []
    targs = []
    test_loss, test_acc = 0, 0
    results = {}
    count_tokens = 0
    
    for input, target in dataloader:
        
        input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
        # Compute output
        bag_prob = model(input, None) 
        loss = criterion(bag_prob, target)
        test_loss += loss.item()
    
        # Calculate and accumulate accuracy
        predictions = torch.argmax(bag_prob, dim=1)
        test_acc += ((predictions == target).sum().item()/len(predictions))
        
        preds.append(predictions.cpu().numpy()); targs.append(target.cpu().numpy())
        count_tokens += (model.count_tokens/len(predictions))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss/len(dataloader); test_acc = test_acc/len(dataloader)
    count_tokens = count_tokens/len(dataloader)

    if wandb!=print:
        wandb.log({"Val Loss":test_loss},step=epoch)
        wandb.log({"Val Accuracy":test_acc},step=epoch)
        
    # Compute Metrics
    preds=np.concatenate(preds); targs=np.concatenate(targs)
    results['confusion_matrix'], results['f1_score'] = confusion_matrix(targs, preds), f1_score(targs, preds, average=None) 
    results['precision'], results['recall'] = precision_score(targs, preds, average=None), recall_score(targs, preds, average=None)
    results['bacc'] = balanced_accuracy_score(targs, preds)
    results['acc1'], results['loss'] = accuracy_score(targs, preds), test_loss
    results['count_tokens'] = count_tokens

    return results
