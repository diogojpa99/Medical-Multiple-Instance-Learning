"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from timm.utils import ModelEma

from typing import Optional
from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, \
    balanced_accuracy_score
      
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
               args = None):
        
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    train_stats = {}
    lr_num_updates = epoch * len(dataloader)
    preds = []
    targs = []

    # Loop through data loader data batches
    for batch_idx, (input, target, input_idx, mask) in enumerate(dataloader):
        
        # Send data to device
        input, target, mask = input.to(device), target.to(device), mask.to(device)
        
        # 1. Clear gradients
        optimizer.zero_grad()

        #with torch.cuda.amp.autocast():
        bag_prob = model(input, mask) # 2.Forward pass
        loss = criterion(bag_prob, target) # 3. Compute and accumulate loss
        
        train_loss += loss.item() 
        
        if loss_scaler is not None:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward() # 3. Backward pass
            optimizer.step() # 5. Update weights

        # Update LR Scheduler
        if not args.cosine_one_cycle:
            lr_scheduler.step_update(num_updates=lr_num_updates)
            
        # Update Model Ema
        if model_ema is not None:
            if device == 'cuda:0' or device == 'cuda:1':
                torch.cuda.synchronize()
            model_ema.update(model)

        # Calculate and accumulate accuracy metric across all batches
        predictions = torch.argmax(bag_prob, dim=1)
        train_acc += (predictions == target).sum().item()/len(predictions)
        
        preds.append(predictions.cpu().numpy())
        targs.append(target.cpu().numpy())
        
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    train_stats['train_loss'] = train_loss
    train_stats['train_acc'] = train_acc
    train_stats['train_lr'] = optimizer.param_groups[0]['lr']
    
    if wandb!=print:
        wandb.log({"Train Loss":train_loss}, step=epoch)
        wandb.log({"Train Accuracy":train_acc},step=epoch)
        wandb.log({"Train LR":optimizer.param_groups[0]['lr']},step=epoch)
        
    # Compute Metrics
    preds=np.concatenate(preds)
    targs=np.concatenate(targs)
    train_stats['confusion_matrix'], train_stats['f1_score'] = confusion_matrix(targs, preds), f1_score(targs, preds, average=None) 
    train_stats['precision'], train_stats['recall'] = precision_score(targs, preds, average=None), recall_score(targs, preds, average=None)
    train_stats['bacc'] = balanced_accuracy_score(targs, preds)
    train_stats['acc1'], train_stats['loss'] = train_acc, train_loss
    
    return train_stats

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
    
    for input, target , input_idxs, mask in dataloader:
        
        input, target, mask = input.to(device, non_blocking=True), target.to(device, non_blocking=True), mask.to(device, non_blocking=True)

        # Compute output
        with torch.no_grad():
            
            if not args.mask_val:
                bag_prob = model(input, None)
            else:
                bag_prob = model(input, mask)
                
            loss = criterion(bag_prob, target)
            test_loss += loss.item()
    
        # Calculate and accumulate accuracy
        predictions = torch.argmax(bag_prob, dim=1)
        test_acc += ((predictions == target).sum().item()/len(predictions))
        
        preds.append(predictions.cpu().numpy())
        targs.append(target.cpu().numpy())

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)

    if wandb!=print:
        wandb.log({"Val Loss":test_loss},step=epoch)
        wandb.log({"Val Accuracy":test_acc},step=epoch)
        
    # Compute Metrics
    preds=np.concatenate(preds)
    targs=np.concatenate(targs)
    results['confusion_matrix'], results['f1_score'] = confusion_matrix(targs, preds), f1_score(targs, preds, average=None) 
    results['precision'], results['recall'] = precision_score(targs, preds, average=None), recall_score(targs, preds, average=None)
    results['bacc'] = balanced_accuracy_score(targs, preds)
    results['acc1'], results['loss'] = test_acc, test_loss

    return results

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            
        elif score < self.best_score + self.delta:
            # If we don't have an improvement, increase the counter 
            self.counter += 1
            #self.trace_func(f'\tEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # If we have an imporvement, save the model
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            #self.trace_func(f'\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model as checkpoint.pth')
            self.trace_func(f'\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            
        #torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_patch_extractor(model: torch.nn.Module, 
                          current_epoch: int, 
                          warmup_epochs: int, 
                          flag: bool, 
                          args
):    
    """ Function to train the patch extractor for the first warmup_epochs epochs.
    Returns:
        bool: flag that defines if the patch extractor is trainable or not.
    """
    if current_epoch < warmup_epochs:
        for param in model.patch_extractor.parameters():
            param.requires_grad = False
        flag = True
        
    elif current_epoch >= warmup_epochs:
        for param in model.patch_extractor.parameters():
            param.requires_grad = True
        flag = False
        
    return flag

def Class_Weighting(train_set, val_set, device, args):
    """ Class weighting for imbalanced datasets.

    Args:
        train_set (torch.utils.data.Dataset): Training set.
        val_set (torch.utils.data.Dataset): Validation set.
        device (str): Device to use.
        args (*args): Arguments.

    Returns:
        torch.Tensor: Class weights. (shape: (2,))
    """
    
    train_dist = dict(Counter(train_set.targets))
    val_dist = dict(Counter(val_set.targets))
    
    train_dist['MEL'] = train_dist.pop(0)
    train_dist['NV'] = train_dist.pop(1)
    val_dist['MEL'] = val_dist.pop(0)
    val_dist['NV'] = val_dist.pop(1)
    
    n_train_samples = len(train_set)
    
    print(f"Classes: {train_set.classes}\n")
    print(f"Classes map: {train_set.class_to_idx}\n")
    print(f"Train distribution: {train_dist}\n")
    print(f"Val distribution: {val_dist}\n")
    
    if args.class_weights:
        if args.class_weights_type == 'Median':
            class_weight = torch.Tensor([n_train_samples/train_dist['MEL'], 
                                         n_train_samples/ train_dist['NV']]).to(device)
        elif args.class_weights_type == 'Manual':                   
            class_weight = torch.Tensor([n_train_samples/(2*train_dist['MEL']), 
                                         n_train_samples/(2*train_dist['NV'])]).to(device)
    else: 
        class_weight = None
    
    print(f"Class weights: {class_weight}\n")
    
    return class_weight