"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from timm.utils import ModelEma

from typing import Optional
from collections import Counter

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
    
import mil

import Breast_Scripts.engine as breast_engine
import Skin_Scripts.engine as skin_engine

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
    
    if args.dataset_type == 'Skin':
        return skin_engine.train_step(model, dataloader, criterion, optimizer, device, epoch, loss_scaler, max_norm, lr_scheduler, wandb, model_ema, args)
    elif args.dataset_type == 'Breast':
        return breast_engine.train_step(model, dataloader, criterion, optimizer, device, epoch, loss_scaler, max_norm, lr_scheduler, wandb, model_ema, args)
    else:
        raise ValueError(f"Dataset {args.dataset_type} not supported for training.")
    
@torch.no_grad()
def evaluation(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               device: torch.device,
               epoch: int,
               wandb=print,
               args=None):
     
     if args.dataset_type == 'Skin':
          return skin_engine.evaluation(model, dataloader, criterion, device, epoch, wandb, args)
     elif args.dataset_type == 'Breast':
          return breast_engine.evaluation(model, dataloader, criterion, device, epoch, wandb, args)
     else:
          raise ValueError(f"Dataset {args.dataset_type} not supported for evaluation.")

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
            
        if not args.pos_encoding_flag and args.feature_extractor in mil.vits_backbones:
            for i, (param_name, param) in enumerate(model.patch_extractor.named_parameters()):
                if param_name == 'pos_embed':
                    param.requires_grad = False
                    break 
                
        flag = False
        
    return flag

def Class_Weighting(train_set:torch.utils.data.Dataset, 
                    val_set:torch.utils.data.Dataset, 
                    device:str='cuda:0', 
                    args=None):
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
            
    if args.class_weights == 'median':
        class_weights = torch.Tensor([(len(train_set)/x) for x in train_dist.values()]).to(device)
    else:                   
        class_weights = torch.Tensor(compute_class_weight(class_weight=args.class_weights, 
                                                        classes=np.unique(train_set.targets), y=train_set.targets)).to(device)

    print(f"Classes map: {train_set.class_to_idx}"); print(f"Train distribution: {train_dist}"); print(f"Val distribution: {val_dist}")
    print(f"Class weights: {class_weights}\n")
    
    return class_weights