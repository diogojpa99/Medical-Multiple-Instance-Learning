import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import numpy as np
import random

import os
from collections import Counter
from pathlib import Path

import torch
import torchvision.utils as vutils
import torch.serialization
import torch.distributed as dist
from torchinfo import summary


def Plot_TrainSet(trainset, args):
    
    output_dir = args.output_dir 
    
    # Create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create List of training images
    img_list = [trainset[i][0] for i in range(5)]
    labels_list = [trainset[i][1] for i in range(5)]
    
    # Create a grid of Images
    grid = vutils.make_grid(img_list, nrow=int(len(img_list)/2), normalize=True)

    # Convert the grid to a numpy array and transpose the dimensions
    grid_np = grid.permute(1, 2, 0)

    # Plot the grid using matplotlib
    plt.imshow(grid_np)
    plt.axis('off')
    
    if all( i == 0 for i in labels_list):
        plt.title('Melanoma training examples')
    elif all ( i == 1 for i in labels_list):
        plt.title('Non-melanoma training examples')
    else:
        plt.title('Mixed training examples')
    
    plt.savefig('train_images.png', bbox_inches='tight', pad_inches=0)
    
def Class_Weighting(train_set, val_set, device, args):
    
    # Check the distribution of the dataset
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

def plot_confusion_matrix(confusion_matrix, class_names, output_dir, args):
    
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(str(output_dir) + '/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.clf()
    
def plot_loss_curves(train_loss, test_loss, output_dir, args):
    """Plots training curves of a results dictionary.
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "test_loss": [...],
            }
    """
    epochs = range(len(train_loss))
    
    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot loss
    ax.plot(epochs, train_loss, label="Training Loss")
    ax.plot(epochs, test_loss, label="Validation Loss")
    ax.set_title("Losses")
    ax.set_xlabel("Epochs")
    ax.legend()

    # Save the figure
    plt.savefig(str(output_dir) + '/loss_curves.png')
    plt.clf()
    
def plot_loss_and_acc_curves(results_train, results_val, output_dir, args):
    """Plots training curves of a results dictionary.
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...]}
    """
    train_loss = results_train['loss']
    val_loss = results_val['loss']

    train_acc = results_train['acc']
    val_acc = results_val['acc']

    epochs = range(len(results_val['loss']))
    
    """ window_size = 1 # Adjust the window size as needed
    val_loss_smooth = np.convolve(val_loss, np.ones(window_size) / window_size, mode='valid')
    val_acc_smooth = np.convolve(val_acc, np.ones(window_size) / window_size, mode='valid')
    epochs_smooth = range(len(val_loss_smooth)) """
    #plt.figure(figsize=(15, 7))
    fig, axs = plt.subplots(2, 1)

    # Plot the original image
    axs[0].plot(epochs, train_loss, label="Train Loss")
    axs[0].plot(epochs, val_loss, label="Val. Loss")
    #axs[0].plot(epochs_smooth, val_loss_smooth, label="Val. Loss")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].legend()
    
    axs[1].plot(epochs, train_acc, label="Train Acc.")
    axs[1].plot(epochs, val_acc, label="Val Acc.")
    #axs[1].plot(epochs_smooth, val_acc_smooth, label="Val. Acc.")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].legend()
    
    plt.subplots_adjust(wspace=2, hspace=0.6)

    # Plot loss
    """ plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs_smooth, val_loss_smooth, label="Val. Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Acc.")
    plt.plot(epochs_smooth, val_acc_smooth, label="Val. Acc.")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend() """
    
    # Save the figure
    plt.savefig(str(output_dir) + '/loss_curves.png')
    plt.clf()
    
def Load_Pretrained_FeatureExtractor(path, model, args):
    
    if path.startswith('https:'):
        checkpoint = torch.hub.load_state_dict_from_url(path, 
                                                        map_location=torch.device('cpu'),
                                                        check_hash=True) 
    else:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))

    state_dict = model.state_dict()
        
    # Remove last layer
    """ for k in ['fc.weight', 'fc.bias']:
        if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k] """

    # Load the pre-trained weights into the model
    model.load_state_dict(checkpoint, strict=False)

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.
    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def configure_seed(
    seed: int = 42
):
  """Configure the random seed.
    Args:
        seed (int): The random seed. Default value is 42.
  """
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
      
def model_summary(model, args):
    # Print a summary using torchinfo (uncomment for actual output)
    summ = summary(model=model, 
                   input_size=(args.batch_size, 3, 224, 224), # (batch_size, color_channels, height, width)
                   col_names=["input_size", "output_size", "num_params", "trainable"],
                   col_width=20,
                   row_settings=["var_names"])   
    return summ    

def Load_Pretrained_MIL_Model(path, model, args):
    # Load the pretrained Mil model
    if path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
        
    checkpoint_keys = set(checkpoint['model'].keys())
    model_keys = set(model.state_dict().keys())
    unmatched_keys = checkpoint_keys.symmetric_difference(model_keys)
    for k in unmatched_keys:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint['model'][k]
            
    model.load_state_dict(checkpoint['model'])