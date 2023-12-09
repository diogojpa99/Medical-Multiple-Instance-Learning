import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import numpy as np
import random
import io
import os
from pathlib import Path

import torch
import torchvision.utils as vutils
import torch.serialization
import torch.distributed as dist
from torchinfo import summary
import torch.nn.functional as F

import mil

import Feature_Extractors.DenseNet as densenet

def Adjust_topk(topk, args):
    """The top-k hyperparmeter is given as percentage. This function transforms the top-k into the number of patches to be selected.

    Args:
        topk (float): k (in % over the total number of patches N) in top-k average pooling operator
        args (*args): args
    """
    
    N=(args.input_size // args.patch_size) ** 2    
    if args.feature_extractor in mil.cnns_backbones or mil.deits_backbones:
        args.topk = int(N*(topk/100))
    elif args.feature_extractor in mil.evits_backbones:
        n_patches = int(N*(args.base_keep_rate**3)) # Assuming the default token removal layer composition in EViT: (3,6,9)
        args.topk = n_patches*(topk/100)
    else:
        raise ValueError('Feature extractor not supported... Yet!')

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

    # Compare the keys of the checkpoint and the model
    if args.feature_extractor in mil.vits_backbones:
        checkpoint = checkpoint['model']
        if args.pos_encoding_flag:
            Load_Pretrained_ViT_Interpolate_Pos_Embed(model, checkpoint)
        
    elif args.feature_extractor == 'densenet169.tv_in1k':
        checkpoint = densenet._filter_torchvision_pretrained(checkpoint)

    if len(set(state_dict.keys()).intersection(set(checkpoint.keys())))==0:
        raise RuntimeError("No shared keys between checkpoint and model.")

    # Load the pre-trained weights into the model
    model.load_state_dict(checkpoint, strict=False)
    
def Load_Pretrained_ViT_Interpolate_Pos_Embed(model, checkpoint_model):
    
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5) # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5) # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens] # only the position tokens are interpolated
    
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed
    
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
                
    checkpoint_keys = set(checkpoint['model'].keys()); model_keys = set(model.state_dict().keys())
    unmatched_keys = checkpoint_keys.symmetric_difference(model_keys)
    for k in unmatched_keys:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint['model'][k]
            
    model.load_state_dict(checkpoint['model'], strict=True)

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)