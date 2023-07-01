import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import mil

#### Utils Functions

def ShowVis(activation_map, img):
    
    heatmap = torch.nn.functional.interpolate(activation_map, scale_factor=(224//14), mode='bilinear', align_corners=True)  #14->224
    heatmap = heatmap.reshape(224, 224).data.cpu().numpy() 
    heatmap = cv2.applyColorMap(np.uint8(heatmap* 255), cv2.COLORMAP_JET)        
    heatmap = np.float32(heatmap) / 255
    
    cam = heatmap*0.9 + np.float32(img)
    cam = cam / np.max(cam)
    
    vis =  np.uint8(255 * cam)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    return vis

def ShowKeyPatch(patch_prob, image):
    
    insts_map = patch_prob[:,:,0].squeeze(0) # Get the instance scores for the predicted class

    key_patch_idx = int(torch.argmax(insts_map)) # Get the index of the key patch
    instances_map = insts_map.clone().detach() # Clone the instance map

    instances_map.zero_() # Set all the patches to 0
    instances_map[key_patch_idx] = 1 # Set the key patch to 1
    
    instances_map = instances_map.reshape(14, 14).data.cpu().numpy() # Reshape to (14, 14)
    key_patch_idx = np.argwhere(instances_map == 1) # Get the index of the key patch
    
    top_left_x = key_patch_idx[0][1] * (224 // 14); top_left_y = key_patch_idx[0][0] * (224 // 14)
    bottom_right_x = top_left_x + (224 // 14); bottom_right_y = top_left_y + (224 // 14)
    
    key_patch = np.copy(np.array(image))
    cv2.rectangle(key_patch, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
    
    return key_patch

def ProcessMaskedPatchProbs(patch_prob, mask):
    
    pooled_mask=mil.Mask_Setup(mask)
    masked_probs=patch_prob.permute(0, 2, 1) * pooled_mask
    return masked_probs.permute(0, 2, 1)

def Collate_Binary(batch):
    
    # (1) Separate the batch into MEL (0) and NV (1) classes
    mel = [item for item in batch if item[1]==0]
    nv = [item for item in batch if item[1]==1]
    
    # (2) Determine the desired number of instances for each class in the batch
    instances_per_class = len(batch) // 2
    
    # Slice the batches to have an equal number of instances for each class
    mel = mel[:instances_per_class]
    nv = nv[:instances_per_class]
    
    return mel+nv
  
def VisualizationLoader_Binary(val_set:torch.utils.data.Dataset, args=None):
    
    # (1) Obtain the idxs of the melanoma and nevus samples    
    mel_idx=[]; nv_idx=[]
    for i, (_, label,_ ,_) in enumerate(val_set):
        if label==0: 
            mel_idx.append(i)
        elif label==1: 
            nv_idx.append(i)
        if i==len(val_set)-1:
            break

    # (2) Shuffle the indices randomly
    """ random.shuffle(mel_idx)
    random.shuffle(nv_idx) """

    # Select an equal number of indices for each class
    num_samples_per_class = min(len(mel_idx), len(nv_idx))
    mel_idx = mel_idx[:num_samples_per_class]
    nv_idx = nv_idx[:num_samples_per_class]
    
    # (3) Create Subset objects for each class
    mel_subset = Subset(val_set, mel_idx)
    nv_subset = Subset(val_set, nv_idx)
    
    # (4) Create separate DataLoaders for each class subset
    mel_loader = DataLoader(mel_subset, batch_size= (args.visualize_num_images//2), shuffle=True, collate_fn=Collate_Binary)
    nv_loader = DataLoader(nv_subset, batch_size=(args.visualize_num_images//2), shuffle=True, collate_fn=Collate_Binary)
    
    return DataLoader(ConcatDataset([mel_loader.dataset, nv_loader.dataset]), batch_size=args.visualize_num_images, shuffle=True)

def Get_Predicted_Class(label, predicted_class):
    if label == 0:
        prefix = 'Mel'
    elif label == 1:
        prefix = 'NV'
    else:
        prefix = ''
    
    pred_class = 'MEL' if predicted_class == 0 else 'NV'
    
    return f'{prefix} | Pred: {pred_class}'

##### Grad-CAM Function

def Grad_CAM(input, model, prediction, label, img):
    """ 
    Function inspired by: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    
    Note: In the beginning, when we set the prediction as backward we can do it in two ways:
    
    1) prediction[:, label].backward(retain_graph=True)
    2) One hot technique
    
    Both methods are equivalent. At least I tested both and the results are the same.
    
    """   
     
    model.zero_grad() 
    prediction[:,label].backward(retain_graph=True)
    gradients = model.get_activations_gradient()
    activations = model.get_activations(input).detach()
    
    grad_activations = activations * gradients
    
    heatmap = torch.sum(grad_activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap) 

    vis = ShowVis(heatmap.unsqueeze(0).unsqueeze(0), img)
        
    return vis
  
####  Binary Visualization Functions
        
def Visualize_Activation_Instance_Binary(model: torch.nn.Module, 
                                        dataloader:torch.utils.data.DataLoader,
                                        device: torch.device, 
                                        outputdir=None, 
                                        args=None):

    fig, axs = plt.subplots(4, args.visualize_num_images, figsize=(4*(args.visualize_num_images), 17))
    
    mean = IMAGENET_DEFAULT_MEAN; std = IMAGENET_DEFAULT_STD
    reverse_transform = transforms.Compose([
        transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]), (1.0 / std[0], 1.0 / std[1], 1.0 / std[2])),
        transforms.ToPILImage()
    ])

    for j, (inputs, labels, idxs, masks) in enumerate(dataloader):
        
        inputs, labels, masks = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                
        for i in range(args.visualize_num_images):
            
            input=inputs[i].unsqueeze(0)
            mask=masks[i].unsqueeze(0)
            image=reverse_transform(inputs[i])
                            
            # (3) Set model to eval mode
            model.eval()
            
            # (4) Obtain the bag scores
            bag_prob = model(input, None) if not args.mask_val else model(input, mask)
            predicted_class = int(torch.argmax(bag_prob))
                        
            # (5) obtain the instance softmax scores
            patch_prob = model.get_patch_probs()
            if args.pooling_type == 'mask_max' or args.pooling_type == 'mask_avg':
                patch_prob = ProcessMaskedPatchProbs(patch_prob, mask)
            
            # (6) Transform to (batch_size, num_classes, 14, 14)
            patch_prob_map = patch_prob.permute(0, 2, 1)
            patch_prob_map = patch_prob_map.reshape(1, 2, 14, 14)
   
            # (7) Normalize the input image
            img = input.permute(0, 2, 3, 1).squeeze(0).data.cpu().numpy()
            img = (img - np.min(img)) / (np.max(img) - np.min(img))    
            
            if args.pooling_type == 'max' or args.pooling_type == 'mask_max':
                key_patch = ShowKeyPatch(patch_prob, image) # (8) Obtain the key patch for melanoma class

            # (9) Obtain the class probabilities for 'MEL' heatmap 
            activation_map = patch_prob_map[:,0, :, :].unsqueeze(0)
            vis = ShowVis(activation_map, img)
            
            # (10) Grad CAM for 'MEl' class
            grad_cam_mel = Grad_CAM(input, model, bag_prob, 0, img)
            grad_cam_nv = Grad_CAM(input, model, bag_prob, 1, img)
            
            # Plot the original image
            axs[0, i].imshow(image)
            axs[0, i].set_title(Get_Predicted_Class(labels[i], predicted_class), fontsize=16)
            axs[0, i].axis('off');

            # Plot Probability Heatmap
            axs[1, i].imshow(vis)
            axs[1, i].set_title("'MEL' Probability Heatmap")
            axs[1, i].axis('off');
            
            # Plot Grad-CAM
            axs[2, i].imshow(grad_cam_mel)
            axs[2, i].set_title("Grad-CAM [MEL]")
            axs[2, i].axis('off');

            # Plot the key patch or the "attention" map
            if args.pooling_type == 'max' or args.pooling_type == 'mask_max':
                axs[3, i].imshow(key_patch, cmap='jet')
                axs[3, i].set_title("'MEL' Key Patch")
            else:
                axs[3, i].imshow(grad_cam_nv)
                axs[3, i].set_title("Grad-CAM [NV]")
            axs[3, i].axis('off');
                       
        title = f"| MIL Class Activation Maps ({args.dataset}) | MIL Type: {args.mil_type} | Pooling Type: {args.pooling_type} |"
        plt.suptitle(title, fontsize=20)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(str(outputdir) + f'/MIL-{args.mil_type}-{args.pooling_type}-{args.dataset}-Class_Activations-zBatch_{j}.jpg', dpi=300, bbox_inches='tight')  
        
        if j == (args.vis_num-1):
            break

def Visualize_Activation_Embedding_Binary(model: torch.nn.Module, 
                                        dataloader:torch.utils.data.DataLoader, 
                                        device: torch.device,
                                        outputdir=None, 
                                        args=None):

    fig, axs = plt.subplots(3, args.visualize_num_images, figsize=(4*(args.visualize_num_images), 13))
    
    mean = IMAGENET_DEFAULT_MEAN; std = IMAGENET_DEFAULT_STD
    reverse_transform = transforms.Compose([
        transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]), (1.0 / std[0], 1.0 / std[1], 1.0 / std[2])),
        transforms.ToPILImage()
    ])

    for j, (inputs, labels, idxs, masks) in enumerate(dataloader):

        inputs, labels, masks = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        
        for i in range(args.visualize_num_images):
                        
            input=inputs[i].unsqueeze(0)
            mask=masks[i].unsqueeze(0)
            image=reverse_transform(inputs[i])
                            
            # (3) Set model to eval mode
            model.eval()
            
            # (4) Obtain the bag scores
            bag_prob = model(input, None) if not args.mask_val else model(input, mask)
            predicted_class = int(torch.argmax(bag_prob))
                                            
            # (7) Normalize the input image
            img = input.permute(0, 2, 3, 1).squeeze(0).data.cpu().numpy()
            img = (img - np.min(img)) / (np.max(img) - np.min(img))    
            
            # Grad CAM
            grad_cam_mel = Grad_CAM(input,model,bag_prob,0, img)
            grad_cam_nv = Grad_CAM(input,model,bag_prob,1,img)
            
            # Plot the original image
            axs[0, i].imshow(image)
            axs[0, i].set_title(Get_Predicted_Class(labels[i], predicted_class), fontsize=16)
            axs[0, i].axis('off');
            
            # Plot the Grand-CAM for the MEL class
            axs[1, i].imshow(grad_cam_mel)
            axs[1, i].set_title("Grad-CAM [MEL]")
            axs[1, i].axis('off');
            
            # Plot the Grand-CAM for the NV class
            axs[2, i].imshow(grad_cam_nv)
            axs[2, i].set_title("Grad-CAM [NV]")
            axs[2, i].axis('off');
           
        title = f"| MIL Class Activation Maps ({args.dataset}) | MIL Type: {args.mil_type} | Pooling Type: {args.pooling_type} |"
        plt.suptitle(title, fontsize=20)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(str(outputdir) + f'/MIL-{args.mil_type}-{args.pooling_type}-{args.dataset}-Class_Activations-zBatch_{j}.jpg', dpi=300, bbox_inches='tight')  
        
        if j == (args.vis_num-1):
            break

####  Multiclass Visualization Functions  