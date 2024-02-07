import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import mil
import visualization

def Grad_CAM_Eval(model: torch.nn.Module,
                  input: torch.Tensor,
                  prediction: torch.Tensor, 
                  label:int=0) -> torch.Tensor:
    """This function computes the GradCam heatmap for the given input image.
    Note thar computing the loss using a Sum or a Mean operation may produce different gradients.
    Since the gradients are computed taking into account the value of the loss, I think that using the 
    Sum operation or the Mean operation may produce different gradients... Investigate this!

    Args:
        model (torch.nn.Module): Weights of the model to be evaluated
        input (torch.Tensor): Batch of input image. Shape: (batch_size, channels, height, width)
        prediction (torch.Tensor): Prediction of the model. Shape: (batch_size, num_classes)
        label (int, optional): Class to produce the GradCam heatmap. Defaults to 0.

    Returns:
        torch.Tensor: GradCam heatmap normalized (min-max normalization)
    """
    model.zero_grad() 
    
    torch.sum(prediction[:,label]).backward(retain_graph=True)
    #torch.mean(prediction[:,label]).backward(retain_graph=True)

    gradients = model.get_activations_gradient()
    activations = model.get_activations(input).detach()
    
    grad_activations = activations * gradients
    
    heatmap = torch.sum(grad_activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap = heatmap.reshape(heatmap.size()[0], int((heatmap.size()[-1])**2)) # Output shape: (batch_size, N)
    heatmap_max,_ = torch.max(heatmap, dim=1)
    heatmap_max[heatmap_max == 0] = 1
    heatmap_norm = heatmap / heatmap_max.view(-1,1) 
    heatmap_norm = heatmap_norm.reshape(heatmap.size()[0], int(math.sqrt(heatmap.size()[-1])), int(math.sqrt(heatmap.size()[-1]))) # Output shape: (batch_size, 14, 14)      
            
    return heatmap_norm

def Compute_ROIs(model: torch.nn.Module,
                 input: torch.Tensor,
                 mask: torch.Tensor,
                 args=None) -> torch.Tensor:
    """This function computes the ROIs for the different MIL pooling operators.
    This function is analogous to the functions used for visualization of the ROIs.
    
    In this function we will compute the ROIs for the Instance and Embedding-level approaches
    We will use the ROIs identified in the Probability heatmap and by the Grad-CAM method.
    
    Args:
        model (torch.nn.Module): Weights of the model to be evaluated
        input (torch.Tensor): Batch of input image. Shape: (batch_size, channels, height, width)
        mask (torch.Tensor): Binary mask of the lesion. Shape: (batch_size, 1, height, width)
        args (_type_, optional): Arguments. Defaults to None.
        
    Returns:
        bag_prob (torch.Tensor): Probability of the bag. Shape: (batch_size, nb_classes)
        patch_prob (torch.Tensor): Probability heatmap for the patches in the image. Shape: (batch_size, nb_classes, 14, 14)
        grad_cam (torch.Tensor): GradCam visualization of the ROIs. Shape: (batch_size, nb_classes, 14, 14)
    """
    # (1) Compute the output of the model
    bag_prob = model(input, None) if not args.mask_val else model(input, mask)
            
    # (2) Compute the ROIs for the different MIL pooling approaches
    patch_prob = None; grad_cam = None

    # (3) Compute the patch probabilities for the Instance-level approach
    if args.mil_type == 'instance':
        patch_prob = model.get_patch_probs() # Obtain patch probabilities
        patch_prob = patch_prob.permute(0, 2, 1)
        patch_prob = patch_prob.reshape(patch_prob.size()[0], args.nb_classes, int(math.sqrt(patch_prob.size()[-1])), int(math.sqrt(patch_prob.size()[-1]))) # Output shape: (batch_size, nb_classes, 14, 14)
    
    # (4) Compute the GradCam heatmaps for all the classes
    for i in range(args.nb_classes):
        grad_cam_comp = Grad_CAM_Eval(model, input, bag_prob, i)
        grad_cam = grad_cam_comp.unsqueeze(1) if i == 0 else torch.cat((grad_cam, grad_cam_comp.unsqueeze(1)), dim=1) # Output shape: (batch_size, nb_classes, 14, 14)
        
    return bag_prob, patch_prob, grad_cam

def MEL_Confusion_Matrix(patch_prob_mask: torch.Tensor,
                         grad_cam_mask: torch.Tensor,
                         mask: torch.Tensor,
                         mask_inv: torch.Tensor,
                         args=None) -> torch.Tensor:
    """This function computes a sort of a confusion matrix for melanoma lesions (positive class).
    We have the ROI patches and we also have the ground truth segmentation masks. 
    With we will consider the following cases (only for Melanoma lesions - positive class):
    
        1. TP: Melanoma patches (melanoma patch probability > 0.5) inside the segmentation mask (ground truth)
        2. FP: Melanoma patches (melanoma patch probability > 0.5) outside the segmentation mask (ground truth)
        3. FN: Nevus patches (melanoma patch probability < 0.5) inside the segmentation mask (ground truth)
        4. TN: Nevus patches (melanoma patch probability < 0.5) outside the segmentation mask (ground truth)
        
    Note that for a Nevus lesion this isn't as straightforward. Since we can say that the nevu skin lesion
    belongs to the Nevu class, but is dificult to say that healthy skin belongs to the melanoma class...
    
    Binary Case: Melanoma vs Nevus -> 0 vs 1
    
    Variables:
        cf_mel (torch.Tensor): Confusion matrix for the patch probability and GradCam ROIs, for the melanoma lesions.
        cf_mel[0] -> Patch probability heatmap; cf_mel[1] -> GradCam heatmap

    Args:
        patch_prob_mask (torch.Tensor): Binary heatmap of the patch probabilities. Shape: (batch_size, nb_classes, 14, 14)
        grad_cam_mask (torch.Tensor): Binary heatmap of the GradCam ROIs. Shape: (batch_size, nb_classes, 14, 14)
        mask (torch.Tensor): Heatmap of the ground truth segmentation mask. Shape: (batch_size, 1, 14, 14)
        mask_inv (torch.Tensor): Inverted heatmap of the ground truth segmentation mask. Shape: (batch_size, 1, 14, 14)
        args (**args, optional): Dictionary of arguments. Defaults to None.

    Returns:
        torch.Tensor: Output confusion matrix. Shape: (nb_classes, nb_classes) for the melanoma class
    """
    cf_mel = torch.zeros((2, args.nb_classes, args.nb_classes)) 
    
    # (1) Compute the metrics for the heatmap of the patch probabilities
    if args.mil_type == 'instance':
        cf_mel[0, 0, 0] = torch.sum(patch_prob_mask[:, 0, :, :] * mask).item() # TP
        cf_mel[0, 0, 1] = torch.sum(patch_prob_mask[:, 0, :, :] * mask_inv).item() # FP
        cf_mel[0, 1, 0] = torch.sum(patch_prob_mask[:, 1, :, :] * mask).item() # FN
        cf_mel[0, 1, 1] = torch.sum(patch_prob_mask[:, 1, :, :] * mask_inv).item() # TN
        
    # (2) Compute the metrics for the heatmap of the GradCam ROIs
    cf_mel[1, 0, 0] = torch.sum(grad_cam_mask[:, 0, :, :] * mask).item() # TP
    cf_mel[1, 0, 1] = torch.sum(grad_cam_mask[:, 0, :, :] * mask_inv).item() # FP
    cf_mel[1, 1, 0] = torch.sum(grad_cam_mask[:, 1, :, :] * mask).item() # FN
    cf_mel[1, 1, 1] = torch.sum(grad_cam_mask[:, 1, :, :] * mask_inv).item() # TN
    
    return cf_mel
 
def Basic_Eval_ROI_Vis_Inst(model: torch.nn.Module, 
                            dataloader: torch.utils.data.DataLoader, 
                            device: torch.device,
                            outputdir:os.path,
                            args=None):
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        dataloader (torch.utils.data.DataLoader): _description_
        device (torch.device): _description_
        args (_type_, optional): _description_. Defaults to None.
    """    
    mean = IMAGENET_DEFAULT_MEAN; std = IMAGENET_DEFAULT_STD
    reverse_transform = transforms.Compose([
        transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]), (1.0 / std[0], 1.0 / std[1], 1.0 / std[2])),
        transforms.ToPILImage()
    ])
        
    for input, target, input_idx, original_mask in dataloader:
                
        input, target, original_mask = input.to(device, non_blocking=True), target.to(device, non_blocking=True), original_mask.to(device, non_blocking=True)
        
        mask = mil.Mask_Setup(original_mask)
        mask = mask.reshape(mask.size()[0], int(math.sqrt(mask.size()[1])), int(math.sqrt(mask.size()[1]))) # Output shape: (batch_size, 14, 14)
        inv_mask = ~mask.bool()
        inv_mask = inv_mask.float()
        
        bag_prob, patch_prob, grad_cam = Compute_ROIs(model, input, mask, args)
        predictions = torch.argmax(bag_prob, dim=1)
        patch_prob_seg = (patch_prob > args.roi_patch_prob_threshold).float(); grad_cam_seg = (grad_cam > args.roi_gradcam_threshold).float()
                             
        for i in range(input.size()[0]):
        
            if target[i] == 0 and predictions[i] == 0:
                
                print(f"********************** Image {input_idx[i]} **********************")
                fig, axs = plt.subplots(2,6, figsize=(20, 5))
                
                # (1) Plot the input image
                image=reverse_transform(input[i])
                axs[0, 0].imshow(image); axs[0, 0].set_title(visualization.Get_Predicted_Class(target[i], predictions[i]), fontsize=7); axs[0, 0].axis('off'); 
                
                # Normalize input image
                img = input[i].permute(1, 2, 0).squeeze(0).data.cpu().numpy()
                img = (img - np.min(img)) / (np.max(img) - np.min(img))    
                
                # (2) Plot segmentation mask
                axs[0, 1].imshow(original_mask[i].permute(1,2,0), cmap='gray'); axs[0, 1].set_title("Seg. Mask", fontsize=7); axs[0, 1].axis('off'); 
                
                # (3) Plot the patch probability heatmap
                patch_mel = patch_prob[i, 0, :, :]; patch_nv = patch_prob[i, 1, :, :]
                vis_patch_prob = visualization.ShowVis(patch_mel.unsqueeze(0).unsqueeze(0), img)
                axs[0, 2].imshow(vis_patch_prob); axs[0, 2].set_title("MEL Heatmap", fontsize=7); axs[0, 2].axis('off'); 
                
                # (4) Plot Intersection between the patch probability heatmap and the seg. mask
                patch_seg_mel = patch_prob_seg[i, 0, :, :]; patch_seg_nv = patch_prob_seg[i, 1, :, :]
                patch_tp = patch_seg_mel * mask[i]; patch_fp = patch_seg_mel * inv_mask[i]; patch_fn = patch_seg_nv * mask[i]; patch_tn = patch_seg_nv * inv_mask[i]
                
                # (5) Plot TP overlay
                tp_overlay = visualization.Show_Mask_Border(patch_tp, img, color=(0, 1, 0)) # Green
                axs[0, 3].imshow(tp_overlay); axs[0, 3].set_title("TP Mask", fontsize=7); axs[0, 3].axis('off');
                
                # (6) Plot FP overlay
                fp_overlay = visualization.Show_Mask_Border(patch_fp, img, color=(1, 1, 0)) # Yelllow
                axs[0, 4].imshow(fp_overlay); axs[0, 4].set_title("FP Mask", fontsize=7); axs[0, 4].axis('off');
                
                # (7) Plot FN overlay
                fn_overlay = visualization.Show_Mask_Border(patch_fn, img, color=(1, 0, 0)) # Red
                axs[0, 5].imshow(fn_overlay); axs[0, 5].set_title("FN Mask", fontsize=7); axs[0, 5].axis('off');
                
                print(f'-------- Patch Probability Results --------\n',
                f"Precision: {(torch.sum(patch_tp).item() / (torch.sum(patch_tp).item() + torch.sum(patch_fp).item()))} \n",
                f"Recall: {(torch.sum(patch_tp).item() / (torch.sum(patch_tp).item() + torch.sum(patch_fn).item()))} \n")
                
                # Same thing but for GradCam
                axs[1, 0].imshow(image); axs[1, 0].set_title(visualization.Get_Predicted_Class(target[i], predictions[i]), fontsize=7); axs[1, 0].axis('off'); 
                axs[1, 1].imshow(original_mask[i].permute(1,2,0), cmap='gray'); axs[1, 1].set_title("Seg. Mask", fontsize=7); axs[1, 1].axis('off'); 
                
                # (8) Plot the GradCam heatmap
                grad_cam_mel = grad_cam[i, 0, :, :]; grad_cam_nv = grad_cam[i, 1, :, :]
                vis_grad_cam = visualization.ShowVis(grad_cam_mel.unsqueeze(0).unsqueeze(0), img)
                axs[1, 2].imshow(vis_grad_cam); axs[1, 2].set_title("MEL GradCam", fontsize=7); axs[1, 2].axis('off'); 
                
                # (9) Plot Intersection between the GradCam heatmap and the segmentation mask
                grad_cam_seg_mel = grad_cam_seg[i, 0, :, :]; grad_cam_seg_nv = grad_cam_seg[i, 1, :, :]
                grad_cam_tp = grad_cam_seg_mel * mask[i]; grad_cam_fp = grad_cam_seg_mel * inv_mask[i]; grad_cam_fn = grad_cam_seg_nv * mask[i]; grad_cam_tn = grad_cam_seg_nv * inv_mask[i]
                
                # (10) Plot TP overlay
                tp_overlay = visualization.Show_Mask_Border(grad_cam_tp, img, color=(0, 1, 0))
                axs[1, 3].imshow(tp_overlay); axs[1, 3].set_title("TP Mask", fontsize=7); axs[1, 3].axis('off');
                
                # (11) Plot FP overlay
                fp_overlay = visualization.Show_Mask_Border(grad_cam_fp, img, color=(1, 1, 0))
                axs[1, 4].imshow(fp_overlay); axs[1, 4].set_title("FP Mask", fontsize=7); axs[1, 4].axis('off');
                
                # (12) Plot FN overlay
                fn_overlay = visualization.Show_Mask_Border(grad_cam_fn, img, color=(1, 0, 0))
                axs[1, 5].imshow(fn_overlay); axs[1, 5].set_title("FN Mask", fontsize=7); axs[1, 5].axis('off');
                
                print(f'-------- GradCam Results --------\n',
                f"Precision: {(torch.sum(grad_cam_tp).item() / (torch.sum(grad_cam_tp).item() + torch.sum(grad_cam_fp).item()))} \n",
                f"Recall: {(torch.sum(grad_cam_tp).item() / (torch.sum(grad_cam_tp).item() + torch.sum(grad_cam_fn).item()))} \n")
            
                title = f"Image {input_idx[i]}"
                plt.suptitle(title, fontsize=14, fontweight='bold')
                plt.savefig(str(outputdir) + f'/MIL-{args.mil_type}-{args.pooling_type}-{args.dataset}-{input_idx[i]}.jpg', dpi=300, bbox_inches='tight')
                plt.clf(); plt.close()
                
        #break
    return  
        
def Basic_Evaluation_ROIs(model: torch.nn.Module, 
                          dataloader: torch.utils.data.DataLoader, 
                          device: torch.device,
                          args=None):
    """Evaluate the quality of the ROIs generated by the model.
    This function measures the number of patches of the ROI that 
    are inside the ground truth segmentation mask of the lesion.
    
    This function performs the following operations:
    
        1. Generates the ROIs for the different MIL pooling operators
        2. Generates a ROI binary mask
        3. Computes the intersection between the ROI mask and the ground truth
        4. Compute Basic metrics:
            4.1. Precision: Divides the intersection by the number of patches of the ROI
            4.2. Loss: Subtracts the intersection to the number of patches of the ROI. 
                 Then divides by the number of patches of the ROI (Loss = 1 - Precision)
                 
    Variables:
        results (torch.Tensor): Evaluation metrics for ROI detection.
        results[0] -> Patch probability heatmap; results[1] -> GradCam heatmap
        
    Args:
        model (torch.nn.Module): Model to be evaluated
        dataloader (torch.utils.data.DataLoader): Dataloader for the evaluation
        device (torch.device): Device to run the evaluation on
        args (_type_, optional): Arguments. Defaults to None.
    """
    results = torch.zeros((2,4)) # Evaluation metrics for ROI detection
        
    # (1) Set the model to evaluation mode
    model.eval()
    
    # (2) Iterate over the dataloader
    for input, target, input_idx, mask in dataloader:
            
        # (3) Set input, target and mask to the device
        input, target, mask = input.to(device, non_blocking=True), target.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        
        # (4) Process segmentation mask
        mask = mil.Mask_Setup(mask)
        mask = mask.reshape(mask.size()[0], int(math.sqrt(mask.size()[1])), int(math.sqrt(mask.size()[1]))) # Output shape: (batch_size, 14, 14)
        inv_mask = ~mask.bool(); inv_mask = inv_mask.float()
        
        # (5) Compute the ROIs
        bag_prob, patch_prob, grad_cam = Compute_ROIs(model, input, mask, args)
        
        # (6) Generate the ROI binary mask
        patch_prob_seg = (patch_prob > args.roi_patch_prob_threshold).float(); grad_cam_seg = (grad_cam > args.roi_gradcam_threshold).float()
                        
        # (7) Compute metrics
        cf_mel = MEL_Confusion_Matrix(patch_prob_seg, grad_cam_seg, mask, inv_mask, args)
        
        # (8) Compute evaluation metrics        
        if args.mil_type == 'instance':
            results[0,0] += ((cf_mel[0, 0, 0] + cf_mel[0, 1, 1]) / torch.sum(patch_prob_seg).item())/len(dataloader)
            results[0,1] += (cf_mel[0, 0, 0] / torch.sum(patch_prob_seg[:, 0, :, :]).item())/len(dataloader)
            results[0,2] += (cf_mel[0, 0, 0] / (cf_mel[0, 0, 0] + cf_mel[0, 1, 0]))/len(dataloader)
            results[0,3] += (cf_mel[0, 0, 0] / (cf_mel[0, 0, 0] + cf_mel[0, 0, 1]))/len(dataloader)    
            
        results[1,0] += ((cf_mel[1, 0, 0] + cf_mel[1, 1, 1]) / torch.sum(grad_cam_seg).item())/len(dataloader)
        results[1,1] += (cf_mel[1, 0, 0] / torch.sum(grad_cam_seg[:, 0, :, :]).item())/len(dataloader)
        results[1,2] += (cf_mel[1, 0, 0] / (cf_mel[1, 0, 0] + cf_mel[1, 1, 0]))/len(dataloader)
        results[1,3] += (cf_mel[1, 0, 0] / (cf_mel[1, 0, 0] + cf_mel[1, 0, 1]))/len(dataloader)
        
    print(results)
    exit()         