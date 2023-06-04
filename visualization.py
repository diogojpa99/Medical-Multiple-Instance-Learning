import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import os

def get_predicted_class(image_file, predicted_class):
    if image_file[0].lower() == 'm':
        prefix = 'Mel'
    elif image_file[0].lower() == 'n':
        prefix = 'NV'
    else:
        prefix = ''
    
    pred_class = 'MEL' if predicted_class == 0 else 'NV'
    
    return f'{prefix} | Pred: {pred_class}'

def Grad_CAM(input, model, prediction, label, img):
    """ 
    Function inspired by: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    
    Note: In the beginning, when we set the prediction as backward we can do it in two ways:
    
    1) prediction[:, label].backward(retain_graph=True)
    2) One hot technique
    
    Both methods are equivalent. At least I tested both and the results are the same.
    
    """   
     
    """ one_hot = np.zeros((1, prediction.size()[-1]), dtype=np.float32) 
    one_hot[0, label] = 1  
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)  
    one_hot = torch(one_hot * prediction) """ 
    
    model.zero_grad() 
    #one_hot.backward(retain_graph=True)
    prediction[:, label].backward(retain_graph=True)
    
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(input).detach()
    
    for i in range(256):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap) 

    heatmap = torch.nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0), scale_factor=(224//14), mode='bilinear',align_corners=True)  #14->224
    heatmap = heatmap.reshape(224, 224).data.cpu().numpy() 
    heatmap = cv2.applyColorMap(np.uint8(heatmap* 255), cv2.COLORMAP_JET)   
   
    heatmap = np.float32(heatmap) / 255
    cam = heatmap*0.9 + np.float32(img)
    cam = cam / np.max(cam)
    
    vis =  np.uint8(255 * cam)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        
    return vis
    
def Visualize_Most_Relevant_Patch_InstanceMax(model: torch.nn.Module, datapath, outputdir = None, args = None):
    
    image_files = os.listdir(datapath)
    fig, axs = plt.subplots(3, len(image_files), figsize=(4 * len(image_files), 13))
    
    # Transform the images for the model
    transform = transforms.Compose(
    [   transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    for i, image_file in enumerate(image_files):
        
        image_path = os.path.join(datapath, image_file)
        image = Image.open(image_path)

        input = transform(image).unsqueeze(0)
        
        # (3) Set model to eval mode
        model.eval()
        
        # (4) Obtain the bag scores
        bag_score = model(input)
        predicted_class =int(torch.argmax(bag_score))
                    
        # (5) obtain the instance scores
        instance_scores = model.patch_scores
        
        # (6) Obtain probabilities (Softmax) & Transform to (batch_size, num_classes, 14, 14)
        prob = torch.softmax(instance_scores, dim=1)  # Softmax over the patches,i.e, see the most relevant patches for the predicted class
        prob = prob.permute(0, 2, 1)
        prob = prob.reshape(1, 2, 14, 14)
        
        # (7) Obtain the key patch
        instances_map = instance_scores[:,:,predicted_class].squeeze(0) # Get the instance scores for the predicted class
        key_patch_idx = int(torch.argmax(instances_map)) # Get the index of the key patch
        instances_map.zero_() # Set all the patches to 0
        instances_map[key_patch_idx] = 1 # Set the key patch to 1
        instances_map = instances_map.reshape(14, 14).data.cpu().numpy() # Reshape to (14, 14)
        key_patch_idx = np.argwhere(instances_map == 1) # Get the index of the key patch
        top_left_x = key_patch_idx[0][1] * (224 // 14); top_left_y = key_patch_idx[0][0] * (224 // 14)
        bottom_right_x = top_left_x + (224 // 14); bottom_right_y = top_left_y + (224 // 14)
        key_patch = np.copy(np.array(image))
        cv2.rectangle(key_patch, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

        # (8) Obtain the activtion map for the predicted class
        activation_map = prob[:,predicted_class, :, :].unsqueeze(0)
        heatmap = torch.nn.functional.interpolate(activation_map, scale_factor=(224//14), mode='bilinear', align_corners=True)  #14->224
        heatmap = heatmap.reshape(224, 224).data.cpu().numpy() 
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap = cv2.applyColorMap(np.uint8(heatmap* 255), cv2.COLORMAP_JET)
        
        img = input.permute(0, 2, 3, 1).squeeze(0).data.cpu().numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))    
        heatmap = np.float32(heatmap) / 255
        cam = heatmap*0.9 + np.float32(img)
        cam = cam / np.max(cam)
        
        vis =  np.uint8(255 * cam)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        
        # Plot the original image
        axs[0, i].imshow(image)
        axs[0, i].set_title(get_predicted_class(image_file, predicted_class), fontsize=16)
        axs[0, i].axis('off');
        
        # Plot the original the key patch
        axs[1, i].imshow(key_patch, cmap='jet')
        axs[1, i].set_title("Key Patch")
        axs[1, i].axis('off');

        # Plot the heatmap overlay
        axs[2, i].imshow(vis)
        axs[2, i].set_title("Normalized Activation Map")
        axs[2, i].axis('off');
        
        # Plot the original 14x14 heatmap
        """ axs[2, i].imshow(activation_map.squeeze().cpu().detach().numpy(), cmap='jet')
        axs[2, i].axis('off'); """
            
    title = f"| Most Relevant Patches for Predicted Class | MIL Type: {args.mil_type} | Pooling Type: {args.pooling_type} |"
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.1 ,hspace=0.1)
    plt.savefig(str(outputdir) + f'/MIL-{args.mil_type}-{args.pooling_type}-MostRelevantPatch-PredictedClass.jpg', dpi=300, bbox_inches='tight')
    
def Visualize_Most_Relevant_Patches(model: torch.nn.Module, datapath, outputdir = None, args = None):
    
    image_files = os.listdir(datapath)
    fig, axs = plt.subplots(2, len(image_files), figsize=(4 * len(image_files), 8.5))

    # Transform the images for the model
    transform = transforms.Compose(
    [   transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    for i, image_file in enumerate(image_files):
        
        image_path = os.path.join(datapath, image_file)
        image = Image.open(image_path)

        input = transform(image).unsqueeze(0)
        
        # (3) Set model to eval mode
        model.eval()
        
        # (4) Obtain the bag scores
        bag_score = model(input)
        predicted_class =int(torch.argmax(bag_score))
                
        # (5) obtain the instance scores
        instance_scores = model.patch_scores 
        
        # (6) Obtain probabilities (Softmax) & Transform to (batch_size, num_classes, 14, 14)
        prob = torch.softmax(instance_scores, dim=1)  # Softmax over the patches,i.e, see the most relevant patches for the predicted class 
        #prob = instance_scores
        prob = prob.permute(0, 2, 1)
        prob = prob.reshape(1, 2, 14, 14)
        
        # (7) Obtain the activtion map for the predicted class
        activation_map = prob[:,predicted_class, :, :].unsqueeze(0)
        
        heatmap = torch.nn.functional.interpolate(activation_map, scale_factor=(224//14), mode='bilinear',align_corners=True)  #14->224
        heatmap = heatmap.reshape(224, 224).data.cpu().numpy() 
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap = cv2.applyColorMap(np.uint8(heatmap* 255), cv2.COLORMAP_JET)
        
        img = input.permute(0, 2, 3, 1).squeeze(0).data.cpu().numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))    
        
        heatmap = np.float32(heatmap) / 255
        cam = heatmap*0.9 + np.float32(img)
        cam = cam / np.max(cam)
        
        vis =  np.uint8(255 * cam)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        
        # Plot the original image
        axs[0, i].imshow(image)
        axs[0, i].set_title(get_predicted_class(image_file, predicted_class), fontsize=16)
        axs[0, i].axis('off');

        # Plot the heatmap overlay
        axs[1, i].imshow(vis)
        axs[1, i].set_title("Normalized Activation Map")
        axs[1, i].axis('off');
        
        # Plot the original 14x14 heatmap
        """ axs[2, i].imshow(activation_map.squeeze().cpu().detach().numpy(), cmap='jet')
        axs[2, i].axis('off'); """

        # Plot the original 14x14 heatmap
        """ heat = heatmap / np.max(heatmap)
        heat = np.uint8(255 * heat)
        heat = cv2.cvtColor(np.array(heat), cv2.COLOR_RGB2BGR)
        axs[2, i].imshow(heat[::16, ::16])
        axs[2, i].axis('off');  """
            
    title = f"| Most Relevant Patches for Predicted Class | MIL Type: {args.mil_type} | Pooling Type: {args.pooling_type} |"
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(str(outputdir) + f'/MIL-{args.mil_type}-{args.pooling_type}-MostRelevantPatch-PredictedClass.jpg', dpi=300, bbox_inches='tight')
    
def Visualize_ActivationMap(model: torch.nn.Module, datapath, outputdir = None, args = None):
    
    image_files = os.listdir(datapath)
    fig, axs = plt.subplots(3, len(image_files), figsize=(4 * len(image_files), 13))

    # Transform the images for the model
    transform = transforms.Compose(
    [   transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    for i, image_file in enumerate(image_files):
        
        image_path = os.path.join(datapath, image_file)
        image = Image.open(image_path)

        input = transform(image).unsqueeze(0)
        
        # (3) Set model to eval mode
        model.eval()
        
        # (4) Obtain the bag scores
        bag_score = model(input)
        predicted_class =int(torch.argmax(bag_score))
            
        # (5) obtain the instance scores
        instance_scores = model.patch_scores 
        
        # (6) Obtain probabilities (Softmax) & Transform to (batch_size, num_classes, 14, 14)
        prob = torch.softmax(instance_scores, dim=2)
        prob = prob.permute(0, 2, 1)
        prob = prob.reshape(1, 2, 14, 14)
        
        # (7) Obtain the activtion map for the predicted class
        activation_map = prob[:,0, :, :].unsqueeze(0)
        
        # (8) Overlay the heatmap on the original image
        heatmap = torch.nn.functional.interpolate(activation_map, scale_factor=(224//14), mode='bilinear',align_corners=True)  #14->224
        heatmap = heatmap.reshape(224, 224).data.cpu().numpy() 
        heatmap = np.clip(heatmap, 0, 1) # Testar Sem o Clip
        heatmap = cv2.applyColorMap(np.uint8(heatmap* 255), cv2.COLORMAP_JET)
        
        img = input.permute(0, 2, 3, 1).squeeze(0).data.cpu().numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        heatmap = np.float32(heatmap) / 255
        cam = heatmap*0.9 + np.float32(img)
        cam = cam / np.max(cam)
        
        vis =  np.uint8(255 * cam)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        
        # Grad CAM
        grad_cam = Grad_CAM(input,model,bag_score,0, img)
        
        # Plot the original image
        axs[0, i].imshow(image)
        axs[0, i].set_title(get_predicted_class(image_file, predicted_class), fontsize=16)
        axs[0, i].axis('off');

        # Plot the heatmap overlay
        axs[1, i].imshow(vis)
        axs[1, i].set_title("Activation Map (0-1)")
        axs[1, i].axis('off');
        
        # Plot the original 14x14 heatmap
        axs[2, i].imshow(grad_cam)
        axs[2, i].set_title("Grad-CAM")
        axs[2, i].axis('off');
                    
    title = f"| 'MEL' Class Activation Map | MIL Type: {args.mil_type} | Pooling Type: {args.pooling_type} | "
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(str(outputdir) + f'/MIL-{args.mil_type}-{args.pooling_type}-Heatmap_01-MEL.jpg', dpi=300, bbox_inches='tight')

def Visualize_Embedding_ActivationMap(model: torch.nn.Module, datapath, outputdir = None, args = None):
    
    image_files = os.listdir(datapath)
    fig, axs = plt.subplots(3, len(image_files), figsize=(4 * len(image_files), 13))

    # Transform the images for the model
    transform = transforms.Compose(
    [   transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    for i, image_file in enumerate(image_files):
        
        image_path = os.path.join(datapath, image_file)
        image = Image.open(image_path)

        input = transform(image).unsqueeze(0)
        
        # (3) Set model to eval mode
        model.eval()
        
        # (4) Obtain the bag scores
        bag_score = model(input)
        predicted_class =int(torch.argmax(bag_score))
        
        img = input.permute(0, 2, 3, 1).squeeze(0).data.cpu().numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        # Grad CAM
        grad_cam_mel = Grad_CAM(input,model,bag_score,0, img)
        grad_cam_nv = Grad_CAM(input,model,bag_score,1,img)
        
        # Plot the original image
        axs[0, i].imshow(image)
        axs[0, i].set_title(get_predicted_class(image_file, predicted_class), fontsize=16)
        axs[0, i].axis('off');
        
        # Plot the original 14x14 heatmap
        axs[1, i].imshow(grad_cam_mel)
        axs[1, i].set_title("Grad-CAM [MEL]")
        axs[1, i].axis('off');
        
        # Plot the original 14x14 heatmap
        axs[2, i].imshow(grad_cam_nv)
        axs[2, i].set_title("Grad-CAM [NV]")
        axs[2, i].axis('off');
        
    title = f"| Class Activation Map | MIL Type: {args.mil_type} | Pooling Type: {args.pooling_type} | "
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(str(outputdir) + f'/MIL-{args.mil_type}-{args.pooling_type}-GradCAM.jpg', dpi=300, bbox_inches='tight')
        
def Visualize_Instance_Scores_indv(model: torch.nn.Module, datapath, outputdir = None, args = None):
    
    # (1) Load the images
    image_1 = Image.open(datapath)
    
    # (2) transform the images
    transform = transforms.Compose(
    [   transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    
    img = transform(image_1).unsqueeze(0)
    
    # (3) Set model to eval mode
    model.eval()
    
    # (4) Obtain the bag scores
    with torch.no_grad():
        bag_scores = model(img)
        
    # (5) obtain the instance scores
    instance_scores = model.patch_scores 
    
    # (6) Obtain probabilities (Softmax) & Transform to (batch_size, num_classes, 14, 14)
    prob = torch.softmax(instance_scores, dim=1)
    prob = prob.permute(0, 2, 1)
    prob = prob.reshape(1, 2, 14, 14)
    
    # (7) Obtain the activtion map for the predicted class
    predicted_class =int(torch.argmax(bag_scores))
    activation_map = prob[:,0, :, :].unsqueeze(0)
    
    heatmap = torch.nn.functional.interpolate(activation_map, scale_factor=16, mode='bilinear',align_corners=True)  #14->224
    heatmap = heatmap.reshape(224, 224).data.cpu().numpy() 
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(heatmap* 255), cv2.COLORMAP_JET)
    img1 = cv2.imread(datapath , cv2.IMREAD_COLOR)

    #cv2.imwrite("vis.jpg", heatmap*0.3 + np.float32(img1)*0.7)
    vis = heatmap*0.3 + np.float32(img1)*0.7
    
    return vis, predicted_class
    
def Visualize_ActivationMaps(model: torch.nn.Module, datapath, outputdir, args=None):
    # Create the main output directory
    os.makedirs(outputdir, exist_ok=True)

    mil_pooling_dir = os.path.join(outputdir, f"MIL-{args.mil_type}-{args.pooling_type}_Heatmap01")
    os.makedirs(mil_pooling_dir, exist_ok=True)

    mel_dir = os.path.join(mil_pooling_dir, "MEL"); nv_dir = os.path.join(mil_pooling_dir, "NV")
    os.makedirs(mel_dir, exist_ok=True); os.makedirs(nv_dir, exist_ok=True)

    mel_folder = os.path.join(datapath, "MEL"); mel_files = os.listdir(mel_folder)
    for image_file in mel_files:
        
        image_path = os.path.join(mel_folder, image_file)
        vis, predicted_class = Visualize_Instance_Scores_indv(model, image_path)
        
        # Save the image based on the predicted class and correctness
        if predicted_class == 0:
            save_dir = os.path.join(mel_dir, "Correct")
        else:
            save_dir = os.path.join(mel_dir, "Wrong")
            
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_file)
        cv2.imwrite(save_path, vis)

    nv_folder = os.path.join(datapath, "NV"); nv_files = os.listdir(nv_folder)
    for image_file in nv_files:
        image_path = os.path.join(nv_folder, image_file)
        vis, predicted_class = Visualize_Instance_Scores_indv(model, image_path)
        
        # Save the image based on the predicted class and correctness
        if predicted_class == 1:
            save_dir = os.path.join(nv_dir, "Correct")
        else:
            save_dir = os.path.join(nv_dir, "Wrong")
            
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_file)
        cv2.imwrite(save_path, vis)
