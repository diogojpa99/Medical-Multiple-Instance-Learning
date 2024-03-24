import os
from PIL import Image
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

from torchvision import transforms
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.datasets import ImageFolder

from typing import Union

#global std, mean

def define_mean_std(dataset:str) -> tuple:
    """This function defines the mean and standard deviation for the dataset.

    Args:
        dataset (str): Dataset name.

    Returns:
        tuple: mean, std.
    """
    
    # if dataset=="MIAS_CLAHE-mass_normal":
    #     return 0.3229, 0.2409
    # elif dataset=="MIAS_CLAHE-benign_malignant":
    #     return 0.3254, 0.2403
    # elif dataset=="CBIS_CLAHE-benign_malignant":
    #     return 0.3040, 0.2678
    
    if dataset=="DDSM_CLAHE-mass_normal":
        print('Mean and std for DDSM_CLAHE-mass_normal: 0.2333, 0.2410')
        return 0.2333, 0.2410
    elif dataset=="DDSM+CBIS_CLAHE-mass_normal":
        print('Mean and std for DDSM+CBIS_CLAHE-mass_normal: 0.2333, 0.2410')
        return 0.2505, 0.2497
    elif dataset=="DDSM_CLAHE-benign_malignant":
        print('Mean and std for DDSM_CLAHE-benign_malignant: 0.2693, 0.2504')
        return 0.2693, 0.2504
    elif dataset=="DDSM+CBIS_CLAHE-benign_malignant":
        print('Mean and std for DDSM+CBIS_CLAHE-benign_malignant: 0.2807, 0.2567')
        return 0.2807, 0.2567
    elif dataset=="DDSM+CBIS+MIAS_CLAHE-benign_malignant":
        print('Mean and std for DDSM+CBIS+MIAS_CLAHE-benign_malignant: 0.2820, 0.2563')
        return 0.2820, 0.2563
    else:
        ValueError('Dataset not found.')
        
    return None, None
        
def Gray_PIL_Loader_Wo_He(path: str) -> Image.Image:
    """This function opens the image using PIL and converts it to grayscale.
    Then resizes the grayscale image to a square shape (width equals height) using bilinear interpolation  

    Args:
        path (str): Path to the image.

    Returns:
        Image.Image: loaded image.
    """
    image = Image.open(path)
    return image.convert('L').resize((max(image.size),max(image.size)), resample=Image.BILINEAR)

def Gray_PIL_Loader_Wo_He_No_Resize(path: str) -> Image.Image:
    """This function opens the image using PIL and converts it to grayscale.
    Then resizes the grayscale image to a square shape (width equals height) using bilinear interpolation  

    Args:
        path (str): Path to the image.

    Returns:
        Image.Image: loaded image.
    """
    return Image.open(path).convert('L')

def Gray_PIL_Loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        image = np.array(Image.open(f))

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), 256, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = (cdf - cdf.min())*255/(cdf.max()-cdf.min())  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    image = image_equalized.reshape(image.shape)

    return Image.fromarray(np.uint8(image), 'L')

def Gray_to_RGB_Transform(x: torch.Tensor) -> torch.Tensor:
    """This function converts a grayscale image to RGB by concatenating the grayscale image to itself 3 times.

    Args:
        x (torch.Tensor): Grayscale image.

    Returns:
        torch.Tensor: RGB image.
    """
    return torch.cat([x, x, x], 0)

class CLAHE_Transform:
    def __init__(self, clip_limit):
        self.clip_limit = clip_limit
        
    def __call__(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        return Image.fromarray((clahe.apply(np.array(img)).astype(np.float32) / 255.0)) 

def apply_clahe(x:Image.Image) -> Image.Image:
    """Apply CLAHE to the input image.

    Args:
        image (PIL): The input image.

    Returns:
        _type_: _description_
    """
    clahe = cv2.createCLAHE(clipLimit=0.02, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(np.array(x)))

def padding_image_one_side(x:torch.Tensor) -> torch.Tensor:
    """
    Pad the input image tensor to make it square.
    We only want to pad the side with the least amount of information.
        
    Args:
        image (torch.Tensor): The input image tensor with shape (C, H, W).
        min_size (int): Minimum size for the image after padding.
        padding_value (int): Value to use for padding. Default is 0 (black).
    
    Returns:
        torch.Tensor: Padded image tensor with square shape.
    """

    min_size=224
    padding_value=0
    
    c, h, w = x.size()
    max_side = max(h, w, min_size)
    padding_left = max_side - w
    padding_right = max_side - w 
    padding_top = (max_side - h) // 2
    padding_bottom = max_side - h - padding_top
    
    # Compute the side that has the least amount of information
    column_sums = x.sum(dim=[0, 1])
    values, idx = torch.topk(column_sums, k=int(0.10 * column_sums.size(-1)), largest=False, sorted=True)
    dist_left = torch.mean(idx.float()) # Average distance from the columns with the least amount of information to the left edge
    dist_right = torch.mean((torch.ones(len(idx))*x.size(-1)) - idx.float()) # Average distance from the columns with the least amount of information to the right edge

    if dist_left < dist_right:
        padding = (padding_left, 0, 0, 0)
    else:
        padding = (0, 0, padding_right, 0)
                
    return F.pad(x, padding, padding_value, 'constant')

def transform_images_to_left(x:torch.Tensor) -> torch.Tensor:
    """Transforms the input image tensor to the left.

    Args:
        x (torch.Tensor): The input image tensor.

    Returns:
    """
    # Compute the side that has the least amount of information
    column_sums = x.sum(dim=[0, 1])
    values, idx = torch.topk(column_sums, k=int(0.10 * column_sums.size(-1)), largest=False, sorted=True)
    dist_left = torch.mean(idx.float()) # Average distance from the columns with the least amount of information to the left edge
    dist_right = torch.mean((torch.ones(len(idx))*x.size(-1)) - idx.float()) # Average distance from the columns with the least amount of information to the right edge

    # If the breast is on the right side then perform an horizontal flip
    if dist_left < dist_right:
        x = F.hflip(x)
                
    return x

def General_Img_Transform(t:list, input_size:int=224, args=None) -> transforms.Compose:
    """_summary_

    Args:
        t (list): _description_
        input_size (int, optional): _description_. Defaults to 224.
        args (_type_, optional): _description_. Defaults to None.

    Returns:
        transforms.Compose: _description_
    """
    if args.breast_clahe: 
        clahe_transform = CLAHE_Transform(clip_limit=args.clahe_clip_limit)
        #t.append(transforms.Lambda(apply_clahe))
        t.append(clahe_transform)
        
    t.append(transforms.ToTensor())
    
    if args.breast_padding: 
        t.append(transforms.Lambda(padding_image_one_side))
        
    mean, std = define_mean_std(args.dataset)
    t.append(transforms.Normalize(mean=[mean], std=[std]))
    
    t.append(transforms.Resize([224, 224], antialias=args.breast_antialias))
        
    if args.breast_transform_left:
        t.append(transforms.Lambda(transform_images_to_left))
        
    if args.breast_transform_rgb:           
        t.append(transforms.Lambda(Gray_to_RGB_Transform)) #t.append(transforms.Grayscale(num_output_channels=3))
        
    return t

def Train_Transform(input_size:int=224, args=None) -> transforms.Compose:
    """Builds the data transformation pipeline.
    Since the dataset is grayscale, we need to convert it to RGB.
    The grayscale images are converted to RGB by concatenating the grayscale image to itself 3 times.
    
    During training, we use data augmentation techniques such as:
        1. Random cropping; 
        2. Random horizontal and vertical flipping, 
        3. Random rotation.

    Args:
        input_size (int, optional): Input size of the model. Defaults to 224.
        args (*args): Arguments.

    Returns:
        torchvision.transforms.Compose(): Transformation pipeline.
    """
    t = []
    
    t = General_Img_Transform(t, input_size, args)
        
    # Data augmentation
    if args.breast_strong_aug:
        if not args.breast_transform_left:
            t.append(transforms.RandomHorizontalFlip())
        t.append(transforms.RandomVerticalFlip())
        t.append(transforms.RandomRotation(7))
        #t.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
        #t.append(transforms.RandomCrop(input_size, padding=0))
        #t.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
        #t.append(transforms.RandomAffine(degrees=0, translate=(0.0, 0.1)))
        #t.append(transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)))
        #t.append(transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)))
        #t.append(transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)))
        
    return transforms.Compose(t)

def Test_Transform(input_size:int=224,
                   args=None) -> transforms.Compose:
    """Builds the data transformation pipeline for the testset.

    Args:
        input_size (int, optional): Input size of the model. Defaults to 224.
        args (Parser.Arguments, optional): Arguments. Defaults to None.

    Returns:
        transforms.Compose: Transformation pipeline for the test/validation set.
    """
    t = []
    t = General_Img_Transform(t, input_size, args)
    return transforms.Compose(t)
  
def Build_Datasets(data_path:str,
                   input_size:int=224, 
                   args=None) -> Union[Dataset, Dataset]:
    """This function returns the training, validation datasets.
    If there is no 'val' folder in the data path, then we need to split the training set into training and validation sets.
    In this case the training-validation split is an argument of the program and it is performs a 80-20 split (by default).

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        input_size (int, optional): Input size for the model. Defaults to 224.
        args (argparse.ArgumentParser, optional): Arguments. Defaults to None.

    Returns:
        (Dataset, Dataset): Training and validation Datasets.
    """
    root = os.path.join(data_path)
    train_val_splt = False
    train_path, val_path = None, None
    
    # Check if the data path has a 'train' and 'val' folders
    root_dict = os.listdir(root)

    if 'train' not in root_dict:
        ValueError('No "train" folder found in the data path. Make sure the data path has a "train" folder.')
        
    if 'val' in root_dict:
        train_val_splt = False
        train_path = os.path.join(root, 'train'); val_path = os.path.join(root, 'val')
    elif args.test_val_flag:
        print('Alert: A "test" folder was found, but no "val" folder found in the data path. Using the "test" folder as the validation folder.')
        train_val_splt = False
        train_path = os.path.join(root, 'train'); val_path = os.path.join(root, 'test')
    else:
        print('Alert: No "val" folder found in the data path. Train-validation split will be performed.')
        train_val_splt = True
        train_path = os.path.join(root, 'train'); val_path = os.path.join(root, 'train')

    # Build the Transform pipelines
    train_transform = Train_Transform(input_size=input_size, args=args)
    val_transform = Test_Transform(input_size=input_size, args=args)
    
    # Build the datasets
    if args.breast_loader=='Gray_PIL_Loader_Wo_He':
        train_set = ImageFolder(root=train_path, transform=train_transform, loader=Gray_PIL_Loader_Wo_He)
        val_set = ImageFolder(root=val_path, transform=val_transform, loader=Gray_PIL_Loader_Wo_He)
    elif args.breast_loader=='Gray_PIL_Loader_Wo_He_No_Resize':
        train_set = ImageFolder(root=train_path, transform=train_transform, loader=Gray_PIL_Loader_Wo_He_No_Resize)
        val_set = ImageFolder(root=val_path, transform=val_transform, loader=Gray_PIL_Loader_Wo_He_No_Resize)
    else:
        train_set = ImageFolder(root=train_path, transform=train_transform, loader=Gray_PIL_Loader)
        val_set = ImageFolder(root=val_path, transform=val_transform, loader=Gray_PIL_Loader)
    
    # Assert that the number of classes
    args.nb_classes = len(train_set.classes)
    
    # Train-val split
    if train_val_splt:
        splt = torch.split(torch.randperm(len(train_set.samples)), int(args.train_val_split * len(train_set.samples)))
        train_set.samples = [s for [i, s] in enumerate(train_set.samples) if i in splt[0]]
        val_set.samples = [s for [i, s] in enumerate(val_set.samples) if i in splt[1]]
        train_set.targets = [s for [i, s] in enumerate(train_set.targets) if i in splt[0]]
        val_set.targets = [s for [i, s] in enumerate(val_set.targets) if i in splt[1]]
        train_set.imgs = train_set.samples; val_set.imgs = val_set.samples
        
        
    return train_set, val_set
         
def Get_Testset(data_path:str,
                input_size:int=224, 
                args=None) -> DataLoader:
    """This function returns the Test set. 
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        input_size (int, optional): Input size for the model. Defaults to 224.
        args (argparse.ArgumentParser, optional): Arguments. Defaults to None.

    Returns:
        DataLoader: Test data loader.
    """
    root = os.path.join(data_path)
    test_path = None; test_set = None
        
    if 'test' in os.listdir(root):
        test_path = os.path.join(root, 'test')
    elif 'test' not in os.listdir(root) and 'val' in os.listdir(root):
        test_path = os.path.join(root, 'val')
        print('No test folder found in the data path. Using the validation folder as the test folder.')
    elif 'test' not in os.listdir(root) and 'val' not in os.listdir(root) and 'train' in os.listdir(root):
        test_path = os.path.join(root, 'train')
        print('No test folder (nor val folder) found in the data path. Using the training folder as the test folder.')
    else:
        ValueError('No test folder (nor val folder) found in the data path. Make sure the data path has a test folder.')
        
    test_transform = Test_Transform(input_size=input_size, args=args)
    
    if args.breast_loader=='Gray_PIL_Loader_Wo_He':
        test_set = ImageFolder(root=test_path, transform=test_transform, loader=Gray_PIL_Loader_Wo_He)
    elif args.breast_loader=='Gray_PIL_Loader_Wo_He_No_Resize':
        test_set = ImageFolder(root=test_path, transform=test_transform, loader=Gray_PIL_Loader_Wo_He_No_Resize)
    else:
        test_set = ImageFolder(root=test_path, transform=test_transform, loader=Gray_PIL_Loader)
            
    return test_set
                   