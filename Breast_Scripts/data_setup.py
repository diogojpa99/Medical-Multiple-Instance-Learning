import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

from torchvision import transforms
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from typing import Union

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

def Train_Transform(input_size:int=224,
                    args=None) -> transforms.Compose:
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
    
    t.append(transforms.RandomVerticalFlip())
    t.append(transforms.RandomHorizontalFlip())
    t.append(transforms.Resize([input_size, input_size], antialias=True))
    t.append(transforms.RandomCrop(input_size, padding=0)), 
    t.append(transforms.ToTensor())
    t.append(transforms.Lambda(Gray_to_RGB_Transform)) #t.append(transforms.Grayscale(num_output_channels=3))
    #t.append(transforms.RandomRotation(10))
    
    if args.breast_strong_aug:
        t.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
        t.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))

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
    
    t.append(transforms.ToTensor())
    t.append(transforms.Resize([input_size, input_size], antialias=True))
    t.append(transforms.Lambda(Gray_to_RGB_Transform)) #t.append(transforms.Grayscale(num_output_channels=3)) 
    
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
    if args.loader=='Gray_PIL_Loader_Wo_He':
        train_set = ImageFolder(root=train_path, transform=train_transform, loader=Gray_PIL_Loader_Wo_He)
        val_set = ImageFolder(root=val_path, transform=val_transform, loader=Gray_PIL_Loader_Wo_He)
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
         
def Get_TestLoader(data_path:str,
                   input_size:int=224, 
                   batch_size:int=64, 
                   num_workers:int=0,
                   args=None) -> DataLoader:
    """This function returns the training, validation data loaders. 
    In this case, there is no 'val' folder in the data path. Thus, we need to split the training set into training and validation sets.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        input_size (int, optional): Input size for the model. Defaults to 224.
        batch_size (int, optional): Batch size. Defaults to 64.
        num_workers (int, optional): Number of workers. Defaults to 0.
        args (argparse.ArgumentParser, optional): Arguments. Defaults to None.

    Returns:
        DataLoader: Test data loader.
    """
    root = os.path.join(data_path)
    test_loader = None
    
    if 'test' in os.listdir(root):
        test_path = os.path.join(root, 'test')
    elif 'test' not in os.listdir(root) and 'val' in os.listdir(root):
        test_path = os.path.join(root, 'val')
        print('No test folder found in the data path. Using the validation folder as the test folder.')
    else:
        ValueError('No test folder (nor val folder) found in the data path. Make sure the data path has a test folder.')
        
    test_transform = Test_Transform(input_size=input_size, args=args)
    
    test_set = ImageFolder(root=test_path, transform=test_transform, loader=Gray_PIL_Loader_Wo_He)
    
    test_loader = DataLoader(test_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=(torch.cuda.is_available()),
                            drop_last=False)
    
    return test_loader
                   