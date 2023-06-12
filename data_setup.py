import os
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class ISIC2019_Dataset(Dataset):
    
    def __init__(self, 
                 image_dir, 
                 mask_dir=None, 
                 img_transform=None,
                 mask_is_val = False, 
                 mask_transform=None):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_transform = mask_transform
        self.img_transform = img_transform
        self.idx = []
        self.targets = []
        self.classes, self.class_to_idx = find_classes(self.image_dir)
        
        if not mask_is_val:
            self.mask_dir = None
            
        
        for label in os.listdir(self.image_dir):

            label_dir = os.path.join(self.image_dir, label)
            label_idx = self.classes.index(label)
            
            for img_name in os.listdir(label_dir):
                
                img_idx = img_name[:12] # This is specific for ISIC2019
                img_path = os.path.join(label_dir, img_name)
                
                if self.mask_dir is not None:
                    mask_path = os.path.join(self.mask_dir, label, f"{img_idx}.png")
                else:
                    mask_path = None
                    
                self.idx.append((img_path, label_idx, img_idx, mask_path))
                self.targets.append(label_idx)
                
    def __len__(self):
        return len(self.idx)                    

    def __getitem__(self, index):
        
        img_path, label, img_idx, mask_path = self.idx[index]
        img = Image.open(img_path)
        img = self.img_transform(img)
        
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = self.mask_transform(mask)
            mask = ~mask.bool()
            mask = mask.float()
        else:
            mask = torch.zeros((1, img.shape[1], img.shape[2]))
    
        return img, label, img_idx, mask

def replace_values(x):
    """ 
        Function to replace values in a tensor with 0 or 1. Needed for the mask.
        Some values in the mask are not exactly 0 or 1, but very close to it.
    """
    return torch.where(x < 0.5, torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32))

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im and args.input_size != 224:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def build_dataset(is_train, args):
    
    img_root = os.path.join(args.data_path, 'train' if is_train else 'val')
            
    if args.batch_aug:
        img_transform = build_transform(is_train, args)
    else:
        img_transform = transforms.Compose([   
            #transforms.Resize(size=(224, 224)), # Input images are already 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    
    mask_is_val = False
    if args.mask:   
        if args.mask_val: 
            mask_is_val = True    
        mask_root = os.path.join(args.mask_path, args.mask_is_train_path if is_train else args.mask_val)
        mask_transform = transforms.Compose([
            #transforms.Resize(int((256 / 224) * args.input_size), interpolation=3),
            #transforms.Resize(size=(args.input_size, args.input_size)), # Equivalent to above if input_size=224
            #transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Lambda(replace_values)
        ])
    else:
        mask_root = None
        mask_transform = None
        
    dataset = ISIC2019_Dataset(image_dir = img_root,
                                mask_dir = mask_root, 
                                img_transform=img_transform,
                                mask_is_val=mask_is_val,
                                mask_transform=mask_transform)
    nb_classes = len(dataset.classes)
            
    return dataset, nb_classes
