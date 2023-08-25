import data_setup, utils, mil, engine, visualization
import Feature_Extractors.ResNet as resnet
import Feature_Extractors.VGG as vgg
import Feature_Extractors.DenseNet as densenet
import Feature_Extractors.EfficientNet as efficientnet
import Feature_Extractors.EViT as evit
import Feature_Extractors.DEiT as deit, Feature_Extractors.ViT as vit


import torch
import torch.backends.cudnn as cudnn
from timm.optim import create_optimizer
from timm.utils import get_state_dict, ModelEma, NativeScaler
from timm.scheduler import create_scheduler
from timm.models import create_model
import torch.optim as optim

import argparse
from pathlib import Path
import time
import datetime
import numpy as np
import wandb

from typing import List, Union

import os
#os.environ["WANDB_MODE"] = "offline"

def get_args_parser():
   
    parser = argparse.ArgumentParser('Deep-MIL', add_help=False)
    
    ## Add arguments here
    parser.add_argument('--output_dir', default='MIL-Output', help='path where to save, empty for no saving')
    parser.add_argument('--data_path', default='', help='path to input file')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--gpu', default='cuda:1', help='GPU id to use.')
    parser.add_argument('--dataset', default='ISIC2019-Clean', type=str, choices=['ISIC2019-Clean', 'PH2', 'Derm7pt'], metavar='DATASET')
    
    # Wanb parameters
    parser.add_argument('--project_name', default='Thesis', help='name of the project')
    parser.add_argument('--hardware', default='Server', choices=['Server', 'Colab', 'MyPC'], help='hardware used')
    parser.add_argument('--run_name', default='MIL', help='name of the run')
    parser.add_argument('--wandb_flag', action='store_false', default=True, help='whether to use wandb')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    
    # Data parameters
    parser.add_argument('--input_size', default=224, type=int, help='image size')
    parser.add_argument('--patch_size', default=16, type=int, help='patch size')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--pin_mem', default=True, type=bool, help='pin memory')
    
    # Training parameters
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--training', default=True, type=bool, help='training or testing')
    parser.add_argument('--finetune', default=False, type=bool, help='finetune or not')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--classifier_warmup_epochs', type=int, default=5, metavar='N')
    
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='drop rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
        
    # MIL parameters
    parser.add_argument('--pooling_type', default='max', choices=['max', 'avg', 'topk', 'mask_avg', 'mask_max'], type=str, help="")
    parser.add_argument('--mil_type', default='instance', choices=['instance', 'attention', 'embedding'], type=str, help="")
    parser.add_argument('--topk', default=25, type=int, help='topk for topk pooling')
    
    # Feature Extractor parameters
    parser.add_argument('--feature_extractor', default='resnet18.tv_in1k', type=str, metavar='MODEL',
                        choices=['resnet18.tv_in1k', 'resnet50.tv_in1k', 'deit_small_patch16_224', 'deit_base_patch16_224','vgg16.tv_in1k',
                                 'densenet169.tv_in1k', 'efficientnet_b3', 'deit_small_patch16_shrink_base'], 
                        help='Feature Extractor model architecture (default: "resnet18")')
    
    parser.add_argument('--pretrained_feature_extractor_path', default='https://download.pytorch.org/models/resnet18-5c106cde.pth', type=str, 
                        choices=['https://download.pytorch.org/models/resnet18-5c106cde.pth', 
                                 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 
                                 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
                                 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
                                 'https://download.pytorch.org/models/vgg16-397923af.pth',
                                 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
                                 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth',
                                 'Feature_Extractors/Pretrained_EViTs/evit-0.7-img224-deit-s.pth',
                                 'Feature_Extractors/Pretrained_EViTs/evit-0.7-fuse-img224-deit-s.pth'], 
                        metavar='PATH', help="Download the pretrained feature extractor from the given path.")
    
    parser.add_argument('--feature_extractor_pretrained_dataset', default='ImageNet1k', type=str, metavar='DATASET')
    parser.add_argument('--efficientnet_feature_flag', action='store_true', default=False, help='efficientnet feature extractor flag')

    # Evaluation parameters
    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate model on validation set')
    parser.add_argument('--evaluate_model_name', default='MIL_model_0.pth', type=str, help="")
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize model')
    parser.add_argument('--images_path', default='', type=str, help="")
    parser.add_argument('--visualize_num_images', default=8, type=int, help="")
    parser.add_argument('--vis_mask_path', default='', type=str, help="")
    parser.add_argument('--vis_num', default=1, type=int, help="")
    
    # Imbalanced dataset parameters
    parser.add_argument('--class_weights', action='store_true', default=True, help='Enabling class weighting')
    parser.add_argument('--class_weights_type', default='Manual', choices=['Median', 'Manual'], type=str, help="")
    
    # Optimizer parameters 
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', choices=['adamw', 'sgd'],
                        help='Optimizer (default: "adamw")')
    
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')

    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters 
    parser.add_argument('--lr_scheduler', action='store_true', default=False)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', choices=['step', 'multistep', 'cosine', 'plateau','poly', 'exp'],
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    
    # * Lr Cosine Scheduler Parameters
    parser.add_argument('--cosine_one_cycle', type=bool, default=False, help='Only use cosine one cycle lr scheduler')
    parser.add_argument('--lr_k_decay', type=float, default=1.0, help='LR k rate (default: 1.0)')
    parser.add_argument('--lr_cycle_mul', type=float, default=1.0, help='LR cycle mul (default: 1.0)')
    parser.add_argument('--lr_cycle_decay', type=float, default=1.0, help='LR cycle decay (default: 1.0)')
    parser.add_argument('--lr_cycle_limit', type=int, default=1, help= 'LR cycle limit(default: 1)')
    
    parser.add_argument('--lr-noise', type=Union[float, List[float]], default=None, help='Add noise to lr')
    parser.add_argument('--lr-noise-pct', type=float, default=0.1, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.1)')
    parser.add_argument('--lr-noise-std', type=float, default=0.05, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 0.05)')
    
    # * Warmup parameters
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_lr', type=float, default=1e-3, metavar='LR',
                        help='warmup learning rate (default: 1e-3)')
    
    parser.add_argument('--min_lr', type=float, default=1e-4, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')

    # * StepLR parameters
    parser.add_argument('--decay_epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    
    # * MultiStepLRScheduler parameters
    parser.add_argument('--decay_milestones', type=List[int], nargs='+', default=(10, 15), 
                        help='epochs at which to decay learning rate')
    
    # * The decay rate is transversal to many schedulers | However it has a different meaning for each scheduler
    # MultiStepLR: decay factor of learning rate | PolynomialLR: power factor | ExpLR: decay factor of learning rate
    parser.add_argument('--decay_rate', '--dr', type=float, default=1., metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Model EMA parameters -> Exponential Moving Average Model
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=12, metavar='N')
    parser.add_argument('--delta', type=float, default=0.0, metavar='N')
    parser.add_argument('--counter_saver_threshold', type=int, default=12, metavar='N')
    
    # Data augmentation parameters 
    parser.add_argument('--batch_aug', action='store_true', default=False, help='whether to augment batch')
    parser.add_argument('--color-jitter', type=float, default=0.0, metavar='PCT', help='Color jitter factor (default: 0.)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + \
                        "(default: rand-m9-mstd0.5-inc1)'),
    
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.1, metavar='PCT', help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='const', help='Random erase mode (default: "const")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')
    
    # Loss scaler parameters
    parser.add_argument('--loss_scaler', action='store_true', default=False, help='Use loss scaler')

    # Segmentation Mask parameters
    parser.add_argument('--mask', action='store_true', default=False, help='Use segmentation mask')
    parser.add_argument('--mask_path', default='', type=str, help='path to segmentation mask')
    parser.add_argument('--mask_is_train_path', default='train', type=str, help='path to train directory of multiclass binary segmentation masks')
    parser.add_argument('--mask_val', default='', type=str, help='path to val directory of binary segmentation masks')
    
    # EViT Parameters
    parser.add_argument('--fuse_token', action='store_true', help='Whether to fuse the inattentive tokens')
    parser.add_argument('--fuse_token_filled', action='store_true', help='Whether to fuse the inattentive tokens with filled tokens')
    parser.add_argument('--base_keep_rate', type=float, default=0.7, help='Base keep rate (default: 0.7)')
    parser.add_argument('--drop_loc', default='(3, 6, 9)', type=str, help='the layer indices for shrinking inattentive tokens')
    
    # *DeiT with cls token
    parser.add_argument('--cls_token', action='store_true', help='whether to add cls token')
    
    # Multiclass Parameters
    parser.add_argument('--multiclass_method', type=str, default='first', choices=['first', 'second'],
                        help="For MIL there is no absolute solution for the multiclass problem. This first method consists in applying the\
                            MIL-Pooling function on the scores of the different classes, and then apply the softmax function to the resulting tensor.\
                            The second method consists in applying the Softmax function on the scores of the different patch, and \
                            then apply the MIL-Pooling function to the resulting tensor.")

    return parser

def main(args):
    
    if args.visualize or args.evaluate or args.finetune:
        args.training = False

    # Start a new wandb run to track this script
    if args.wandb_flag:
        wandb.init(
            project=args.project_name,
            #mode="offline",
            config={
            "Feature Extractor model": args.feature_extractor,
            "Feature Extractor dataset": args.feature_extractor_pretrained_dataset,
            "Model": "MIL", "MIL type": args.mil_type,
            "Pooling": args.pooling_type, "Topk": args.topk,
            "Dataset": args.dataset,
            "epochs": args.epochs,"batch_size": args.batch_size,
            "warmup_epochs": args.warmup_epochs, "Warmup lr": args.warmup_lr,
            "cooldown_epochs": args.cooldown_epochs, "patience_epochs": args.patience_epochs,
            "lr_scheduler": args.sched, "lr": args.lr, "min_lr": args.min_lr,
            "drop": args.drop, "weight_decay": args.weight_decay,
            "optimizer": args.opt, "momentum": args.momentum,
            "seed": args.seed, "class_weights": args.class_weights,
            "early_stopping_patience": args.patience, "early_stopping_delta": args.delta,
            "model_ema": args.model_ema, "Batch_augmentation": args.batch_aug, "Loss_scaler": args.loss_scaler,
            "PC": args.hardware,
            }
        )
        wandb.run.name = args.run_name
        
        """ if args.debug:
            wandb=print """
    
    if args.training: # Print arguments
        print("----------------- Args -------------------")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print("------------------------------------------\n")
        
    # Validation regarding the use of segmentation masks
    if args.pooling_type == 'mask_max' or args.pooling_type == 'mask_avg':
        if not args.mask:
            raise ValueError('Masked pooling type requires mask flag to be True.') 
        if args.feature_extractor == 'deit_small_patch16_shrink_base':
            raise ValueError('Masked pooling type is not supported for this feature extractor.')
    
    device = args.gpu if torch.cuda.is_available() else "cpu" # Set device
    print(f"Device: {device}\n")
    
    utils.configure_seed(args.seed) # Fix the seed for reproducibility
    cudnn.benchmark = True
    
    ################## Data Setup ##################
    if args.data_path:
    
        train_set, args.nb_classes = data_setup.build_dataset(is_train=True, args=args)
        val_set,_ = data_setup.build_dataset(is_train=False, args=args)
        
        ## Data Loaders 
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(val_set)
        
        data_loader_train = torch.utils.data.DataLoader(
            train_set, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_val = torch.utils.data.DataLoader(
            val_set, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
     
    ############################ Define the Feature Extractor ############################
    
    num_chs = 256
    model_args = {}
    if args.feature_extractor == 'resnet18.tv_in1k':
        """ feature_extractor = ResNet.ResNet(block=ResNet.BasicBlock, 
                                          layers=[2, 2, 2], 
                                          desired_output_size=(args.input_size // args.patch_size)) """            
        model_args = dict(block=resnet.BasicBlock, 
                          layers=[2, 2, 2], 
                          desired_output_size=(args.input_size // args.patch_size),
                          drop_rate=args.drop,
                          drop_path_rate=0.,
                          drop_block_rate=0.,
                          feature_extractor=True)
    elif args.feature_extractor == 'resnet50.tv_in1k':
        model_args = dict(block=resnet.Bottleneck, 
                          layers=[3, 4, 6], 
                          desired_output_size=(args.input_size // args.patch_size),
                          drop_rate=args.drop,
                          drop_path_rate=0.,
                          drop_block_rate=0.,
                          feature_extractor=True)
    elif args.feature_extractor == 'deit_small_patch16_224':
        num_chs = 384
        model_args = dict(img_size=args.input_size,
                          patch_size=args.patch_size,
                          embed_dim=num_chs, 
                          depth=12, 
                          num_heads=6,
                          drop_rate=args.drop,
                          drop_path_rate=args.drop_path,
                          attn_drop_rate=0.,
                          feature_extractor=True)
    elif args.feature_extractor == 'deit_base_patch16_224':
        num_chs = 768
        model_args = dict(img_size=args.input_size,
                          patch_size=args.patch_size,
                          embed_dim=num_chs, 
                          depth=12, 
                          num_heads=12,
                          drop_rate=args.drop,
                          drop_path_rate=args.drop_path,
                          attn_drop_rate=0.,
                          feature_extractor=True)
    elif args.feature_extractor == 'vgg16.tv_in1k': 
        model_args = dict(cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
                          drop_rate=args.drop,
                          feature_extractor=True)
    elif args.feature_extractor == 'densenet169.tv_in1k':
        model_args = dict(block_config=(6, 12, 32),
                          drop_rate=args.drop,
                          feature_extractor=True)
    elif args.feature_extractor == 'efficientnet_b3':
        model_args = dict(drop_rate=0.3,
                          drop_path_rate=0.2,
                          feature_extractor=True,
                          desired_output_size=(args.input_size // args.patch_size))
    elif args.feature_extractor == 'deit_small_patch16_224':
        num_chs = 384
        model_args = dict(img_size=args.input_size,
                          patch_size=args.patch_size,
                          embed_dim=num_chs, 
                          depth=12, 
                          num_heads=6,
                          drop_rate=args.drop,
                          drop_path_rate=args.drop_path,
                          attn_drop_rate=0.,
                          feature_extractor=True)
    elif args.feature_extractor == 'deit_base_patch16_224':
        num_chs = 768
        model_args = dict(img_size=args.input_size,
                          patch_size=args.patch_size,
                          embed_dim=num_chs, 
                          depth=12, 
                          num_heads=12,
                          drop_rate=args.drop,
                          drop_path_rate=args.drop_path,
                          attn_drop_rate=0.,
                          feature_extractor=True)
    elif args.feature_extractor == 'deit_small_patch16_shrink_base':
        num_chs = 384
        model_args = dict(base_keep_rate=args.base_keep_rate,
                          drop_loc=eval(args.drop_loc),
                          num_classes=args.nb_classes,
                          drop_rate=args.drop,
                          drop_path_rate=args.drop_path,
                          drop_block_rate=None,
                          fuse_token=args.fuse_token,
                          img_size=(args.input_size, args.input_size),
                          feature_extractor=True)
    else:
        raise NotImplementedError('This MIL implementation does not support that feature extractor...Yet!')
     
    feature_extractor = create_model(model_name=args.feature_extractor,
                                    pretrained=False,  
                                    **dict(model_args))
    
    if args.feature_extractor=='resnet18.tv_in1k' or args.feature_extractor=='resnet50.tv_in1k':
        if feature_extractor.feature_info[-1]['module'] == 'layer3':
            num_chs = feature_extractor.feature_info[-1]['num_chs']
    elif args.feature_extractor=='vgg16.tv_in1k':
        num_chs = feature_extractor.feature_info[-1]['num_chs']
    elif args.feature_extractor=='densenet169.tv_in1k':
        num_chs = feature_extractor.feature_info[-1]['num_chs']
    elif args.feature_extractor=='efficientnet_b3':
        num_chs = list(feature_extractor.named_parameters())[-1][1].shape[0]
    
    feature_extractor.to(device)
    args.pretrained_feature_extractor_path = mil.Pretrained_Feature_Extractures(args.feature_extractor, args)
    if args.training:
        if args.pretrained_feature_extractor_path: # Load the pretrained feature extractor
            utils.Load_Pretrained_FeatureExtractor(args.pretrained_feature_extractor_path, feature_extractor, args)
            
    ############################ Define the MIL Model ############################
    
    mil_args = dict(num_classes=args.nb_classes,
                    N=(args.input_size // args.patch_size) ** 2,
                    embedding_size=num_chs,
                    dropout=args.drop,
                    pooling_type=args.pooling_type,
                    is_training=args.training,
                    patch_extractor_model=args.feature_extractor,
                    patch_extractor=feature_extractor,
                    device=device,
                    args=args) 
    
    if args.mil_type == 'instance':
        model = mil.InstanceMIL(mil_type=args.mil_type, **dict(mil_args))
        
    elif args.mil_type == 'embedding':
        model = mil.EmbeddingMIL(mil_type=args.mil_type, **dict(mil_args))
        
    elif args.mil_type == 'attention':
        raise NotImplementedError('This MIL implementation does not support this MIL type..yet!')
        
    # TODO: Implement -> If finetune_ckpt then load the weights of the model
    # Only then can I do model.to(device) and model_ema.to(device)
    
    model.to(device)
    
    model_ema = None 
    if args.model_ema:
        model_ema = ModelEma(model,decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else '', resume='')
            
    ################## Define Training Parameters ##################
    
    # Define the output directory
    output_dir = Path(args.output_dir)
        
    if args.data_path:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
        print(f"Number of parameters: {n_parameters}\n")
        
        # (1) Define the class weights
        class_weights = engine.Class_Weighting(train_set, val_set, device, args)
        
        # (2) Define the optimizer
        optimizer = create_optimizer(args=args, model=model)

        # Define the loss scaler
        loss_scaler = NativeScaler() if args.loss_scaler else None

        # (3) Create scheduler
        if args.lr_scheduler:
            if args.sched == 'exp':
                lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
            else:    
                lr_scheduler,_ = create_scheduler(args, optimizer)
        
        # (4) Define the loss function with class weighting
        #criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = torch.nn.NLLLoss(weight=class_weights)
        
    ########################## Training or evaluating ###########################
    
    if args.resume:
        utils.Load_Pretrained_MIL_Model(path=args.resume, model=model, args=args)
        
        if args.visualize:
            print('******* Starting visualization process. *******')
            if args.nb_classes == 2:
                val_loader = visualization.VisualizationLoader_Binary(val_set, args)
                if args.mil_type == 'instance':
                    visualization.Visualize_Activation_Instance_Binary(model=model, dataloader=data_loader_val, device=device, outputdir=output_dir, args=args)
                elif args.mil_type == 'embedding':
                    visualization.Visualize_Activation_Embedding_Binary(model=model, dataloader=val_loader, device=device, outputdir=output_dir, args=args)
            
            return

        elif args.evaluate:
            print('******* Starting evaluation process. *******')
            total_time_str = 0
            best_results = engine.evaluation(model=model,
                                             dataloader=data_loader_val,
                                             criterion=torch.nn.NLLLoss(), 
                                             epoch=0, 
                                             device=device,
                                             args=args)
            
            if args.feature_extractor in mil.deits_backbones and args.cls_token:
                print(f"[INFO] CLS token was selected {(best_results['count_tokens']*100):.2f}% of the times.")             
            elif args.feature_extractor in mil.evits_backbones and args.fuse_token:
                print(f"[INFO] Fused tokens were selected {(best_results['count_tokens']*100):.2f}% of the times.")
                            
    elif args.training or args.finetune:
        
        start_time = time.time()  
        train_results = {'loss': [], 'acc': [] , 'lr': []}
        val_results = {'loss': [], 'acc': [], 'f1': [], 'cf_matrix': [], 'bacc': [], 'precision': [], 'recall': []}
        best_val_bacc = 0.0
        freeze_patch_extractor_flag = False      
        early_stopping = engine.EarlyStopping(patience=args.patience, verbose=True, delta=args.delta, path=str(output_dir) +'/checkpoint.pth')
       
        print(f"******* Start training for {(args.epochs + args.cooldown_epochs)} epochs. *******") 
        for epoch in range(args.start_epoch, (args.epochs + args.cooldown_epochs)):
            
            if epoch < args.classifier_warmup_epochs and freeze_patch_extractor_flag==False:
                freeze_patch_extractor_flag = engine.train_patch_extractor(model=model,
                                                                           current_epoch=epoch,
                                                                           warmup_epochs=args.classifier_warmup_epochs,
                                                                           flag=freeze_patch_extractor_flag,
                                                                           args=args)
            elif epoch >= args.classifier_warmup_epochs and freeze_patch_extractor_flag==True:
                freeze_patch_extractor_flag = engine.train_patch_extractor(model=model,
                                                                           current_epoch=epoch,
                                                                           warmup_epochs=args.classifier_warmup_epochs,
                                                                           flag=freeze_patch_extractor_flag,
                                                                           args=args)
            train_stats = engine.train_step(model=model,
                                            dataloader=data_loader_train,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            device=device,  
                                            epoch=epoch+1,
                                            wandb=wandb,
                                            loss_scaler=loss_scaler,
                                            max_norm=args.clip_grad,
                                            lr_scheduler=lr_scheduler,
                                            model_ema=model_ema,
                                            args=args)
        
            if args.lr_scheduler:
                lr_scheduler.step(epoch+1)

            results = engine.evaluation(model=model,
                                        dataloader=data_loader_val,
                                        criterion=criterion,
                                        device=device,
                                        epoch=epoch+1,
                                        wandb=wandb,
                                        args=args) 
            
            # Update results dictionary
            train_results['loss'].append(train_stats['train_loss']); train_results['acc'].append(train_stats['train_acc']); train_results['lr'].append(train_stats['train_lr'])
            val_results['acc'].append(results['acc1']); val_results['loss'].append(results['loss']); val_results['f1'].append(results['f1_score'])
            val_results['cf_matrix'].append(results['confusion_matrix']); val_results['precision'].append(results['precision'])
            val_results['recall'].append(results['recall']); val_results['bacc'].append(results['bacc'])
            
            print(f"Epoch: {epoch+1} | lr: {train_stats['train_lr']:.5f} | Train Loss: {train_stats['train_loss']:.4f} | Train Acc: {train_stats['train_acc']:.4f} |",
                  f"Val. Loss: {results['loss']:.4f} | Val. Acc: {results['acc1']:.4f} | Val. Bacc: {results['bacc']:.4f} | F1-score: {np.mean(results['f1_score']):.4f}")
                            
            if results['bacc'] > best_val_bacc and early_stopping.counter < args.counter_saver_threshold:
                # Only want to save the best checkpoints if the best val bacc and the early stopping counter is less than the threshold
                best_val_bacc = results['bacc']
                checkpoint_paths = [output_dir / f'MIL-{args.mil_type}-{args.pooling_type}-best_checkpoint.pth']
                best_results = results
                for checkpoint_path in checkpoint_paths:
                    checkpoint_dict = {
                        'model':model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                    if args.lr_scheduler:
                        checkpoint_dict['lr_scheduler'] = lr_scheduler.state_dict()
                    if model_ema is not None:
                        checkpoint_dict['model_ema'] = get_state_dict(model_ema)
                    utils.save_on_master(checkpoint_dict, checkpoint_path)
                print(f"\tBest Val. Bacc: {(best_val_bacc*100):.2f}% |[INFO] Saving model as 'best_checkpoint.pth'")
                        
            # Early stopping
            early_stopping(results['loss'], model)
            if early_stopping.early_stop:
                print("\t[INFO] Early stopping - Stop training")
                break
            
            if epoch < 1 and args.mil_type == 'instance':
                print(f"\t[INFO] Number of patches used: {model.num_patches}" )
            
        # Compute the total training time
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        print('\n---------------- Train stats for the last epoch ----------------\n',
            f"Acc: {train_stats['acc1']:.3f} | Bacc: {train_stats['bacc']:.3f} | F1-score: {np.mean(train_stats['f1_score']):.3f} | \n",
            f"Class-to-idx: {train_set.class_to_idx} | \n",
            f"Precisions: {best_results['precision']} | \n",
            f"Recalls: {best_results['recall']} | \n",
            f"Confusion Matrix: {train_stats['confusion_matrix']}\n",
            f"Training time {total_time_str}\n")
        
        utils.plot_loss_and_acc_curves(train_results, val_results, output_dir=output_dir, args=args)

    utils.plot_confusion_matrix(best_results["confusion_matrix"], train_set.class_to_idx, output_dir=output_dir, args=args)
            
    print('\n---------------- Val. stats for the best model ----------------\n',
        f"Acc: {best_results['acc1']} | Bacc: {best_results['bacc']} | F1-score: {np.mean(best_results['f1_score'])} | \n",
        f"Class-to-idx: {train_set.class_to_idx} | \n",
        f"Precisions: {best_results['precision']} | \n",
        f"Recalls: {best_results['recall']} | \n")
    
    if wandb!=print:
        wandb.log({"Best Val. Acc": best_results['acc1'], "Best Val. Bacc": best_results['bacc'], "Best Val. F1-score": np.mean(best_results['f1_score'])})
        wandb.log({"Training time": total_time_str})
        #wandb.finish()
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deep-MIL', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(args)