import data_setup, utils, mil, engine, visualization, evaluation

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
import torch.optim as optim

import argparse
from pathlib import Path
import time
import datetime
import numpy as np
import wandb
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from typing import List, Union

import os
import gc
#os.environ["WANDB_MODE"] = "offline"

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def get_args_parser():
   
    parser = argparse.ArgumentParser('Deep-MIL', add_help=False)
    
    ## Add arguments here
    parser.add_argument('--output_dir', default='MIL-Output', help='path where to save, empty for no saving')
    parser.add_argument('--data_path', default='', help='path to input file')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--gpu', default='cuda:1', help='GPU id to use.')
    
    parser.add_argument('--train', action='store_true', default=False, help='Training mode.')
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluation mode.')
    parser.add_argument('--finetune', action='store_true', default=False, help='Finetune mode.')
    parser.add_argument('--infer', action='store_true', default=False, help='Inference mode.')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')
    
    # Dataset
    parser.add_argument('--dataset', default='ISIC2019-Clean', type=str, metavar='DATASET', help='Training dataset name')
    parser.add_argument('--testset', default=None, type=str, metavar='DATASET', help='Test dataset name')
    parser.add_argument('--dataset_type', default='Skin', type=str, choices=['Breast', 'Skin'], metavar='DATASET')
    
    # Wanb parameters
    parser.add_argument('--project_name', default='Thesis', help='name of the project')
    parser.add_argument('--hardware', default='Server', choices=['Server', 'Colab', 'MyPC'], help='hardware used')
    parser.add_argument('--run_name', default='MIL', help='name of the run')
    parser.add_argument('--wandb_flag', action='store_false', default=True, help='whether to use wandb')
    
    # Data parameters
    parser.add_argument('--input_size', default=224, type=int, help='image size')
    parser.add_argument('--patch_size', default=16, type=int, help='patch size')
    parser.add_argument('--nb_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--pin_mem', default=True, type=bool, help='pin memory')
    
    # Training parameters
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
        
    # MIL parameters
    parser.add_argument('--pooling_type', default='max', choices=['max', 'avg', 'topk', 'mask_avg', 'mask_max'], type=str, help="")
    parser.add_argument('--mil_type', default='instance', choices=['instance', 'attention', 'embedding'], type=str, help="")
    parser.add_argument('--topk', default=25.0, type=float, help='k (%) for top-k average pooling')
    
    # Feature Extractor parameters
    parser.add_argument('--feature_extractor', default='resnet18.tv_in1k', type=str, metavar='MODEL',
                        choices=['resnet18.tv_in1k', 'resnet50.tv_in1k', 'deit_small_patch16_224', 'deit_base_patch16_224','vgg16.tv_in1k',
                                 'densenet169.tv_in1k', 'efficientnet_b3', 'deit_small_patch16_shrink_base'], 
                        help='Feature Extractor model architecture (default: "resnet18")')
    
    parser.add_argument('--pretrained_feature_extractor_path', default=None, type=str, 
                        metavar='PATH', help="Download the pretrained feature extractor from the given path.")
    parser.add_argument('--from_pretrained_mil_model_flag', action='store_true', default=False, help='Whether to load the feature extractor from a pretrained MIL model')
    parser.add_argument('--feature_extractor_pretrained_dataset', default='ImageNet1k', type=str, metavar='DATASET')
    parser.add_argument('--efficientnet_feature_flag', action='store_true', default=False, help='efficientnet feature extractor flag')

    # Evaluation parameters
    parser.add_argument('--evaluate_model_name', default='MIL_model_0.pth', type=str, help="")
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    
    # Imbalanced dataset parameters
    parser.add_argument('--class_weights', default=None, choices=[None, 'balanced', 'median'], type=str, 
                        help="Class weights for loss function.")
    
    # Optimizer parameters 
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', choices=['adamw', 'adam', 'sgd'],
                        help='Optimizer (default: "adamw")')
    
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')

    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters 
    parser.add_argument('--sched', default=None, type=str, metavar='SCHEDULER', choices=['step', 'multistep', 'cosine', 'plateau','poly', 'exp'],
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
    parser.add_argument('--skin_batch_aug', action='store_true', default=False, help='whether to augment batch')
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
    parser.add_argument('--pos_encoding_flag', action='store_false', default=True, help='Whether to use positional encoding or not.')

    # Multiclass Parameters
    parser.add_argument('--multiclass_method', type=str, default='first', choices=['first', 'second'],
                        help="For MIL there is no absolute solution for the multiclass problem. This first method consists in applying the\
                            MIL-Pooling function on the scores of the different classes, and then apply the softmax function to the resulting tensor.\
                            The second method consists in applying the Softmax function on the scores of the different patch, and \
                            then apply the MIL-Pooling function to the resulting tensor.")

    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize model')
    parser.add_argument('--images_path', default='', type=str, help="")
    parser.add_argument('--visualize_num_images', default=8, type=int, help="")
    parser.add_argument('--vis_mask_path', default='', type=str, help="")
    parser.add_argument('--vis_num', default=5, type=int, help="")
    
    # ROI evaluation parameters
    parser.add_argument('--roi_eval', action='store_true', default=False, help='Evaluate the quality of the ROIs generated by the model')
    parser.add_argument('--roi_eval_type', type=str, default='basic', choices=['basic', 'IoU', 'dice'], help='ROI evaluation type.')
    parser.add_argument('--roi_patch_prob_threshold', type=float, default=0.5, help='Threshold for the patch probability being considered as a ROI')
    parser.add_argument('--roi_gradcam_threshold', type=float, default=0.0, help='Threshold for the GradCam probability being considered as a ROI')
    parser.add_argument('--roi_eval_vis', action='store_true', default=False, help='Visualize the ROIs generated by the model')
    
    # Breast Data setup parameters
    parser.add_argument('--breast_loader', default='Gray_PIL_Loader_Wo_He_No_Resize', type=str, metavar='LOADER', 
                        choices=['Gray_PIL_Loader', 'Gray_PIL_Loader_Wo_He', 'Gray_PIL_Loader_Wo_He_No_Resize'])
    parser.add_argument('--test_val_flag', action='store_true', default=False, help='If True, the test set is used as the validation set.')
    parser.add_argument('--train_val_split', default=0.8, type=float, help='Train-validation split')
    parser.add_argument('--breast_strong_aug', action='store_true', default=False, help='Whether to use strong augmentation for the breast dataset')
    parser.add_argument('--breast_clahe', action='store_true', default=False, help='Whether to use CLAHE for the breast dataset')
    parser.add_argument('--clahe_clip_limit', type=float, default=0.01, metavar='PCT', help='CLAHE clip limit (default: 0.01)')
    parser.add_argument('--breast_padding', action='store_true', default=False, help='Whether to use padding for the breast dataset') 
    parser.add_argument('--breast_antialias', action='store_true', default=False, help='Whether to use antialias for the breast dataset')
    parser.add_argument('--breast_transform_rgb', action='store_true', default=False, help='Whether to transform the breast dataset to RGB')
    parser.add_argument('--breast_transform_left', action='store_true', default=False, help='Whether to transform the breast dataset to left')
       
    # Dropout parameters
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate used in the classification head (default: 0.)')
    parser.add_argument('--pos_drop_rate', type=float, default=0.0, metavar='PCT', help='Dropout rate for the positional encoding (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT', help='Dropout rate for the attention layers (default: 0.)')
    parser.add_argument('--drop_layers_rate', type=float, default=0.0, metavar='PCT', help='Dropout rate for the layers (default: 0.)')
    parser.add_argument('--drop_block_rate', type=float, default=None, metavar='PCT', help='Dropout rate for the blocks (default: 0.)')
    
    # Grad Stats parameters
    parser.add_argument('--print_grad_stats', action='store_true', default=False, help='Whether to compute the gradient statistics')
    
    # Classifiers Warmup parameters
    parser.add_argument('--classifier_warmup_epochs', type=int, default=0, metavar='N', help='Epochs to warmup classifier')
    
    return parser

def main(args):
    
    if not args.train and not args.eval and not args.finetune and not args.infer and not args.visualize and not args.roi_eval:
        raise ValueError('The mode is not specified. Please specify the mode: --train, --eval, --finetune, --infer, --visualize, --roi_eval.')

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
            "Train_set": args.dataset, "Test_set": args.testset,  
            "Dataset type": args.dataset_type, "Input size": args.input_size, "Patch size": args.patch_size,
            "epochs": args.epochs,"batch_size": args.batch_size,
            "warmup_epochs": args.warmup_epochs, "Warmup lr": args.warmup_lr,
            "cooldown_epochs": args.cooldown_epochs, "patience_epochs": args.patience_epochs,
            "lr_scheduler": args.sched, "lr": args.lr, "min_lr": args.min_lr,
            "drop": args.drop, "weight_decay": args.weight_decay,
            "optimizer": args.opt, "momentum": args.momentum,
            "seed": args.seed, "class_weights": args.class_weights,
            "early_stopping_patience": args.patience, "early_stopping_delta": args.delta,
            "model_ema": args.model_ema, "Batch_augmentation": args.skin_batch_aug, "Loss_scaler": args.loss_scaler,
            "PC": args.hardware,
            }
        )
        wandb.run.name = args.run_name
        
    # if args.debug:
    #    wandb=print
    
    if args.train or args.finetune: 
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
        
        train_set, val_set = data_setup.Build_Dataset(data_path = args.data_path, input_size=args.input_size, args=args)
        
        ## Data Loaders 
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(val_set)
        
        data_loader_train = torch.utils.data.DataLoader(
            train_set, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(torch.cuda.is_available()),
            drop_last=True,
        )
        data_loader_val = torch.utils.data.DataLoader(
            val_set, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=(torch.cuda.is_available()),
            drop_last=False
        )
     
    ############################ Define the Feature Extractor ############################
    
    feature_extractor, num_chs = mil.Define_Feature_Extractor(args.feature_extractor, args.nb_classes, args)
    
    if args.finetune and args.pretrained_feature_extractor_path is None:
        args.pretrained_feature_extractor_path = mil.Pretrained_Feature_Extractures(args.feature_extractor, args)
        utils.Load_Pretrained_FeatureExtractor(args.pretrained_feature_extractor_path, feature_extractor, args)
            
    ############################ Define the MIL Model ############################
    
    if args.pooling_type == 'topk':
        utils.Adjust_topk(args.topk, args)
    
    mil_args = dict(num_classes=args.nb_classes,
                    N=(args.input_size // args.patch_size) ** 2,
                    embedding_size=num_chs,
                    dropout=args.drop,
                    pooling_type=args.pooling_type,
                    is_training=(args.train or args.finetune),
                    patch_extractor_model=args.feature_extractor,
                    patch_extractor=feature_extractor,
                    device=device,
                    args=args) 
    
    if args.mil_type == 'instance':
        model = mil.InstanceMIL(mil_type=args.mil_type, **dict(mil_args))
        
    elif args.mil_type == 'embedding':
        model = mil.EmbeddingMIL(mil_type=args.mil_type, **dict(mil_args))
        
    elif args.mil_type == 'attention':
        # TODO: Implement an attention MIL model
        raise NotImplementedError('This MIL implementation does not support this MIL type..yet!')
     
    if args.finetune and args.pretrained_feature_extractor_path is not None and args.from_pretrained_mil_model_flag:
        print(f"[Info] Loading the pretrained MIL model from:\n'{args.pretrained_feature_extractor_path}'")
        utils.Load_Pretrained_MIL_Model(path=args.pretrained_feature_extractor_path, model=model, args=args)
    if args.resume:
        print(f"[Info] Loading the finetuned model from:\n'{args.resume}'")
        utils.Load_Pretrained_MIL_Model(path=args.resume, model=model, args=args)

    feature_extractor.to(device)
    model.to(device)
    
    model_ema = None 
    if args.model_ema:
        model_ema = ModelEma(model,decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else '', resume='')
        #model_ema.ema.to(device)
            
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
        if args.sched == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
        else:    
            lr_scheduler,_ = create_scheduler(args, optimizer)
        
        # (4) Define the loss function with class weighting
        #criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = torch.nn.NLLLoss(weight=class_weights)
        
    ########################## Training or evaluating ###########################
    
    if args.resume:
                
        if args.visualize:
            print('******* Starting visualization process. *******')
            if args.nb_classes == 2:
                val_loader = visualization.VisualizationLoader_Binary(val_set, args)
                if args.mil_type == 'instance':
                    visualization.Visualize_Activation_Instance_Binary(model=model, dataloader=data_loader_val, device=device, outputdir=output_dir, args=args)
                elif args.mil_type == 'embedding':
                    visualization.Visualize_Activation_Embedding_Binary(model=model, dataloader=val_loader, device=device, outputdir=output_dir, args=args)
            
            return
        
        elif args.roi_eval:
            print('******* Starting ROI evaluation process. *******')
            if args.nb_classes == 2:
                if args.roi_eval_type == 'basic':
                    if args.roi_eval_vis:
                        evaluation.Basic_Eval_ROI_Vis_Inst(model=model, dataloader=data_loader_val, device=device, outputdir=output_dir, args=args)
                    else:
                        evaluation.Basic_Evaluation_ROIs(model=model, dataloader=data_loader_val, device=device, args=args)

        elif args.eval:
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
                
        elif args.infer:
            raise NotImplementedError('This MIL implementation does not support this MIL type..yet!')
            # TODO: Add inference code
            # Receive an input image
            # Infer with the already finetuned model
            # Return the prediction
            # Note: Should define its own inference_loader, and so on
                            
    elif args.train or args.finetune:
        
        start_time = time.time()  
        train_results = {'loss': [], 'acc': [] , 'lr': []}
        val_results = {'loss': [], 'acc': [], 'f1': [], 'cf_matrix': [], 'bacc': [], 'precision': [], 'recall': []}
        best_val_bacc = 0.0; best_results = None
        early_stopping = engine.EarlyStopping(patience=args.patience, verbose=True, delta=args.delta, path=str(output_dir) +'/checkpoint.pth')
        gradient_stats_tracker = utils.GradientStatsTracker(classifier_layer_names=['classifier'], warmup_epochs=args.warmup_epochs) if args.print_grad_stats else None

        print(f"******* Start training for {(args.epochs + args.cooldown_epochs)} epochs. *******") 
        for epoch in range(args.start_epoch, (args.epochs + args.cooldown_epochs)):
    
            engine.Classifier_Warmup(model, epoch, args.classifier_warmup_epochs, args)

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
                                            gradient_tracker=gradient_stats_tracker,
                                            args=args)
        
            if lr_scheduler is not None:
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
            
            if epoch % 10 == 0:
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
                    if args.sched is not None:
                        checkpoint_dict['lr_scheduler'] = lr_scheduler.state_dict()
                    if model_ema is not None:
                        checkpoint_dict['model_ema'] = get_state_dict(model_ema)
                    utils.save_on_master(checkpoint_dict, checkpoint_path)
                print(f"\tBest Val. Bacc: {(best_val_bacc*100):.2f}% | [INFO] Saving model as 'best_checkpoint.pth'")
                        
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
        wandb.finish()
        
    # Clean up
    # gc.collect()
    # torch.cuda.empty_cache()
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deep-MIL', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(args)