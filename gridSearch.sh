datapath="../Data/ISIC2019bea_mel_nevus_limpo"

################################## MIL - DenseNet #########################################

mil_types=('instance')
pooling_types=('max')
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'
now=$(date +"%Y%m%d")


for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-DenseNet169-$mil_t-$pool-lr_init_$lr-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --mask \
        --mask_val "val" \
        --mask_path "../Data/Fine_MASKS_bea_all" \
        --feature_extractor "densenet169.tv_in1k" \
        --num_workers 12 \
        --batch_size 128 \
        --epochs 80 \
        --input_size 224 \
        --mil_type $mil_t \
        --pooling_type $pool \
        --drop $drop \
        --opt "$opt" \
        --lr $lr \
        --lr_scheduler \
        --sched "$sched" \
        --lr_cycle_decay 0.8 \
        --min_lr 2e-6 \
        --warmup_epochs 5 \
        --warmup_lr 1e-6 \
        --patience 100 \
        --counter_saver_threshold 100 \
        --delta 0.0 \
        --batch_aug \
        --color-jitter 0.0 \
        --data_path "$datapath" \
        --loss_scaler \
        --output_dir "DenseNet/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

mil_types=('instance')
pooling_types=('topk')
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'
now=$(date +"%Y%m%d")


for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-DenseNet169-$mil_t-$pool-lr_init_$lr-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --mask \
        --mask_val "val" \
        --mask_path "../Data/Fine_MASKS_bea_all" \
        --feature_extractor "densenet169.tv_in1k" \
        --num_workers 12 \
        --batch_size 128 \
        --epochs 50 \
        --input_size 224 \
        --mil_type $mil_t \
        --pooling_type $pool \
        --drop $drop \
        --opt "$opt" \
        --lr $lr \
        --lr_scheduler \
        --sched "$sched" \
        --lr_cycle_decay 0.8 \
        --min_lr 2e-6 \
        --warmup_epochs 5 \
        --warmup_lr 1e-6 \
        --patience 100 \
        --counter_saver_threshold 100 \
        --delta 0.0 \
        --batch_aug \
        --color-jitter 0.0 \
        --data_path "$datapath" \
        --loss_scaler \
        --output_dir "DenseNet/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

################################## MIL - Resnet18 #########################################

mil_types=('instance' 'embedding')
pooling_types=('mask_max' 'mask_avg')
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'
now=$(date +"%Y%m%d")


for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-resnet18-$mil_t-$pool-lr_init_$lr-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "resnet18.tv_in1k" \
        --mask \
        --mask_val "val" \
        --mask_path "../Data/Fine_MASKS_bea_all" \
        --num_workers 12 \
        --batch_size 512 \
        --epochs 100 \
        --input_size 224 \
        --mil_type $mil_t \
        --pooling_type $pool \
        --drop $drop \
        --opt "$opt" \
        --lr $lr \
        --lr_scheduler \
        --sched "$sched" \
        --lr_cycle_decay 0.8 \
        --min_lr 2e-6 \
        --warmup_epochs 5 \
        --warmup_lr 1e-6 \
        --weight-decay 1e-6 \
        --patience 150 \
        --counter_saver_threshold 100 \
        --delta 0.0 \
        --batch_aug \
        --color-jitter 0.0 \
        --data_path "$datapath" \
        --loss_scaler \
        --output_dir "Resnet18_v2/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done


################################## MIL - EffNet #########################################

mil_types=('instance' 'embedding')
pooling_types=( 'max' 'avg' 'topk' )
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'
now=$(date +"%Y%m%d")


for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-EfficientNet_b3-$mil_t-$pool-lr_init_$lr-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "efficientnet_b3" \
        --mask \
        --mask_val "val" \
        --mask_path "../Data/Fine_MASKS_bea_all" \
        --num_workers 10 \
        --batch_size 128 \
        --epochs 90 \
        --input_size 224 \
        --mil_type $mil_t \
        --pooling_type $pool \
        --drop $drop \
        --opt "$opt" \
        --lr $lr \
        --lr_scheduler \
        --sched "$sched" \
        --lr_cycle_decay 0.8 \
        --min_lr 2e-6 \
        --warmup_epochs 5 \
        --warmup_lr 1e-6 \
        --patience 150 \
        --counter_saver_threshold 100 \
        --delta 0.0 \
        --batch_aug \
        --color-jitter 0.0 \
        --data_path "$datapath" \
        --loss_scaler \
        --output_dir "EffNet/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

################################## MIL - DEiT_Small #########################################

mil_types=('instance' 'embedding')
pooling_types=( 'max' 'avg' 'topk' )
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'
now=$(date +"%Y%m%d")


for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-deitSmall-$mil_t-$pool-lr_init_$lr-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "deit_small_patch16_224" \
        --mask \
        --mask_val "val" \
        --mask_path "../Data/Fine_MASKS_bea_all" \
        --num_workers 12 \
        --batch_size 256 \
        --epochs 90 \
        --input_size 224 \
        --mil_type $mil_t \
        --pooling_type $pool \
        --drop $drop \
        --drop_path 0.1 \
        --opt "$opt" \
        --lr $lr \
        --lr_scheduler \
        --sched "$sched" \
        --lr_cycle_decay 0.8 \
        --min_lr 2e-6 \
        --warmup_epochs 5 \
        --warmup_lr 1e-6 \
        --weight-decay 1e-6 \
        --patience 150 \
        --counter_saver_threshold 100 \
        --delta 0.0 \
        --batch_aug \
        --color-jitter 0.0 \
        --data_path "$datapath" \
        --loss_scaler \
        --output_dir "deitSmall_V2/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

################################## MIL - DEiT_Small #########################################

mil_types=('instance' 'embedding')
pooling_types=( 'max' 'avg' 'topk' )
lr=2e-5
sched='cosine'
drop=0.0
opt='adamw'
now=$(date +"%Y%m%d")


for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-deitSmall_v3-$mil_t-$pool-lr_init_$lr-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "deit_small_patch16_224" \
        --mask \
        --mask_val "val" \
        --mask_path "../Data/Fine_MASKS_bea_all" \
        --num_workers 12 \
        --batch_size 256 \
        --epochs 90 \
        --input_size 224 \
        --mil_type $mil_t \
        --pooling_type $pool \
        --drop $drop \
        --drop_path 0.1 \
        --opt "$opt" \
        --lr $lr \
        --lr_scheduler \
        --sched "$sched" \
        --lr_cycle_decay 0.8 \
        --min_lr 2e-6 \
        --warmup_epochs 5 \
        --warmup_lr 1e-6 \
        --weight-decay 1e-6 \
        --patience 150 \
        --counter_saver_threshold 100 \
        --delta 0.0 \
        --batch_aug \
        --color-jitter 0.0 \
        --data_path "$datapath" \
        --loss_scaler \
        --output_dir "deitSmall_V3/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

datapath="../Data/Bea_LIMPO/limpo/"

################################# MIL - Multiclass DEiT #########################################

mil_types=('embedding')
pooling_types=('topk')
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'

now=$(date +"%Y%m%d")

for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-Multiclass-deitSmall-$mil_t-$pool-lr_init_$lr-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "deit_small_patch16_224" \
        --drop_path 0.1 \
        --num_workers 12 \
        --batch_size 256 \
        --epochs 90 \
        --input_size 224 \
        --mil_type $mil_t \
        --pooling_type $pool \
        --drop $drop \
        --opt "$opt" \
        --lr $lr \
        --lr_scheduler \
        --sched "$sched" \
        --lr_cycle_decay 0.8 \
        --min_lr 2e-4 \
        --warmup_epochs 5 \
        --warmup_lr 1e-4 \
        --weight-decay 1e-6 \
        --patience 200 \
        --counter_saver_threshold 100 \
        --batch_aug \
        --loss_scaler \
        --data_path "$datapath" \
        --output_dir "Multiclass/DeitSmall/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done


################################## MIL - Multiclass Vgg16 #########################################

mil_types=('instance' 'embedding')
pooling_types=( 'avg' 'max' 'topk' )
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'

now=$(date +"%Y%m%d")

for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-Multiclass-VGG16-$mil_t-$pool-lr_init_$lr-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "vgg16.tv_in1k" \
        --num_workers 12 \
        --batch_size 128 \
        --epochs 100 \
        --input_size 224 \
        --mil_type $mil_t \
        --pooling_type $pool \
        --drop $drop \
        --opt "$opt" \
        --lr $lr \
        --lr_scheduler \
        --sched "$sched" \
        --lr_cycle_decay 0.8 \
        --min_lr 2e-4 \
        --warmup_epochs 5 \
        --warmup_lr 1e-4 \
        --weight-decay 1e-6 \
        --patience 200 \
        --counter_saver_threshold 100 \
        --batch_aug \
        --loss_scaler \
        --data_path "$datapath" \
        --output_dir "Multiclass/VGG16/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

################################## MIL - Multiclass DenseNet #########################################

now=$(date +"%Y%m%d")

for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-Multiclass-DenseNet169-$mil_t-$pool-lr_init_$lr-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "densenet169.tv_in1k" \
        --num_workers 12 \
        --batch_size 128 \
        --epochs 90 \
        --input_size 224 \
        --mil_type $mil_t \
        --pooling_type $pool \
        --drop $drop \
        --opt "$opt" \
        --lr $lr \
        --lr_scheduler \
        --sched "$sched" \
        --lr_cycle_decay 0.8 \
        --min_lr 2e-4 \
        --warmup_epochs 5 \
        --warmup_lr 1e-4 \
        --weight-decay 1e-6 \
        --patience 200 \
        --counter_saver_threshold 100 \
        --batch_aug \
        --loss_scaler \
        --data_path "$datapath" \
        --output_dir "Multiclass/DenseNet169/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done


################################## MIL - Multiclass Evit #########################################

now=$(date +"%Y%m%d")

for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-Multiclass-EvitSmall-$mil_t-$pool-lr_init_$lr-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "deit_small_patch16_shrink_base" \
        --drop_path 0.1 \
        --num_workers 12 \
        --batch_size 256 \
        --epochs 90 \
        --input_size 224 \
        --mil_type $mil_t \
        --pooling_type $pool \
        --drop $drop \
        --opt "$opt" \
        --lr $lr \
        --lr_scheduler \
        --sched "$sched" \
        --lr_cycle_decay 0.8 \
        --min_lr 2e-4 \
        --warmup_epochs 5 \
        --warmup_lr 1e-4 \
        --weight-decay 1e-6 \
        --patience 200 \
        --counter_saver_threshold 100 \
        --batch_aug \
        --loss_scaler \
        --data_path "$datapath" \
        --output_dir "Multiclass/EViTSmall/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done


###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################

################################## MIL-Evit-With many fused tokens in the place of the inattentive tokens #########################################

datapath="../Data/ISIC2019bea_mel_nevus_limpo"
now=$(date +"%Y%m%d")

mil_types=('instance' 'embedding')
pooling_types=( 'avg' 'max' 'topk' )
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'

for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-evit_small_07_fusedToken_filled-$mil_t-$pool-lr_init_$lr-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "deit_small_patch16_shrink_base" \
        --fuse_token \
        --fuse_token_filled \
        --drop_path 0.1 \
        --num_workers 12 \
        --batch_size 256 \
        --epochs 90 \
        --input_size 224 \
        --mil_type $mil_t \
        --pooling_type $pool \
        --drop $drop \
        --opt "$opt" \
        --lr $lr \
        --lr_scheduler \
        --sched "$sched" \
        --lr_cycle_decay 0.8 \
        --min_lr 2e-4 \
        --warmup_epochs 5 \
        --warmup_lr 1e-4 \
        --weight-decay 1e-6 \
        --patience 200 \
        --counter_saver_threshold 100 \
        --batch_aug \
        --loss_scaler \
        --data_path "$datapath" \
        --output_dir "evit_small_07/evit_small_07_fusedToken/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

################################## MIL-Evit-with fused token #########################################

datapath="../Data/ISIC2019bea_mel_nevus_limpo"
now=$(date +"%Y%m%d")

mil_types=('instance' 'embedding')
pooling_types=( 'avg' 'max' 'topk' )
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'

for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-evit_small_07_fusedToken_filled-$mil_t-$pool-lr_init_$lr-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "deit_small_patch16_shrink_base" \
        --fuse_token \
        --drop_path 0.1 \
        --num_workers 12 \
        --batch_size 256 \
        --epochs 90 \
        --input_size 224 \
        --mil_type $mil_t \
        --pooling_type $pool \
        --drop $drop \
        --opt "$opt" \
        --lr $lr \
        --lr_scheduler \
        --sched "$sched" \
        --lr_cycle_decay 0.8 \
        --min_lr 2e-4 \
        --warmup_epochs 5 \
        --warmup_lr 1e-4 \
        --weight-decay 1e-6 \
        --patience 200 \
        --counter_saver_threshold 100 \
        --batch_aug \
        --loss_scaler \
        --data_path "$datapath" \
        --output_dir "evit_small_07/evit_small_07_fusedToken/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done