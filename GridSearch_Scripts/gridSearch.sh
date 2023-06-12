datapath="../Data/ISIC2019bea_mel_nevus_limpo"


################################## MIL - DEiT_Small #########################################

mil_types=( 'instance' 'embedding' )
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

        logdir="MIL-deitSmall-$mil_t-$pool-lr_init_$lr-Dropout_$drop-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "deit_small_patch16_224" \
        --num_workers 10 \
        --batch_size 256 \
        --epochs 200 \
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
        --patience 150 \
        --counter_saver_threshold 100 \
        --delta 0.0 \
        --batch_aug \
        --color-jitter 0.0 \
        --data_path "$datapath" \
        --loss_scaler \
        --output_dir "deitSmall/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

################################## MIL - Resnet18 #########################################

mil_types=('instance' 'embedding')
pooling_types=( 'mask_avg' 'mask_max' )
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'
now=$(date +"%Y%m%d")


for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-resnet18-$mil_t-$pool-lr_init_$lr-Dropout_$drop-Mask_TrainON_ValON_Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --mask \
        --mask_val "val" \
        --mask_path "../Data/Fine_MASKS_bea_all" \
        --feature_extractor "resnet18.tv_in1k" \
        --num_workers 10 \
        --batch_size 512 \
        --epochs 200 \
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
        --output_dir "resnet18/mask_val_ON/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-resnet18-$mil_t-$pool-lr_init_$lr-Dropout_$drop-Mask_TrainON_ValOFF_Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --mask \
        --mask_path "../Data/Fine_MASKS_bea_all" \
        --feature_extractor "resnet18.tv_in1k" \
        --num_workers 10 \
        --batch_size 512 \
        --epochs 200 \
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
        --output_dir "resnet18/mask_val_ON/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

################################## MIL - Resnet18 #########################################

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

        logdir="MIL-resnet18-$mil_t-$pool-lr_init_$lr-Dropout_$drop-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "resnet18.tv_in1k" \
        --num_workers 10 \
        --batch_size 512 \
        --epochs 200 \
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
        --output_dir "resnet18/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

################################## MIL - Resnet50 #########################################

mil_types=('instance' 'embedding')
pooling_types=( 'mask_avg' 'mask_max' )
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'
now=$(date +"%Y%m%d")


for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-resnet50-$mil_t-$pool-lr_init_$lr-Dropout_$drop-Mask_TrainON_ValON_Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --mask \
        --mask_val "val" \
        --mask_path "../Data/Fine_MASKS_bea_all" \
        --feature_extractor "resnet50.tv_in1k" \
        --num_workers 10 \
        --batch_size 512 \
        --epochs 200 \
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
        --output_dir "resnet50/mask_val_ON/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

################################## MIL - DEiT_Small #########################################

mil_types=( 'instance' 'embedding' )
pooling_types=( 'mask_max' 'mask_avg' )
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'
now=$(date +"%Y%m%d")


for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-deitSmall-$mil_t-$pool-lr_init_$lr-Dropout_$drop-Mask_TrainON_ValOFF-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --mask \
        --mask_path "../Data/Fine_MASKS_bea_all" \
        --feature_extractor "deit_small_patch16_224" \
        --num_workers 10 \
        --batch_size 256 \
        --epochs 200 \
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
        --patience 150 \
        --counter_saver_threshold 100 \
        --delta 0.0 \
        --batch_aug \
        --color-jitter 0.0 \
        --data_path "$datapath" \
        --loss_scaler \
        --output_dir "deitSmall/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done


################################## MIL - VGG16 #########################################

mil_types=( 'instance' 'embedding' )
pooling_types=( 'max' 'avg' 'topk'  )
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'
now=$(date +"%Y%m%d")


for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-VGG16-$mil_t-$pool-lr_init_$lr-Dropout_$drop-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "vgg16.tv_in1k" \
        --num_workers 10 \
        --batch_size 256 \
        --epochs 200 \
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
        --output_dir "vgg16/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

################################## MIL - DEiT_Base #########################################

mil_types=('instance' 'embedding')
pooling_types=( 'max' 'avg' 'topk' )
lr=2e-4
sched='cosine'
drop=0.2
opt='adamw'
now=$(date +"%Y%m%d")


for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-deitBase-$mil_t-$pool-lr_init_$lr-Dropout_$drop-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --feature_extractor "deit_base_patch16_224" \
        --num_workers 10 \
        --batch_size 128 \
        --epochs 200 \
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
        --patience 150 \
        --counter_saver_threshold 100 \
        --delta 0.0 \
        --batch_aug \
        --color-jitter 0.0 \
        --data_path "$datapath" \
        --loss_scaler \
        --output_dir "deitBase/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done