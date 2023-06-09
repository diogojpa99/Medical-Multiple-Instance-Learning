datapath="../Data/ISIC2019bea_mel_nevus_limpo"


################################## MIL - Resnet50 #########################################

mil_types=('instance' 'embedding')
pooling_types=( 'mask_avg' 'mask_max' 'max' 'avg' 'topk' )
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'
now=$(date +"%Y%m%d")


for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-resnet50-$mil_t-$pool-lr_init_$lr-Dropout_$drop-WarmupLr_1e-6-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --mask \
        --mask_path "../Data/Fine_MASKS_bea_all" \
        --feature_extractor "resnet50.tv_in1k" \
        --pretrained_feature_extractor_path "https://download.pytorch.org/models/resnet50-19c8e357.pth" \
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
        --output_dir "MIL/resnet50/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done

################################## MIL - Resnet18 #########################################

mil_types=('instance' 'embedding')
pooling_types=( 'mask_avg' 'mask_max' 'max' 'avg' 'topk' )
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'
now=$(date +"%Y%m%d")


for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

        logdir="MIL-resnet18-$mil_t-$pool-lr_init_$lr-Dropout_$drop-WarmupLr_1e-6-Time_$now"
        echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:1" \
        --mask \
        --mask_path "../Data/Fine_MASKS_bea_all" \
        --feature_extractor "resnet18.tv_in1k" \
        --pretrained_feature_extractor_path "https://download.pytorch.org/models/resnet18-5c106cde.pth" \
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
        --output_dir "MIL/resnet18/$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done