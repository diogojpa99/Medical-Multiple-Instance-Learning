datapath="../Datasets/ISIC2019bea_mel_nevus_limpo"

################################## MIL - Resnet18 #########################################

mil_t='instance'
pool='mask_avg'
lr=2e-4
sched='cosine'
drop=0.0
opt='adamw'
now=$(date +"%Y%m%d")


logdir="MIL-resnet18-$mil_t-$pool-lr_init_$lr-Dropout_$drop-Mask_TrainON_ValOFF_Time_$now"
echo "----------------- Output dir: $logdir --------------------"

python main.py \
--project_name "Thesis" \
--run_name "$logdir" \
--hardware "Server" \
--gpu "cuda:1" \
--debug \
--wandb \
--mask \
--mask_path "../Datasets/Fine_MASKS_bea_all" \
--feature_extractor "resnet18.tv_in1k" \
--num_workers 8 \
--batch_size 512 \
--epochs 3 \
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
--warmup_epochs 3 \
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


logdir="MIL-resnet18-$mil_t-$pool-lr_init_$lr-Dropout_$drop-Mask_TrainON_ValON_Time_$now"
echo "----------------- Output dir: $logdir --------------------"

python main.py \
--project_name "Thesis" \
--run_name "$logdir" \
--hardware "Server" \
--gpu "cuda:1" \
--debug \
--wandb \
--mask \
--mask_val "val" \
--mask_path "../Datasets/Fine_MASKS_bea_all" \
--feature_extractor "resnet18.tv_in1k" \
--num_workers 8 \
--batch_size 512 \
--epochs 3 \
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
--warmup_epochs 3 \
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
