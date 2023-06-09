datapath="Data/ISIC2019bea_mel_nevus_limpo"

################################## GridSearch EViT #########################################

ckpt='EViT/Pretrained_Models/deit_base_patch16_224-b5f2ef4d.pth'
drop_loc="(3, 6, 9)"
keep_rate=0.6
lr=2e-4
now=$(date +"%Y%m%d")
dropout=(0.1 0.2 0.3)
sched='cosine'
opt='adamw'

for drop in "${dropout[@]}"
do

    logdir="EViT_Base-ckpt_evit_0.6-keepRate_$keep_rate-DropLoc_Default-lr_init_$lr-Dropout_GridSearch-$drop-FuseTokensOFF-ModelEmaON-LossScalerON-Time_$now"
    echo "----------------- Output dir: $logdir --------------------"
    
    python3 EViT/main.py \
    --model deit_base_patch16_shrink_base \
    --project_name "Thesis" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:1" \
    --finetune $ckpt \
    --batch-size 256 \
    --epochs 100 \
    --input-size 224 \
    --base_keep_rate $keep_rate \
    --drop_loc "$drop_loc" \
    --drop $drop \
    --opt "$opt" \
    --lr_scheduler \
    --lr $lr \
    --sched "$sched" \
    --lr_cycle_decay 0.8 \
    --min_lr 2e-6 \
    --weight-decay 1e-6 \
    --shrink_start_epoch 0 \
    --warmup_epochs 0 \
    --shrink_epochs 0 \
    --patience 100 \
    --counter_saver_threshold 70 \
    --delta 0.0 \
    --batch_aug \
    --color-jitter 0.0 \
    --loss_scaler\
    --data-path "$datapath" \
    --output_dir "EViT/keep_rate_0.6/03-06/$logdir"

done

echo "output dir for the last exp: EViT/$logdir"\

###################### evit model starting from deit base pretrained ############################

ckpt='EViT/Pretrained_Models/deit_base_patch16_224-b5f2ef4d.pth'
drop=0.2
keep_rate=1.0
lr=2e-4
drop_loc=None
now=$(date +"%Y%m%d")
sched='cosine'
opt='adamw'

logdir="EViT_Base-ckpt_deit_base_patch16_224-keepRate_$keep_rate-DropLoc_Default-lr_init_$lr-Dropout_$drop-FuseTokensOFF-ModelEmaON-LossScalerON-Epochs_30-Time_$now"
echo "----------------- Output dir: $logdir --------------------"

python3 EViT/main.py \
--model deit_base_patch16_shrink_base \
--project_name "Thesis" \
--run_name "$logdir" \
--hardware "Server" \
--gpu "cuda:1" \
--finetune $ckpt \
--batch-size 128 \
--epochs 30 \
--input-size 224 \
--base_keep_rate $keep_rate \
--drop_loc "$drop_loc" \
--drop $drop \
--opt "$opt" \
--lr_scheduler \
--lr $lr \
--sched "$sched" \
--lr_cycle_decay 0.8 \
--min_lr 2e-6 \
--weight-decay 1e-6 \
--shrink_start_epoch 0 \
--warmup_epochs 0 \
--shrink_epochs 0 \
--patience 120 \
--counter_saver_threshold 100 \
--delta 0.0 \
--batch_aug \
--color-jitter 0.0 \
--loss_scaler \
--data-path "$datapath" \
--output_dir "EViT/keep_rate_1.0/$logdir"


echo "output dir for the last exp: EViT/$logdir"\

###################### evit model starting from deit base pretrained ############################

ckpt='EViT/Pretrained_Models/deit_base_patch16_224-b5f2ef4d.pth'
drop=0.2
keep_rate=0.6
lr=2e-4
drop_loc=None
now=$(date +"%Y%m%d")
sched='cosine'
opt='adamw'

logdir="EViT_Base-ckpt_deit_base_patch16_224-keepRate_$keep_rate-DropLoc_Default-lr_init_$lr-Dropout_$drop-FuseTokensOFF-ModelEmaON-LossScalerON-Epochs_30-Time_$now"
echo "----------------- Output dir: $logdir --------------------"

python3 EViT/main.py \
--model deit_base_patch16_shrink_base \
--project_name "Thesis" \
--run_name "$logdir" \
--hardware "Server" \
--gpu "cuda:1" \
--finetune $ckpt \
--batch-size 256 \
--epochs 30 \
--input-size 224 \
--base_keep_rate $keep_rate \
--drop_loc "$drop_loc" \
--drop $drop \
--opt "$opt" \
--lr_scheduler \
--lr $lr \
--sched "$sched" \
--lr_cycle_decay 0.8 \
--min_lr 2e-6 \
--weight-decay 1e-6 \
--shrink_start_epoch 0 \
--warmup_epochs 0 \
--shrink_epochs 0 \
--patience 120 \
--counter_saver_threshold 100 \
--delta 0.0 \
--batch_aug \
--color-jitter 0.0 \
--loss_scaler \
--data-path "$datapath" \
--output_dir "EViT/keep_rate_0.6/03-06/$logdir"


echo "output dir for the last exp: EViT/$logdir"\

###################### evit model starting from deit base pretrained ############################

ckpt='EViT/Pretrained_Models/deit_base_patch16_224-b5f2ef4d.pth'
drop=0.2
lr=2e-4
keep_rate=(0.6 0.7 0.8 0.9)
drop_loc="(3, 6, 9)"
now=$(date +"%Y%m%d")
sched='cosine'
opt='adamw'



for kr in "${keep_rate[@]}"
do

    logdir="EViT_Base-ckpt_deit_base_patch16_224-keepRate_GridSearch_$kr-DropLoc_Default-lr_init_$lr-Dropout_$drop-FuseTokensOFF-ModelEmaON-LossScalerON-Time_$now"
    echo "----------------- Output dir: $logdir --------------------"

    python3 EViT/main.py \
    --model deit_base_patch16_shrink_base \
    --project_name "Thesis" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:1" \
    --finetune $ckpt \
    --batch-size 128 \
    --epochs 70 \
    --input-size 224 \
    --base_keep_rate $kr \
    --drop_loc "$drop_loc" \
    --drop $drop \
    --opt "$opt" \
    --lr_scheduler \
    --lr $lr \
    --sched "$sched" \
    --lr_cycle_decay 0.8 \
    --min_lr 2e-6 \
    --weight-decay 1e-6 \
    --shrink_start_epoch 0 \
    --warmup_epochs 0 \
    --shrink_epochs 0 \
    --patience 120 \
    --counter_saver_threshold 100 \
    --delta 0.0 \
    --batch_aug \
    --color-jitter 0.0 \
    --loss_scaler \
    --data-path "$datapath" \
    --output_dir "EViT/keep_rate_GridSearch/$logdir"

done 


echo "output dir for the last exp: EViT/$logdir"\