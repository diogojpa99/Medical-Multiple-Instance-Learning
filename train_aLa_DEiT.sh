datapath=../Data/ISIC2019bea_mel_nevus_limpo
ckpt=Pretrained_Models/evit-0.9-fuse-img224-deit-s.pth

# Params
lr=5e-4
sched='cosine'
drop=0.0


now=$(date +"%Y%m%d_%H%M%S")
logdir="MIL-Sched_$sched-lr_init$lr-Dropout$drop-Wamrup_ON--Opt_Adamw-ModelEma_ON-colorJitter0.0-lossScalerONTime$now"
echo "Output dir: $logdir"


python3 main.py \
--project_name "Thesis" \
--run_name "$logdir" \
--hardware "Server" \
--gpu "cuda:1" \
--finetune $ckpt \
--batch_size 512 \
--epochs 300 \
--num_workers 2 \
--drop $drop \
--lr_scheduler \
--lr $lr \
--input_size 224 \
--sched "$sched" \
--min_lr 1e-5 \
--warmup_epochs 5 \
--warmup_lr 1e-6 \
--patience 25 \
--delta 0.0 \
--loss_scaler \
--batch_aug \
--color-jitter 0.0 \
--data_path "$datapath" \
--output_dir "$logdir"

echo "output dir for the last exp: $logdir"\
