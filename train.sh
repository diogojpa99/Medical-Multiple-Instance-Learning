now=$(date +"%Y%m%d_%H%M%S")
logdir=/content/gdrive/MyDrive/IST/Thesis/MIL_exp/MIL_$now
datapath="/content/gdrive/.shortcut-targets-by-id/1FUYQ7eqJJam0F8pPkHaPpsm2LPDRriGk/ISIC2019bea_mel_nevus_limpo"
ckpt=""


# Params
lr=0.01
dc_rate=6
sched='poly'
drop=0.2

python3 main.py \
	--project_name "Thesis" \
	--run_name "MIL-Sched_$sched-$dc_rate-lr_init$lr-Dropout$drop-Wamrup_ON-Time$now" \
	--hardware "Server" \
	--gpu "cuda:1"\
	--batch_size 512 \
	--epochs 100 \
	--num_workers 2 \
    --opt "sgd" \
	--dropout $drop \
    --lr_scheduler \
	--sched $sched \
	--decay_rate $dc_rate\
	--lr $lr \
	--warmup_epochs 5 \
    --warmup_lr 0.001 \
	--min_lr 6e-4 \
	--patience 30 \
	--delta 0.0 \
	--no-model-ema \
	--data_path $datapath \
	--output_dir "MIL-Sched_$sched-lr_init$lr-Dropout$drop-Warmup_ON-Time$now"

echo "output dir for the last exp: MIL-Sched_$sched-lr_init$lr-Dropout$drop-Time$now"\

