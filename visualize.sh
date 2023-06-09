now=$(date +"%Y%m%d_%H%M%S")
logdir="visualization"
datapath="Images/"

inst="instance"
pool="mask_max"

ckpt="Pretrained_Models/MIL_$inst-Pool_$pool/best_checkpoint.pth"

python main.py \
	--visualize \
	--wandb \
	--mil_type $inst \
	--pooling_type $pool \
	--num_workers 8 \
	--images_path $datapath \
	--resume $ckpt \
	--output_dir $logdir  \

echo "output dir for the last exp: $logdir/MIL-$inst-$pool"\

