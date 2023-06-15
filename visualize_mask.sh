now=$(date +"%Y%m%d_%H%M%S")
logdir="visualization"
img_path="Images/ISIC2019/Images"
mask_path="Images/ISIC2019/Masks"

inst="instance"
pool="mask_max"
backbone="resnet18"

ckpt="Pretrained_Models/$backbone/mask_val_on/MIL_$inst-Pool_$pool/best_checkpoint.pth"

python main.py \
	--visualize \
	--wandb \
	--mask \
	--mil_type $inst \
	--pooling_type $pool \
	--num_workers 8 \
	--images_path $img_path \
	--vis_mask_path $mask_path \
	--resume $ckpt \
	--output_dir "$logdir/$backbone"  \

echo "output dir for the last exp: $logdir/MIL-$inst-$pool"\

