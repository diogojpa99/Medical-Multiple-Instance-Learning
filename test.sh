now=$(date +"%Y%m%d_%H%M%S")

mil_types=("instance" "embedding")
pool_types=("max" "avg" "topk")
feature_ex="resnet18"
mask_path="../Datasets/Fine_MASKS_bea_all"
data_path="../Datasets/ISIC2019bea_mel_nevus_limpo"
dataset="ISIC2019-Clean"

for mil_t in "${mil_types[@]}"
do
    for pool in "${pool_types[@]}"
    do

		if [ "$pool" == "mask_max" ] || [ "$pool" == "mask_avg" ]; then
			ckpt="Pretrained_Models/$feature_ex/mask_val_on/MIL_$mil_t-Pool_$pool/best_checkpoint.pth"
		else
			ckpt="Pretrained_Models/$feature_ex/MIL_$mil_t-Pool_$pool/best_checkpoint.pth"
		fi
		logdir="MIL-$feature_ex-$mil_t-$pool-Test_$data-Time_$now"
		echo "----------------- Output dir: $logdir --------------------"

		python main.py \
			--evaluate \
			--project_name "Thesis" \
			--run_name "$logdir" \
			--hardware "MyPC" \
			--num_workers 8 \
			--mil_type $mil_t \
			--pooling_type $pool \
			--feature_extractor "resnet18.tv_in1k" \
			--mask \
			--mask_val "val" \
			--mask_path $mask_path \
			--visualize_num_images 8 \
			--vis_num 5 \
			--resume $ckpt \
			--dataset $dataset \
			--data_path $data_path \
			--output_dir "$logdir"  \

	done
done