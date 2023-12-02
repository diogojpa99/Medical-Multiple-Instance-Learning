mil_types=("instance")
pool_types=("topk")

feature_ex="efficientnet_b3"

mask_path="../Datasets/Fine_MASKS_bea_all"
data_path="../Datasets/ISIC2019bea_mel_nevus_limpo"
dataset="ISIC2019-Clean"

mask_path="../Datasets/PH2_TEST_FINE_MASKS"
data_path="../Datasets/PH2_test"
dataset="PH2"

#mask_path="../Datasets/DERM7PT_FINE_MASKS_224"
#data_path="../Datasets/derm7pt_like_ISIC2019"
#dataset="Derm7pt"

for mil_t in "${mil_types[@]}"
do
    for pool in "${pool_types[@]}"
    do

		ckpt="../models/MIL-EN-B3-instance-topk-25-best_checkpoint.pth"
		logdir="visualization/$mil_t-$pool/$dataset/$feature_ex/top-25"
		echo "----------------- Output dir: $logdir --------------------"

		python main.py \
			--visualize \
			--wandb \
			--num_workers 8 \
			--mil_type $mil_t \
			--pooling_type $pool \
			--feature_extractor "efficientnet_b3" \
			--mask \
			--topk 49 \
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