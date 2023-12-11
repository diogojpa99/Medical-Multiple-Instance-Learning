mil_types=("instance")
pool_types=("topk")

feature_ex="resnet18.tv_in1k"

mask_path="../Datasets/Fine_MASKS_bea_all"
data_path="../Datasets/ISIC2019bea_mel_nevus_limpo"
dataset="ISIC2019-Clean"

# mask_path="../Datasets/PH2_TEST_FINE_MASKS"
# data_path="../Datasets/PH2_test"
# dataset="PH2"

for mil_t in "${mil_types[@]}"
do
    for pool in "${pool_types[@]}"
    do

		ckpt="../Models/MIL-RN18-instance-topk-13-best_checkpoint.pth"
		logdir="ROI_EVAL"
		echo "----------------- Output dir: $logdir --------------------"

		python main.py \
			--roi_eval \
			--roi_patch_prob_threshold 0.5 \
			--roi_gradcam_threshold 0.0 \
			--wandb \
			--num_workers 8 \
			--batch_size 16 \
			--mil_type $mil_t \
			--pooling_type $pool \
			--topk 13 \
			--feature_extractor $feature_ex \
			--mask \
			--mask_val "val" \
			--mask_path $mask_path \
			--resume $ckpt \
			--dataset $dataset \
			--data_path $data_path \
			--output_dir "$logdir"  \
			
	done
done