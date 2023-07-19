now=$(date +"%Y%m%d_%H%M%S")

mil_types=("instance" "embedding")
pool_types=("mask_max" "mask_avg")

feature_ex="resnet50"

#mask_path="../Datasets/PH2_TEST_FINE_MASKS"
data_path="../Datasets/PH2_test"
dataset="PH2"

for mil_t in "${mil_types[@]}"
do
    for pool in "${pool_types[@]}"
    do

		ckpt="../aDOCS/Experiments_Saved/Binary/MIL/FeatureExtractors/$feature_ex/val_mask_OFF/MIL-$feature_ex-$mil_t-$pool/MIL-$mil_t-$pool-best_checkpoint.pth"
		logdir="MIL-Test-$feature_ex-$mil_t-$pool-$dataset-Time_$now"
		echo "----------------- Output dir: $logdir --------------------"

		python main.py \
			--evaluate \
			--project_name "Thesis" \
			--run_name "$logdir" \
			--hardware "MyPC" \
			--num_workers 8 \
			--mil_type $mil_t \
			--pooling_type $pool \
			--feature_extractor "resnet50.tv_in1k" \
			--mask \
			--resume $ckpt \
			--dataset $dataset \
			--data_path $data_path \
			--output_dir "$logdir"  \

	done
done

mask_path="../Datasets/Fine_MASKS_bea_all"
data_path="../Datasets/ISIC2019bea_mel_nevus_limpo"
dataset="ISIC2019-Clean"

for mil_t in "${mil_types[@]}"
do
    for pool in "${pool_types[@]}"
    do

		ckpt="../aDOCS/Experiments_Saved/Binary/MIL/FeatureExtractors/$feature_ex/val_mask_OFF/MIL-$feature_ex-$mil_t-$pool/MIL-$mil_t-$pool-best_checkpoint.pth"
		logdir="MIL-Test-$feature_ex-$mil_t-$pool-$dataset-Time_$now"
		echo "----------------- Output dir: $logdir --------------------"

		python main.py \
			--evaluate \
			--project_name "Thesis" \
			--run_name "$logdir" \
			--hardware "MyPC" \
			--num_workers 8 \
			--mil_type $mil_t \
			--pooling_type $pool \
			--feature_extractor "resnet50.tv_in1k" \
			--mask \
			--mask_val "" \
			--mask_path $mask_path \
			--resume $ckpt \
			--dataset $dataset \
			--data_path $data_path \
			--output_dir "$logdir"  \

	done
done

now=$(date +"%Y%m%d_%H%M%S")

mil_types=("instance" "embedding")
pool_types=("max" "avg" "topk" "mask_max" "mask_avg")

feature_ex="deitSmall"

#mask_path="../Datasets/Fine_MASKS_bea_all"
#data_path="../Datasets/ISIC2019bea_mel_nevus_limpo"
#dataset="ISIC2019-Clean"

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

		ckpt="../aDOCS/Experiments_Saved/Binary/MIL/FeatureExtractors/$feature_ex/MIL-$feature_ex-$mil_t-$pool/MIL-$mil_t-$pool-best_checkpoint.pth"
		logdir="MIL-Test-$feature_ex-$mil_t-$pool-$dataset-Time_$now"
		echo "----------------- Output dir: $logdir --------------------"

		python main.py \
			--evaluate \
			--project_name "Thesis" \
			--run_name "$logdir" \
			--hardware "MyPC" \
			--num_workers 8 \
			--mil_type $mil_t \
			--pooling_type $pool \
			--feature_extractor "deit_small_patch16_224" \
			--mask \
			--mask_val "val" \
			--mask_path $mask_path \
			--resume $ckpt \
			--dataset $dataset \
			--data_path $data_path \
			--output_dir "$logdir"  \

	done
done


now=$(date +"%Y%m%d_%H%M%S")

mil_types=("instance" "embedding")
pool_types=("max" "avg" "topk" "mask_max" "mask_avg")

feature_ex="deitSmall"

mask_path="../Datasets/DERM7PT_FINE_MASKS_224"
data_path="../Datasets/derm7pt_like_ISIC2019"
dataset="Derm7pt"

for mil_t in "${mil_types[@]}"
do
    for pool in "${pool_types[@]}"
    do

		ckpt="../aDOCS/Experiments_Saved/Binary/MIL/FeatureExtractors/$feature_ex/MIL-$feature_ex-$mil_t-$pool/MIL-$mil_t-$pool-best_checkpoint.pth"
		logdir="MIL-Test-$feature_ex-$mil_t-$pool-$dataset-Time_$now"
		echo "----------------- Output dir: $logdir --------------------"

		python main.py \
			--evaluate \
			--project_name "Thesis" \
			--run_name "$logdir" \
			--hardware "MyPC" \
			--num_workers 8 \
			--mil_type $mil_t \
			--pooling_type $pool \
			--feature_extractor "deit_small_patch16_224" \
			--mask \
			--mask_val "val" \
			--mask_path $mask_path \
			--resume $ckpt \
			--dataset $dataset \
			--data_path $data_path \
			--output_dir "$logdir"  \

	done
done

now=$(date +"%Y%m%d_%H%M%S")

mil_types=("instance" "embedding")
pool_types=("max" "avg" "topk" "mask_max" "mask_avg")

feature_ex="deitSmall"

mask_path="../Datasets/Fine_MASKS_bea_all"
data_path="../Datasets/ISIC2019bea_mel_nevus_limpo"
dataset="ISIC2019-Clean"


for mil_t in "${mil_types[@]}"
do
    for pool in "${pool_types[@]}"
    do

		ckpt="../aDOCS/Experiments_Saved/Binary/MIL/FeatureExtractors/$feature_ex/MIL-$feature_ex-$mil_t-$pool/MIL-$mil_t-$pool-best_checkpoint.pth"
		logdir="TEST-MIL-$feature_ex-$mil_t-$pool-$dataset-Time_$now"
		echo "----------------- Output dir: $logdir --------------------"

		python main.py \
			--evaluate \
			--project_name "Thesis" \
			--run_name "$logdir" \
			--hardware "MyPC" \
			--num_workers 8 \
			--mil_type $mil_t \
			--pooling_type $pool \
			--feature_extractor "deit_small_patch16_224" \
			--mask \
			--mask_val "val" \
			--mask_path $mask_path \
			--resume $ckpt \
			--dataset $dataset \
			--data_path $data_path \
			--output_dir "$logdir"  \

	done
done