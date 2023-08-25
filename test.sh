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


################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
##### Test ######
now=$(date +"%Y%m%d_%H%M%S")

feature_ex="evit_small_07_one_fusedToken"

mil_types=("instance" "embedding")
pool_types=("max" "avg" "topk")
datasets=("ISIC2019-Clean" "PH2" "Derm7pt")

for dataset in "${datasets[@]}"
do
    if [[ $dataset == "ISIC2019-Clean" ]]; then
        mask_path="../Data/Fine_MASKS_bea_all"
        data_path="../Data/ISIC2019bea_mel_nevus_limpo"
    elif [[ $dataset == "PH2" ]]; then
        mask_path="../Data/Test-Datasets/PH2_TEST_FINE_MASKS"
        data_path="../Data/Test-Datasets/PH2_test"
    else
        mask_path="../Data/Test-Datasets/DERM7PT_FINE_MASKS_224"
        data_path="../Data/Test-Datasets/derm7pt_like_ISIC2019"
    fi
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pool_types[@]}"
        do

            ckpt="evit_small_07/$feature_ex/MIL-$feature_ex-$mil_t-$pool-lr_init_2e-4-Time_20230723/MIL-$mil_t-$pool-best_checkpoint.pth"
            logdir="MIL-Test-$feature_ex-$mil_t-$pool-$dataset-Time_$now"
            echo "----------------- Output dir: $logdir --------------------"

            python main.py \
                --evaluate \
                --project_name "Thesis" \
                --run_name "$logdir" \
                --hardware "Server" \
                --gpu "cpu" \
                --num_workers 8 \
                --mil_type $mil_t \
                --pooling_type $pool \
                --feature_extractor "deit_small_patch16_shrink_base" \
                --fuse_token \
                --mask \
                --mask_val "val" \
                --mask_path $mask_path \
                --resume $ckpt \
                --dataset $dataset \
                --data_path $data_path \
                --output_dir "TESTs/$logdir"  \

        done
    done
done







