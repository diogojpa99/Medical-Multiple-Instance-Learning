now=$(date +"%Y%m%d_%H%M%S")
logdir=""
datapath="../Datasets/PH2_test"

mil_types=('instance' 'embedding')
pooling_types=( 'mask_max' 'mask_avg' ) 
data="PH2"
backbone="resnet18"

for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

		ckpt="Pretrained_Models/$backbone/mask_val_off/MIL_$mil_t-Pool_$pool/best_checkpoint.pth"

		logdir="MIL-$backbone-$mil_t-mask_val_OFF-$pool-Test_$data-Time_$now"
		echo "----------------- Output dir: $logdir --------------------"

		python main.py \
			--evaluate \
			--mask \
			--dataset $data \
			--project_name "Thesis" \
			--run_name "$logdir" \
			--hardware "MyPC" \
			--feature_extractor "resnet18.tv_in1k" \
			--mil_type $mil_t \
			--pooling_type $pool \
			--num_workers 8 \
			--data_path $datapath \
			--resume $ckpt \
			--output_dir "Results/$logdir"  \

		echo "output dir for the last exp: $logdir/MIL-$mil_t-$pool"

	done
done

###########################################################################

now=$(date +"%Y%m%d_%H%M%S")
logdir=""
datapath="../Datasets/derm7pt_like_ISIC2019"

mil_types=('instance' 'embedding')
pooling_types=( 'mask_max' 'mask_avg' )
data="Derm7pt"
backbone="resnet18"

for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

		ckpt="Pretrained_Models/$backbone/mask_val_off/MIL_$mil_t-Pool_$pool/best_checkpoint.pth"

		logdir="MIL-$backbone-$mil_t-mask_val_OFF-$pool-Test_$data-Time_$now"
		echo "----------------- Output dir: $logdir --------------------"

		python main.py \
			--evaluate \
			--dataset $data \
			--mask \
			--project_name "Thesis" \
			--run_name "$logdir" \
			--hardware "MyPC" \
			--feature_extractor "resnet18.tv_in1k" \
			--mil_type $mil_t \
			--pooling_type $pool \
			--num_workers 8 \
			--data_path $datapath \
			--resume $ckpt \
			--output_dir "Results/$logdir"  \

		echo "output dir for the last exp: $logdir/MIL-$mil_t-$pool"
		
	done
done

for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

		ckpt="Pretrained_Models/$backbone/mask_val_on/MIL_$mil_t-Pool_$pool/best_checkpoint.pth"

		logdir="MIL-$backbone-$mil_t-mask_val_ON-$pool-Test_$data-Time_$now"
		echo "----------------- Output dir: $logdir --------------------"

		python main.py \
			--evaluate \
			--dataset $data \
			--mask \
			--mask_val 'val' \
			--mask_path '../Datasets/DERM7PT_FINE_MASKS_224' \
			--project_name "Thesis" \
			--run_name "$logdir" \
			--hardware "MyPC" \
			--feature_extractor "resnet18.tv_in1k" \
			--mil_type $mil_t \
			--pooling_type $pool \
			--num_workers 8 \
			--data_path $datapath \
			--resume $ckpt \
			--output_dir "Results/$logdir"  \

		echo "output dir for the last exp: $logdir/MIL-$mil_t-$pool"
		
	done
done

