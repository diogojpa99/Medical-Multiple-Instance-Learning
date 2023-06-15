now=$(date +"%Y%m%d_%H%M%S")
logdir=""
datapath="../Datasets/derm7pt_like_ISIC2019"

mil_types=('instance' 'embedding')
pooling_types=( 'max' 'avg' 'topk' )
data="Derm7pt"
backbone="resnet50"

for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

		ckpt="Pretrained_Models/$backbone/MIL_$mil_t-Pool_$pool/best_checkpoint.pth"

		logdir="MIL-$backbone-$mil_t-$pool-Test_$data-Time_$now"
		echo "----------------- Output dir: $logdir --------------------"

		python main.py \
			--evaluate \
			--dataset $data \
			--project_name "Thesis" \
			--run_name "$logdir" \
			--hardware "MyPC" \
			--feature_extractor "resnet50.tv_in1k" \
			--mil_type $mil_t \
			--pooling_type $pool \
			--num_workers 8 \
			--data_path $datapath \
			--resume $ckpt \
			--output_dir "Results/$logdir"  \

		echo "output dir for the last exp: $logdir/MIL-$mil_t-$pool"
		
	done
done