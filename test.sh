now=$(date +"%Y%m%d_%H%M%S")
logdir=""
datapath="../Datasets/derm7pt_like_ISIC2019"

inst="instance"
pool="max"

ckpt="Pretrained_Models/MIL_$inst-Pool_$pool/best_checkpoint.pth"

logdir="MIL-$inst-$pool-Test_PH2_test-Time_$now"
echo "----------------- Output dir: $logdir --------------------"

python main.py \
	--evaluate \
	--project_name "Thesis" \
	--run_name "$logdir" \
	--hardware "MyPC" \
	--mil_type $inst \
	--pooling_type $pool \
	--num_workers 8 \
	--data_path $datapath \
	--resume $ckpt \
	--output_dir $logdir  \

echo "output dir for the last exp: $logdir/MIL-$inst-$pool"\

