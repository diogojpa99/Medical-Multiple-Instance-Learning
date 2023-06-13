now=$(date +"%Y%m%d")
logdir=""
datapath="../Datasets/derm7pt_like_ISIC2019"

mil_t="embedding"
pool="mask_avg"
data="Derm7pt"

ckpt="Pretrained_Models/MIL_$mil_t-Pool_$pool/best_checkpoint.pth"

logdir="MIL-$mil_t-$pool-Test_$data-Masks_ON-Time_$now"
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
	--mil_type $mil_t \
	--pooling_type $pool \
	--num_workers 8 \
	--data_path $datapath \
	--resume $ckpt \
	--output_dir $logdir  \

echo "output dir for the last exp: $logdir/MIL-$mil_t-$pool"\

