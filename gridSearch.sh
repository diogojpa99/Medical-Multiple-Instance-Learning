datapath="../Data/ISIC2019bea_mel_nevus_limpo"
ckpt=""

mil_types=('instance' 'embedding')
pooling_types=('max' 'avg' 'topk')
now=$(date +"%Y%m%d_%H%M%S")

# Params
lr=5e-4
sched='cosine'
drop=0.0


now=$(date +"%Y%m%d_%H%M%S")

for mil_t in "${mil_types[@]}"
do
    for pool in "${pooling_types[@]}"
    do

    logdir="MIL-GridSearch-$mil_t-$pool-Opt_AdamW-Sched_$sched-lr_init$lr-Dropout$drop-Opt_DataAug_ON-Time$now"
    echo "----------------- Output dir: $logdir --------------------"
        
        python3 main.py \
        --project_name "Thesis" \
        --run_name "$logdir" \
        --hardware "Server" \
        --gpu "cuda:0" \
        --finetune $ckpt \
        --batch_size 512 \
        --epochs 150 \
        --num_workers 2 \
        --mil_type $mil_t \
        --pooling_type $pool \
        --drop $drop \
        --lr_scheduler \
        --lr $lr \
        --input_size 224 \
        --sched "$sched" \
        --min_lr 1e-5 \
        --warmup_epochs 5 \
        --warmup_lr 1e-6 \
        --patience 25 \
        --delta 0.0 \
        --loss_scaler \
        --batch_aug \
        --color-jitter 0.0 \
        --data_path "$datapath" \
        --output_dir "$logdir"

        echo "Output dir for the last experiment: $logdir"
    done
done