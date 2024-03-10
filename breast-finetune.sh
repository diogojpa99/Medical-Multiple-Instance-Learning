################################ DDSM-Benign_vs_Malignant #########################################

datapath="../Data/DDSM-Benign_vs_Malignant"
dataset_name="DDSM-Benign_vs_Malignant"
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

mil_types=('instance' 'embedding')
pooling_types=('avg' 'topk' 'max')

batch=128
n_classes=2
epoch=140
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=45
delta=0.0
sched='cosine'
optimizers=('adamw')
drops=(0.3)
drops_layers_rate=(0.2)
drop_block_rate=None
weight_decay=1e-6

feature_extractor=('densenet169.tv_in1k' 'efficientnet_b3' 'resnet50.tv_in1k' 'deit_small_patch16_224' 'deit_small_patch16_shrink_base')

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            for drop_path in "${drops_layers_rate[@]}"
            do
                for dropout in "${drops[@]}"
                do
                    for opt in "${optimizers[@]}"
                    do
                        now=$(date +"%Y%m%d_%H%M%S")
                        logdir="MIL-Finetune-$dataset_type-$classification_problem-$dataset_name-$feat-$mil_t-lr_$lr-$pool-drop_$dropout-drop_layer_$drop_path-opt_$opt-w_decay_$weight_decay-Date_$now"
                        log_file="Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir/run_log.txt"
                        mkdir -p "$(dirname "$log_file")" 
                        echo "----------------- Starting Program: $logdir --------------------"
            
                        python main.py \
                        --finetune \
                        --feature_extractor $feat \
                        --mil_type $mil_t \
                        --pooling_type $pool \
                        --nb_classes $n_classes \
                        --project_name "MIA-Breast" \
                        --run_name "$logdir" \
                        --hardware "Server" \
                        --gpu "cuda:0" \
                        --num_workers 8 \
                        --epochs $epoch \
                        --classifier_warmup_epochs 5 \
                        --batch_size $batch \
                        --input_size 224 \
                        --sched $sched \
                        --lr $lr \
                        --min_lr $min_lr \
                        --warmup_lr $warmup_lr \
                        --warmup_epochs 10 \
                        --patience $patience \
                        --delta $delta \
                        --counter_saver_threshold $epoch \
                        --drop $dropout\
                        --drop_layers_rate $drop_path \
                        --weight-decay $weight_decay \
                        --class_weights "balanced" \
                        --test_val_flag \
                        --loss_scaler \
                        --dataset_type $dataset_type \
                        --dataset $dataset_name \
                        --data_path $datapath \
                        --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir" #>> "$log_file" 2>&1
                        
                        echo "Output dir for the last experiment: $logdir"
                    done 
                done
            done
        done
    done
done

################################ DDSM-Mass_vs_Normal #########################################

datapath="../Data/DDSM-Mass_vs_Normal"
dataset_name="DDSM-Mass_vs_Normal"
classification_problem="Mass_vs_Normal"
dataset_type="Breast"

mil_types=('instance' 'embedding')
pooling_types=('avg' 'topk' 'max')

batch=128
n_classes=2
epoch=140
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=45
delta=0.0
sched='cosine'
optimizers=('adamw')
drops=(0.3)
drops_layers_rate=(0.0)
drop_block_rate=None
weight_decay=1e-6

feature_extractor=('densenet169.tv_in1k' 'efficientnet_b3' 'resnet50.tv_in1k' 'deit_small_patch16_224' 'deit_small_patch16_shrink_base')

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            for drop_path in "${drops_layers_rate[@]}"
            do
                for dropout in "${drops[@]}"
                do
                    for opt in "${optimizers[@]}"
                    do
                        now=$(date +"%Y%m%d_%H%M%S")
                        logdir="MIL-Finetune-$dataset_type-$classification_problem-$dataset_name-$feat-$mil_t-lr_$lr-$pool-drop_$dropout-drop_layer_$drop_path-opt_$opt-w_decay_$weight_decay-Date_$now"
                        log_file="Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir/run_log.txt"
                        mkdir -p "$(dirname "$log_file")" 
                        echo "----------------- Starting Program: $logdir --------------------"
            
                        python main.py \
                        --finetune \
                        --feature_extractor $feat \
                        --mil_type $mil_t \
                        --pooling_type $pool \
                        --nb_classes $n_classes \
                        --project_name "MIA-Breast" \
                        --run_name "$logdir" \
                        --hardware "Server" \
                        --gpu "cuda:0" \
                        --num_workers 8 \
                        --epochs $epoch \
                        --classifier_warmup_epochs 5 \
                        --batch_size $batch \
                        --input_size 224 \
                        --sched $sched \
                        --lr $lr \
                        --min_lr $min_lr \
                        --warmup_lr $warmup_lr \
                        --warmup_epochs 10 \
                        --patience $patience \
                        --delta $delta \
                        --counter_saver_threshold $epoch \
                        --drop $dropout\
                        --drop_layers_rate $drop_path \
                        --weight-decay $weight_decay \
                        --class_weights "balanced" \
                        --test_val_flag \
                        --loss_scaler \
                        --dataset_type $dataset_type \
                        --dataset $dataset_name \
                        --data_path $datapath \
                        --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir" #>> "$log_file" 2>&1
                        
                        echo "Output dir for the last experiment: $logdir"
                    done 
                done
            done
        done
    done
done

################################ DDSM-Benign_vs_Malignant #########################################

datapath="../Data/DDSM-Benign_vs_Malignant"
dataset_name="DDSM-Benign_vs_Malignant"
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

mil_types=('instance' 'embedding')
pooling_types=('avg' 'topk' 'max')

batch=128
n_classes=2
epoch=140
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=45
delta=0.0
sched='cosine'
optimizers=('adamw')
drops=(0.3)
drops_layers_rate=(0.0)
drop_block_rate=None
weight_decay=1e-6

feature_extractor=('resnet18.tv_in1k' 'vgg16.tv_in1k' 'deit_base_patch16_224')

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            for drop_path in "${drops_layers_rate[@]}"
            do
                for dropout in "${drops[@]}"
                do
                    for opt in "${optimizers[@]}"
                    do
                        now=$(date +"%Y%m%d_%H%M%S")
                        logdir="MIL-Finetune-$dataset_type-$classification_problem-$dataset_name-$feat-$mil_t-lr_$lr-$pool-drop_$dropout-drop_layer_$drop_path-opt_$opt-w_decay_$weight_decay-Date_$now"
                        log_file="Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir/run_log.txt"
                        mkdir -p "$(dirname "$log_file")" 
                        echo "----------------- Starting Program: $logdir --------------------"
            
                        python main.py \
                        --finetune \
                        --feature_extractor $feat \
                        --mil_type $mil_t \
                        --pooling_type $pool \
                        --nb_classes $n_classes \
                        --project_name "MIA-Breast" \
                        --run_name "$logdir" \
                        --hardware "Server" \
                        --gpu "cuda:0" \
                        --num_workers 8 \
                        --epochs $epoch \
                        --classifier_warmup_epochs 5 \
                        --batch_size $batch \
                        --input_size 224 \
                        --sched $sched \
                        --lr $lr \
                        --min_lr $min_lr \
                        --warmup_lr $warmup_lr \
                        --warmup_epochs 10 \
                        --patience $patience \
                        --delta $delta \
                        --counter_saver_threshold $epoch \
                        --drop $dropout\
                        --drop_layers_rate $drop_path \
                        --weight-decay $weight_decay \
                        --class_weights "balanced" \
                        --test_val_flag \
                        --loss_scaler \
                        --dataset_type $dataset_type \
                        --dataset $dataset_name \
                        --data_path $datapath \
                        --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir" #>> "$log_file" 2>&1
                        
                        echo "Output dir for the last experiment: $logdir"
                    done 
                done
            done
        done
    done
done

################################ DDSM-Mass_vs_Normal #########################################

datapath="../Data/DDSM-Mass_vs_Normal"
dataset_name="DDSM-Mass_vs_Normal"
classification_problem="Mass_vs_Normal"
dataset_type="Breast"

mil_types=('instance' 'embedding')
pooling_types=('avg' 'topk' 'max')

batch=128
n_classes=2
epoch=140
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=45
delta=0.0
sched='cosine'
optimizers=('adamw')
drops=(0.3)
drops_layers_rate=(0.0)
drop_block_rate=None
weight_decay=1e-6

feature_extractor=('resnet18.tv_in1k' 'vgg16.tv_in1k' 'deit_base_patch16_224')

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            for drop_path in "${drops_layers_rate[@]}"
            do
                for dropout in "${drops[@]}"
                do
                    for opt in "${optimizers[@]}"
                    do
                        now=$(date +"%Y%m%d_%H%M%S")
                        logdir="MIL-Finetune-$dataset_type-$classification_problem-$dataset_name-$feat-$mil_t-lr_$lr-$pool-drop_$dropout-drop_layer_$drop_path-opt_$opt-w_decay_$weight_decay-Date_$now"
                        log_file="Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir/run_log.txt"
                        mkdir -p "$(dirname "$log_file")" 
                        echo "----------------- Starting Program: $logdir --------------------"
            
                        python main.py \
                        --finetune \
                        --feature_extractor $feat \
                        --mil_type $mil_t \
                        --pooling_type $pool \
                        --nb_classes $n_classes \
                        --project_name "MIA-Breast" \
                        --run_name "$logdir" \
                        --hardware "Server" \
                        --gpu "cuda:0" \
                        --num_workers 8 \
                        --epochs $epoch \
                        --classifier_warmup_epochs 5 \
                        --batch_size $batch \
                        --input_size 224 \
                        --sched $sched \
                        --lr $lr \
                        --min_lr $min_lr \
                        --warmup_lr $warmup_lr \
                        --warmup_epochs 10 \
                        --patience $patience \
                        --delta $delta \
                        --counter_saver_threshold $epoch \
                        --drop $dropout\
                        --drop_layers_rate $drop_path \
                        --weight-decay $weight_decay \
                        --class_weights "balanced" \
                        --test_val_flag \
                        --loss_scaler \
                        --dataset_type $dataset_type \
                        --dataset $dataset_name \
                        --data_path $datapath \
                        --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir" #>> "$log_file" 2>&1
                        
                        echo "Output dir for the last experiment: $logdir"
                    done 
                done
            done
        done
    done
done