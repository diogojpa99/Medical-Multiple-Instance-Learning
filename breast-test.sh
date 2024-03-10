dataset_n="CMMD-only_mass-processed_crop_CLAHE"

################################ CMMD-only_mass_dataset #########################################

datapath="../Data/CMMD/$dataset_n"
dataset_name=$dataset_n
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

mil_types=('instance' 'embedding')
pooling_types=('avg' 'topk' 'max')

batch=128
n_classes=2

feature_extractor=('densenet169.tv_in1k' 'efficientnet_b3' 'resnet50.tv_in1k' 'deit_small_patch16_224')

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
                        ckpt_path="aFinetuned_Models/Binary/$classification_problem/$dataset_name/MIL-Finetune-$dataset_type-$classification_problem-$dataset_name-$feat-$mil_t-lr_$lr-$pool-drop_$dropout-drop_layer_$drop_path-opt_$opt-w_decay_$weight_decay-Date_$now"
                        logdir="MIL-Eval-$dataset_type-$classification_problem-$dataset_name-$feat-$mil_t-lr_$lr-$pool-drop_$dropout-drop_layer_$drop_path-opt_$opt-w_decay_$weight_decay-Date_$now"
                        log_file="aTests/Binary/$classification_problem/$dataset_name/$logdir/run_log.txt"
                        mkdir -p "$(dirname "$log_file")" 
                        echo "----------------- Starting Program: $logdir --------------------"
            
                        python main.py \
                        --eval \
                        --resume $ckpt_path \
                        --feature_extractor $feat \
                        --mil_type $mil_t \
                        --pooling_type $pool \
                        --nb_classes $n_classes \
                        --project_name "MIA-Breast" \
                        --run_name "$logdir" \
                        --hardware "Server" \
                        --gpu "cuda:1" \
                        --num_workers 8 \
                        --batch_size $batch \
                        --input_size 224 \
                        --class_weights "balanced" \
                        --dataset_type $dataset_type \
                        --dataset $dataset_name \
                        --data_path $datapath \
                        --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir" >> "$log_file" 2>&1
                        
                        echo "Output dir for the last experiment: $logdir"
                    done 
                done
            done
        done
    done
done


dataset_n="CMMD-only_mass-processed_crop_CLAHE"

################################ CMMD-only_mass_dataset #########################################

datapath="../Data/CMMD/$dataset_n"
dataset_name=$dataset_n
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

mil_types=('instance' 'embedding')
pooling_types=('avg' 'topk' 'max')

batch=128
n_classes=2

feature_extractor=('resnet18.tv_in1k' 'vgg16.tv_in1k' 'deit_base_patch16_224' 'deit_small_patch16_shrink_base')

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
                        ckpt_path="aFinetuned_Models/Binary/$classification_problem/$dataset_name/MIL-Finetune-$dataset_type-$classification_problem-$dataset_name-$feat-$mil_t-lr_$lr-$pool-drop_$dropout-drop_layer_$drop_path-opt_$opt-w_decay_$weight_decay-Date_$now"
                        logdir="MIL-Eval-$dataset_type-$classification_problem-$dataset_name-$feat-$mil_t-lr_$lr-$pool-drop_$dropout-drop_layer_$drop_path-opt_$opt-w_decay_$weight_decay-Date_$now"
                        log_file="aTests/Binary/$classification_problem/$dataset_name/$logdir/run_log.txt"
                        mkdir -p "$(dirname "$log_file")" 
                        echo "----------------- Starting Program: $logdir --------------------"
            
                        python main.py \
                        --eval \
                        --resume $ckpt_path \
                        --feature_extractor $feat \
                        --mil_type $mil_t \
                        --pooling_type $pool \
                        --nb_classes $n_classes \
                        --project_name "MIA-Breast" \
                        --run_name "$logdir" \
                        --hardware "Server" \
                        --gpu "cuda:1" \
                        --num_workers 8 \
                        --batch_size $batch \
                        --input_size 224 \
                        --class_weights "balanced" \
                        --dataset_type $dataset_type \
                        --dataset $dataset_name \
                        --data_path $datapath \
                        --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir" >> "$log_file" 2>&1
                        
                        echo "Output dir for the last experiment: $logdir"
                    done 
                done
            done
        done
    done
done

dataset_n="CMMD-only_mass"

################################ CMMD-only_mass_dataset #########################################

datapath="../Data/CMMD/$dataset_n"
dataset_name=$dataset_n
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

mil_types=('instance' 'embedding')
pooling_types=('avg' 'topk' 'max')

batch=128
n_classes=2

feature_extractor=('densenet169.tv_in1k' 'efficientnet_b3' 'resnet50.tv_in1k' 'deit_small_patch16_224')

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
                        ckpt_path="aFinetuned_Models/Binary/$classification_problem/$dataset_name/MIL-Finetune-$dataset_type-$classification_problem-$dataset_name-$feat-$mil_t-lr_$lr-$pool-drop_$dropout-drop_layer_$drop_path-opt_$opt-w_decay_$weight_decay-Date_$now"
                        logdir="MIL-Eval-$dataset_type-$classification_problem-$dataset_name-$feat-$mil_t-lr_$lr-$pool-drop_$dropout-drop_layer_$drop_path-opt_$opt-w_decay_$weight_decay-Date_$now"
                        log_file="aTests/Binary/$classification_problem/$dataset_name/$logdir/run_log.txt"
                        mkdir -p "$(dirname "$log_file")" 
                        echo "----------------- Starting Program: $logdir --------------------"
            
                        python main.py \
                        --eval \
                        --resume $ckpt_path \
                        --feature_extractor $feat \
                        --mil_type $mil_t \
                        --pooling_type $pool \
                        --nb_classes $n_classes \
                        --project_name "MIA-Breast" \
                        --run_name "$logdir" \
                        --hardware "Server" \
                        --gpu "cuda:1" \
                        --num_workers 8 \
                        --batch_size $batch \
                        --input_size 224 \
                        --class_weights "balanced" \
                        --dataset_type $dataset_type \
                        --dataset $dataset_name \
                        --data_path $datapath \
                        --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir" >> "$log_file" 2>&1
                        
                        echo "Output dir for the last experiment: $logdir"
                    done 
                done
            done
        done
    done
done


dataset_n="CMMD-only_mass"

################################ CMMD-only_mass_dataset #########################################

datapath="../Data/CMMD/$dataset_n"
dataset_name=$dataset_n
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

mil_types=('instance' 'embedding')
pooling_types=('avg' 'topk' 'max')

batch=128
n_classes=2

feature_extractor=('resnet18.tv_in1k' 'vgg16.tv_in1k' 'deit_base_patch16_224' 'deit_small_patch16_shrink_base')

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
                        ckpt_path="aFinetuned_Models/Binary/$classification_problem/$dataset_name/MIL-Finetune-$dataset_type-$classification_problem-$dataset_name-$feat-$mil_t-lr_$lr-$pool-drop_$dropout-drop_layer_$drop_path-opt_$opt-w_decay_$weight_decay-Date_$now"
                        logdir="MIL-Eval-$dataset_type-$classification_problem-$dataset_name-$feat-$mil_t-lr_$lr-$pool-drop_$dropout-drop_layer_$drop_path-opt_$opt-w_decay_$weight_decay-Date_$now"
                        log_file="aTests/Binary/$classification_problem/$dataset_name/$logdir/run_log.txt"
                        mkdir -p "$(dirname "$log_file")" 
                        echo "----------------- Starting Program: $logdir --------------------"
            
                        python main.py \
                        --eval \
                        --resume $ckpt_path \
                        --feature_extractor $feat \
                        --mil_type $mil_t \
                        --pooling_type $pool \
                        --nb_classes $n_classes \
                        --project_name "MIA-Breast" \
                        --run_name "$logdir" \
                        --hardware "Server" \
                        --gpu "cuda:1" \
                        --num_workers 8 \
                        --batch_size $batch \
                        --input_size 224 \
                        --class_weights "balanced" \
                        --dataset_type $dataset_type \
                        --dataset $dataset_name \
                        --data_path $datapath \
                        --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir" >> "$log_file" 2>&1
                        
                        echo "Output dir for the last experiment: $logdir"
                    done 
                done
            done
        done
    done
done