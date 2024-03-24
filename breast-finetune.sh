###### General Settings ######
feature_extractor=('efficientnet_b3')

mil_types=('instance' 'embedding')
pooling_types=('avg' 'topk' 'max') 

batch=128
n_classes=2
epoch=130
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=20
delta=0.0
sched='cosine'
optimizers=('adamw')
dropout=0.3
drop_path=0.2
weight_decay=1e-4
loader="Gray_PIL_Loader_Wo_He_No_Resize"

############################################################## DDSM - Mass vs. Normal ####################################################################

datapath="../../data/zDiogo_Araujo/DDSM/DDSM_CLAHE-mass_normal"
dataset_name="DDSM_CLAHE-mass_normal"
classification_problem="mass_normal"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            logdir="MIL-Finetune-$dataset_name-$feat-$mil_t-$pool-Date_$now"
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
            --gpu "cuda:1" \
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
            --drop $dropout \
            --drop_layers_rate $drop_path \
            --weight-decay $weight_decay \
            --class_weights "balanced" \
            --test_val_flag \
            --loss_scaler \
            --breast_loader $loader \
            --breast_padding \
            --breast_strong_aug \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $dataset_name \
            --data_path $datapath \
            --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

############################################################## DDSM+CBIS - Mass vs. Normal ####################################################################

datapath="../../data/zDiogo_Araujo/DDSM+CBIS/DDSM+CBIS_CLAHE-mass_normal"
dataset_name="DDSM+CBIS_CLAHE-mass_normal"
classification_problem="mass_normal"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            logdir="MIL-Finetune-$dataset_name-$feat-$mil_t-$pool-Date_$now"
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
            --gpu "cuda:1" \
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
            --drop $dropout \
            --drop_layers_rate $drop_path \
            --weight-decay $weight_decay \
            --class_weights "balanced" \
            --test_val_flag \
            --loss_scaler \
            --breast_loader $loader \
            --breast_padding \
            --breast_strong_aug \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $dataset_name \
            --data_path $datapath \
            --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

############################################################## DDSM - Benign vs. Malignant ####################################################################

datapath="../../data/zDiogo_Araujo/DDSM/DDSM_CLAHE-benign_malignant"
dataset_name="DDSM_CLAHE-benign_malignant"
classification_problem="benign_malignant"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            logdir="MIL-Finetune-$dataset_name-$feat-$mil_t-$pool-Date_$now"
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
            --gpu "cuda:1" \
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
            --drop $dropout \
            --drop_layers_rate $drop_path \
            --weight-decay $weight_decay \
            --class_weights "balanced" \
            --test_val_flag \
            --loss_scaler \
            --breast_loader $loader \
            --breast_padding \
            --breast_strong_aug \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $dataset_name \
            --data_path $datapath \
            --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

############################################################## DDSM+CBIS - Benign vs. Malignant ####################################################################
datapath="../../data/zDiogo_Araujo/DDSM+CBIS/DDSM+CBIS_CLAHE-benign_malignant"
dataset_name="DDSM+CBIS_CLAHE-benign_malignant"
classification_problem="benign_malignant"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            logdir="MIL-Finetune-$dataset_name-$feat-$mil_t-$pool-Date_$now"
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
            --gpu "cuda:1" \
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
            --drop $dropout \
            --drop_layers_rate $drop_path \
            --weight-decay $weight_decay \
            --class_weights "balanced" \
            --test_val_flag \
            --loss_scaler \
            --breast_loader $loader \
            --breast_padding \
            --breast_strong_aug \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $dataset_name \
            --data_path $datapath \
            --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done


############################################################## DDSM+CBIS+MIAS_CLAHE - Benign vs. Malignant ####################################################################
datapath="../../data/zDiogo_Araujo/DDSM+CBIS+MIAS_CLAHE-benign_malignant"
dataset_name="DDSM+CBIS+MIAS_CLAHE-benign_malignant"
classification_problem="benign_malignant"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            logdir="MIL-Finetune-$dataset_name-$feat-$mil_t-$pool-Date_$now"
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
            --gpu "cuda:1" \
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
            --drop $dropout \
            --drop_layers_rate $drop_path \
            --weight-decay $weight_decay \
            --class_weights "balanced" \
            --test_val_flag \
            --loss_scaler \
            --breast_loader $loader \
            --breast_padding \
            --breast_strong_aug \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $dataset_name \
            --data_path $datapath \
            --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done


exit 0