######## General Settings ########
feature_extractor=('efficientnet_b3')

mil_types=('instance' 'embedding')
pooling_types=('avg' 'topk' 'max') 

batch=128
n_classes=2
loader="Gray_PIL_Loader_Wo_He_No_Resize"


################################ DDSM_CLAHE-mass_normal #########################################

trainset="DDSM_CLAHE-mass_normal"
testset="DDSM_CLAHE-mass_normal"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/DDSM/$testset"
classification_problem="mass_normal"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/MIL-Finetune-$trainset-$feat-$mil_t-$pool-*")"
            if [[ $ckpt_path == *$suffix_to_remove ]]; then
                ckpt_path="${ckpt_path%$suffix_to_remove}"
            fi
            ckpt_path="${ckpt_path}/MIL-$mil_t-$pool-best_checkpoint.pth"

            logdir="MIL-Test-trainset_$trainset-testset_$testset-$feat-$mil_t-$pool-Date_$now"
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
            --class_weights "balanced" \
            --breast_loader $loader \
            --breast_padding \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $trainset \
            --testset $testset \
            --data_path $datapath \
            --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

################################ DDSM_CLAHE-mass_normal #########################################

trainset="DDSM_CLAHE-mass_normal"
testset="MIAS_CLAHE-mass_normal"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/MIAS/$testset"
classification_problem="mass_normal"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/MIL-Finetune-$trainset-$feat-$mil_t-$pool-*")"
            if [[ $ckpt_path == *$suffix_to_remove ]]; then
                ckpt_path="${ckpt_path%$suffix_to_remove}"
            fi
            ckpt_path="${ckpt_path}/MIL-$mil_t-$pool-best_checkpoint.pth"

            logdir="MIL-Test-trainset_$trainset-testset_$testset-$feat-$mil_t-$pool-Date_$now"
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
            --class_weights "balanced" \
            --breast_loader $loader \
            --breast_padding \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $trainset \
            --testset $testset \
            --data_path $datapath \
            --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

################################ DDSM+CBIS - Mass_Normal #########################################

trainset="DDSM_CLAHE-mass_normal"
testset="MIAS_CLAHE-mass_normal"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/MIAS/$testset"
classification_problem="mass_normal"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/MIL-Finetune-$trainset-$feat-$mil_t-$pool-*")"
            if [[ $ckpt_path == *$suffix_to_remove ]]; then
                ckpt_path="${ckpt_path%$suffix_to_remove}"
            fi
            ckpt_path="${ckpt_path}/MIL-$mil_t-$pool-best_checkpoint.pth"

            logdir="MIL-Test-trainset_$trainset-testset_$testset-$feat-$mil_t-$pool-Date_$now"
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
            --class_weights "balanced" \
            --breast_loader $loader \
            --breast_padding \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $trainset \
            --testset $testset \
            --data_path $datapath \
            --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

###########################################################################################################
trainset="DDSM+CBIS_CLAHE-mass_normal"
testset="DDSM+CBIS_CLAHE-mass_normal"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/DDSM+CBIS/$testset"
classification_problem="mass_normal"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/MIL-Finetune-$trainset-$feat-$mil_t-$pool-*")"
            if [[ $ckpt_path == *$suffix_to_remove ]]; then
                ckpt_path="${ckpt_path%$suffix_to_remove}"
            fi
            ckpt_path="${ckpt_path}/MIL-$mil_t-$pool-best_checkpoint.pth"

            logdir="MIL-Test-trainset_$trainset-testset_$testset-$feat-$mil_t-$pool-Date_$now"
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
            --class_weights "balanced" \
            --breast_loader $loader \
            --breast_padding \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $trainset \
            --testset $testset \
            --data_path $datapath \
            --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

############################################### Benign_Malignant - DDSM Train #######################################################
trainset="DDSM_CLAHE-benign_malignant"
testset="DDSM_CLAHE-benign_malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/DDSM/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/MIL-Finetune-$trainset-$feat-$mil_t-$pool-*")"
            if [[ $ckpt_path == *$suffix_to_remove ]]; then
                ckpt_path="${ckpt_path%$suffix_to_remove}"
            fi
            ckpt_path="${ckpt_path}/MIL-$mil_t-$pool-best_checkpoint.pth"

            logdir="MIL-Test-trainset_$trainset-testset_$testset-$feat-$mil_t-$pool-Date_$now"
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
            --class_weights "balanced" \
            --breast_loader $loader \
            --breast_padding \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $trainset \
            --testset $testset \
            --data_path $datapath \
            --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

############################################### Benign_Malignant - DDSM Train #######################################################
trainset="DDSM_CLAHE-benign_malignant"
testset="CMMD-only_mass"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/CMMD/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/MIL-Finetune-$trainset-$feat-$mil_t-$pool-*")"
            if [[ $ckpt_path == *$suffix_to_remove ]]; then
                ckpt_path="${ckpt_path%$suffix_to_remove}"
            fi
            ckpt_path="${ckpt_path}/MIL-$mil_t-$pool-best_checkpoint.pth"

            logdir="MIL-Test-trainset_$trainset-testset_$testset-$feat-$mil_t-$pool-Date_$now"
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
            --class_weights "balanced" \
            --breast_loader $loader \
            --breast_clahe \
            --clahe_clip_limit 5.0 \
            --breast_padding \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $trainset \
            --testset $testset \
            --data_path $datapath \
            --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done


############################################### Benign_Malignant - DDSM Train #######################################################
trainset="DDSM_CLAHE-benign_malignant"
testset="CBIS_CLAHE-benign_malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/CBIS/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/MIL-Finetune-$trainset-$feat-$mil_t-$pool-*")"
            if [[ $ckpt_path == *$suffix_to_remove ]]; then
                ckpt_path="${ckpt_path%$suffix_to_remove}"
            fi
            ckpt_path="${ckpt_path}/MIL-$mil_t-$pool-best_checkpoint.pth"

            logdir="MIL-Test-trainset_$trainset-testset_$testset-$feat-$mil_t-$pool-Date_$now"
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
            --class_weights "balanced" \
            --breast_loader $loader \
            --breast_padding \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $trainset \
            --testset $testset \
            --data_path $datapath \
            --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

############################################### Benign_Malignant - DDSM Train #######################################################
trainset="DDSM_CLAHE-benign_malignant"
testset="MIAS_CLAHE-benign_malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/MIAS/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/MIL-Finetune-$trainset-$feat-$mil_t-$pool-*")"
            if [[ $ckpt_path == *$suffix_to_remove ]]; then
                ckpt_path="${ckpt_path%$suffix_to_remove}"
            fi
            ckpt_path="${ckpt_path}/MIL-$mil_t-$pool-best_checkpoint.pth"

            logdir="MIL-Test-trainset_$trainset-testset_$testset-$feat-$mil_t-$pool-Date_$now"
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
            --class_weights "balanced" \
            --breast_loader $loader \
            --breast_padding \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $trainset \
            --testset $testset \
            --data_path $datapath \
            --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

############################################### Benign_Malignant - DDSM Train #######################################################
trainset="DDSM+CBIS_CLAHE-benign_malignant"
testset="DDSM+CBIS_CLAHE-benign_malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/DDSM+CBIS/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/MIL-Finetune-$trainset-$feat-$mil_t-$pool-*")"
            if [[ $ckpt_path == *$suffix_to_remove ]]; then
                ckpt_path="${ckpt_path%$suffix_to_remove}"
            fi
            ckpt_path="${ckpt_path}/MIL-$mil_t-$pool-best_checkpoint.pth"

            logdir="MIL-Test-trainset_$trainset-testset_$testset-$feat-$mil_t-$pool-Date_$now"
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
            --class_weights "balanced" \
            --breast_loader $loader \
            --breast_padding \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $trainset \
            --testset $testset \
            --data_path $datapath \
            --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done


###########################################################################################################
trainset="DDSM+CBIS_CLAHE-benign_malignant"
testset="CMMD-only_mass"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/CMMD/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"


for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/MIL-Finetune-$trainset-$feat-$mil_t-$pool-*")"
            if [[ $ckpt_path == *$suffix_to_remove ]]; then
                ckpt_path="${ckpt_path%$suffix_to_remove}"
            fi
            ckpt_path="${ckpt_path}/MIL-$mil_t-$pool-best_checkpoint.pth"

            logdir="MIL-Test-trainset_$trainset-testset_$testset-$feat-$mil_t-$pool-Date_$now"
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
            --class_weights "balanced" \
            --breast_loader $loader \
            --breast_clahe \
            --clahe_clip_limit 5.0 \
            --breast_padding \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $trainset \
            --testset $testset \
            --data_path $datapath \
            --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

###########################################################################################################
trainset="DDSM+CBIS_CLAHE-benign_malignant"
testset="MIAS_CLAHE-benign_malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/MIAS/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/MIL-Finetune-$trainset-$feat-$mil_t-$pool-*")"
            if [[ $ckpt_path == *$suffix_to_remove ]]; then
                ckpt_path="${ckpt_path%$suffix_to_remove}"
            fi
            ckpt_path="${ckpt_path}/MIL-$mil_t-$pool-best_checkpoint.pth"

            logdir="MIL-Test-trainset_$trainset-testset_$testset-$feat-$mil_t-$pool-Date_$now"
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
            --class_weights "balanced" \
            --breast_loader $loader \
            --breast_padding \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $trainset \
            --testset $testset \
            --data_path $datapath \
            --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

############################################### Benign_Malignant - DDSM+CBIS+MIAS Train #######################################################
trainset="DDSM+CBIS+MIAS_CLAHE-benign_malignant"
testset="DDSM+CBIS+MIAS_CLAHE-benign_malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/MIL-Finetune-$trainset-$feat-$mil_t-$pool-*")"
            if [[ $ckpt_path == *$suffix_to_remove ]]; then
                ckpt_path="${ckpt_path%$suffix_to_remove}"
            fi
            ckpt_path="${ckpt_path}/MIL-$mil_t-$pool-best_checkpoint.pth"

            logdir="MIL-Test-trainset_$trainset-testset_$testset-$feat-$mil_t-$pool-Date_$now"
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
            --class_weights "balanced" \
            --breast_loader $loader \
            --breast_padding \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $trainset \
            --testset $testset \
            --data_path $datapath \
            --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

###########################################################################################################
trainset="DDSM+CBIS+MIAS_CLAHE-benign_malignant"
testset="CMMD-only_mass"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/CMMD/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for feat in "${feature_extractor[@]}"
do
    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do
            now=$(date +"%Y%m%d_%H%M%S")
            ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/MIL-Finetune-$trainset-$feat-$mil_t-$pool-*")"
            if [[ $ckpt_path == *$suffix_to_remove ]]; then
                ckpt_path="${ckpt_path%$suffix_to_remove}"
            fi
            ckpt_path="${ckpt_path}/MIL-$mil_t-$pool-best_checkpoint.pth"

            logdir="MIL-Test-trainset_$trainset-testset_$testset-$feat-$mil_t-$pool-Date_$now"
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
            --class_weights "balanced" \
            --breast_loader $loader \
            --breast_padding \
            --breast_transform_rgb \
            --dataset_type $dataset_type \
            --dataset $trainset \
            --testset $testset \
            --data_path $datapath \
            --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

exit 0