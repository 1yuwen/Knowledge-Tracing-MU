#!/bin/bash

export DATA_DIR="/projectnb/ivc-ml/yuwentan"
export PROJECT="Unlearning"

Unlearn_easy_classes=('Acura MDX' 'Lexus RX' 'Jaguar XK' 'MINI CABRIO' 'Audi A7' 'Audi A5 coupe' 'Cadillac SRX'  'Corvette' 'Mustang')
Unlearn_dataset=Compcars
CHECKPOINT_ROOT="/projectnb/ivc-ml/yuwentan/Unlearning/checkpoint"
declare -a datasets=( "Food101" "Flower102" "caltech101" "OxfordPet" "cifar100")
declare -a methods=('KL' 'Hinge' 'NPO' 'Relabeling' 'neg' 'GDiff' 'SALUN') 

declare -A checkpoints
checkpoints["GDiff"]="$CHECKPOINT_ROOT/Compcars/Unlearning/GDiff/fine/Cosine-Epo_6-Lr_0.00000008/difficult_KLc_20.0_KLf=20.0_Margin_2.0_beta_0.0.pth"
checkpoints["KL"]="$CHECKPOINT_ROOT/Compcars/Unlearning/KL/fine/Cosine-Epo_6-Lr_0.00000008/difficult_KLc_0.0_KLf=0.0_Margin_0.0_beta_0.0.pth" #:$CHECKPOINT_ROOT/Compcars/Unlearning/KL/fine/Cosine-Epo_6-Lr_0.00000008/difficult_KLc_40.0_KLf=20.0_Margin_0.0_beta_0.0.pth"#"$CHECKPOINT_ROOT/Compcars/Unlearning/KL/fine/Cosine-Epo_6-Lr_0.00000008/difficult_KLc_0.0_KLf=0.0_Margin_0.0_beta_0.0.pth:$CHECKPOINT_ROOT/Compcars/Unlearning/KL/fine/Cosine-Epo_6-Lr_0.00000008/difficult_KLc_40.0_KLf=20.0_Margin_0.0_beta_0.0.pth"
checkpoints["NPO"]="$CHECKPOINT_ROOT/Compcars/Unlearning/NPO/fine/Cosine-Epo_6-Lr_0.00000010/difficult_KLc_20.0_KLf=5.0_Margin_0.0_beta_0.4.pth:$CHECKPOINT_ROOT/Compcars/Unlearning/NPO/fine/Cosine-Epo_6-Lr_0.00000010/difficult_KLc_5.0_KLf=20.0_Margin_0.0_beta_0.4.pth"
checkpoints["Hinge"]="$CHECKPOINT_ROOT/Compcars/Unlearning/Hinge/fine/Cosine-Epo_6-Lr_0.00000010/difficult_KLc_20.0_KLf=20.0_Margin_2.0_beta_0.0.pth:$CHECKPOINT_ROOT/Compcars/Unlearning/Hinge/fine/Cosine-Epo_6-Lr_0.00000010/difficult_KLc_5.0_KLf=20.0_Margin_2.0_beta_0.0.pth"
checkpoints["neg"]="$CHECKPOINT_ROOT/Compcars/Unlearning/neg/fine/Cosine-Epo_6-Lr_0.00000010/difficult_fine_min_acc.pth"
checkpoints["SALUN"]="$CHECKPOINT_ROOT/Compcars/Unlearning/SALUN/fine/Cosine-Epo_6-Lr_0.00000020/difficult_fine_min_acc.pth"
checkpoints["ME"]="$CHECKPOINT_ROOT/Compcars/Unlearning/ME/fine/Cosine-Epo_6-Lr_0.00000020/difficult_KLc_0.0_KLf=0.0_Margin_0.0_beta_0.0.pth"
checkpoints["Relabeling"]="$CHECKPOINT_ROOT/Compcars/Unlearning/Relabeling/fine/Cosine-Epo_6-Lr_0.00000020/difficult_fine_min_acc.pth:$CHECKPOINT_ROOT/Compcars/Unlearning/Relabeling/fine/Cosine-Epo_6-Lr_0.00000010/difficult_fine_min_acc.pth"

# Loop through all methods and datasets
for method in "${methods[@]}"; do
    IFS=':' read -ra ADDR <<< "${checkpoints[$method]}"  # Split the checkpoints string into an array
    for checkpoint_path in "${ADDR[@]}"; do
        for dataset in "${datasets[@]}"; do
            echo "Evaluating $method using checkpoint $checkpoint_path on $dataset dataset"
            python evaluate.py -evaluate_dataset $dataset -template $dataset -unlearn_dataset $Unlearn_dataset -unlearn_fine_classes "${Unlearn_easy_classes[@]}" -unlearning_method $method -model_dir "$checkpoint_path"
        done
    done
done
