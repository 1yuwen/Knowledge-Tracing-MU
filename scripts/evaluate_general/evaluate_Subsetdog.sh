#!/bin/bash

export DATA_DIR="/projectnb/ivc-ml/yuwentan"
export PROJECT="Unlearning"

CHECKPOINT_ROOT="/projectnb/ivc-ml/yuwentan/Unlearning/checkpoint"
Unlearn_dataset=Subsetdog
Unlearn_easy_classes=('German short-haired pointer' 'keeshond' 'Samoyed' 'Pomeranian' 'basset hound' 'miniature pinscher' 'English setter' 'beagle' 'Scottish terrier' 'Yorkshire terrier')
#('English setter' 'beagle'  'whippet' 'Ibizan hound' 'Dandie Dinmont terrier' 'standard poodle'  'Border collie'  'Blenheim spaniel' 'cairn terrier' 'Doberman' 'groenendael')
declare -a datasets=("Stanford_Cars" "Food101" "Flower102" "caltech101" "cifar100") 
declare -a methods=('GDiff' 'KL' 'neg' 'Hinge' 'NPO' 'SALUN' 'ME' 'Relabeling') 

declare -A checkpoints
checkpoints["GDiff"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/GDiff/fine/Cosine-Epo_8-Lr_0.00000008/difficult_KLc_5_KLf=20_Margin_2.pth"
checkpoints["KL"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/KL/fine/Cosine-Epo_8-Lr_0.00000008/difficult_final_parameter_margin_0.0_beta_0.0.pth" #/Subsetdog/Unlearning/KL/fine/Cosine-Epo_8-Lr_0.00000008/difficult_KLc_20_KLf=20_Margin_2.pth" #:$CHECKPOINT_ROOT/Compcars/Unlearning/KL/fine/Cosine-Epo_6-Lr_0.00000008/difficult_KLc_40.0_KLf=20.0_Margin_0.0_beta_0.0.pth"#"$CHECKPOINT_ROOT/Compcars/Unlearning/KL/fine/Cosine-Epo_6-Lr_0.00000008/difficult_KLc_0.0_KLf=0.0_Margin_0.0_beta_0.0.pth:$CHECKPOINT_ROOT/Compcars/Unlearning/KL/fine/Cosine-Epo_6-Lr_0.00000008/difficult_KLc_40.0_KLf=20.0_Margin_0.0_beta_0.0.pth"
checkpoints["NPO"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/NPO/fine/Cosine-Epo_8-Lr_0.00000010/difficult_final_parameter_margin_0.0_beta_0.5.pth"
checkpoints["Hinge"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/Hinge/fine/Cosine-Epo_8-Lr_0.00000010/difficult_KLc_10.0_KLf=20.0_Margin_2.0_beta_0.0.pth"
checkpoints["neg"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/neg/fine/Cosine-Epo_8-Lr_0.00000010/difficult_fine_min_acc.pth"
checkpoints["SALUN"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/SALUN/fine/Cosine-Epo_8-Lr_0.00000020/difficult_fine_min_acc.pth"
checkpoints["ME"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/ME/fine/Cosine-Epo_8-Lr_0.00000010/difficult_KLc_0.0_KLf=0.0_Margin_0.0_beta_0.0.pth"
checkpoints["Relabeling"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/Relabeling/fine/Cosine-Epo_8-Lr_0.00000010/difficult_fine_min_acc.pth"



for method in "${methods[@]}"; do
    IFS=':' read -ra ADDR <<< "${checkpoints[$method]}"  # Split the checkpoints string into an array
    for checkpoint_path in "${ADDR[@]}"; do
        for dataset in "${datasets[@]}"; do
            echo "Evaluating $method using checkpoint $checkpoint_path on $dataset dataset"
            python evaluate.py -evaluate_dataset $dataset -template $dataset -unlearn_dataset $Unlearn_dataset -unlearn_fine_classes "${Unlearn_easy_classes[@]}" -unlearning_method $method -model_dir "$checkpoint_path"
        done
    done
done

# #'GDiff'
# declare -A checkpoints
# #checkpoints["GDiff"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/GDiff/fine/Cosine-Epo_8-Lr_0.00000008/specail_KLc_20.0_KLf=20.0_Margin_2.0_beta_0.0.pth"
# #checkpoints["Relabeling"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/Relabeling/fine/Cosine-Epo_8-Lr_0.00000010/easy_fine_min_acc.pth"
# #checkpoints["KL"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/KL/fine/Cosine-Epo_8-Lr_0.00000008/specail_KLc_20.0_KLf=20.0_Margin_0.0_beta_0.0.pth"
# checkpoints["NPO"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/NPO/fine/Cosine-Epo_8-Lr_0.00000010/specail_KLc_5.0_KLf=20.0_Margin_0.0_beta_0.4.pth"
# checkpoints["Hinge"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/Hinge/fine/Cosine-Epo_8-Lr_0.00000010/specail_KLc_10.0_KLf=20.0_Margin_4.0_beta_0.0.pth"
# #checkpoints["neg"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/neg/fine/Cosine-Epo_8-Lr_0.00000010/easy_fine_min_acc.pth"




