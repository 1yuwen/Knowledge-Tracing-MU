export DATA_DIR="/projectnb/ivc-ml/yuwentan"
export PROJECT="Unlearning"

CHECKPOINT_ROOT="/projectnb/ivc-ml/yuwentan/Unlearning/checkpoint"

dataset=Subsetdog
unlearn_goal='fine'
template='Subsetdog_test'
Unlearn_difficult_classes=('German short-haired pointer' 'Boston terrier' 'West Highland white terrier' 'Labrador retriever' 'golden retriever' 'German shepherd dog' 'keeshond' 'Samoyed' 'Pomeranian' 'Border terrier')
Unlearn_easy_classes=('German short-haired pointer' 'keeshond' 'Samoyed' 'Pomeranian' 'basset hound' 'miniature pinscher' 'English setter' 'beagle' 'Scottish terrier' 'Yorkshire terrier')						
echo "Classes to unlearn: ${unlearn_easy_classes[@]}"
declare -A checkpoints
checkpoints["Hinge"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/Hinge/fine/Cosine-Epo_128-Lr_0.00000010/10_difficult_KLc_10.0_KLf=20.0_Margin_2.0_beta_0.0.pth:$CHECKPOINT_ROOT/Subsetdog/Unlearning/Hinge/fine/Cosine-Epo_52-Lr_0.00000010/30_difficult_KLc_10.0_KLf=20.0_Margin_2.0_beta_0.0.pth:$CHECKPOINT_ROOT/Subsetdog/Unlearning/Hinge/fine/Cosine-Epo_10-Lr_0.00000010/150_difficult_KLc_10.0_KLf=20.0_Margin_2.0_beta_0.0.pth"
checkpoints["Hinge"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/Hinge/fine/Cosine-Epo_8-Lr_0.00000010/medium_KLc_10.0_KLf=20.0_Margin_2.0_beta_0.0.pth"
checkpoints["ME"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/ME/fine/Cosine-Epo_8-Lr_0.00000010/easy_KLc_0.0_KLf=0.0_Margin_0.0_beta_0.0.pth"
checkpoints["SALUN"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/SALUN/fine/Cosine-Epo_8-Lr_0.00000020/easy_fine_min_acc.pth"
checkpoints["GDiff"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/GDiff/fine/Cosine-Epo_8-Lr_0.00000008/specail_KLc_20.0_KLf=20.0_Margin_2.0_beta_0.0.pth"
checkpoints["KL"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/KL/fine/Cosine-Epo_8-Lr_0.00000008/easy_KLc_0.0_KLf=0.0_Margin_0.0_beta_0.0.pth"
checkpoints["NPO"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/NPO/fine/Cosine-Epo_8-Lr_0.00000010/specail_KLc_5.0_KLf=20.0_Margin_0.0_beta_0.4.pth"
checkpoints["NPO"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/NPO/fine/Cosine-Epo_8-Lr_0.00000010/special_KLc_5.0_KLf=20.0_Margin_0.0_beta_0.4.pth"
checkpoints["NPO"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/NPO/fine/Cosine-Epo_8-Lr_0.00000010/special_KLc_5.0_KLf=20.0_Margin_0.0_beta_0.5.pth"
checkpoints["NPO"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/NPO/fine/Cosine-Epo_8-Lr_0.00000010/special_KLc_5.0_KLf=20.0_Margin_0.0_beta_0.6.pth"
checkpoints["Hinge"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/Hinge/fine/Cosine-Epo_8-Lr_0.00000010/specail_KLc_10.0_KLf=20.0_Margin_4.0_beta_0.0.pth"
checkpoints["Hinge"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/Hinge/fine/Cosine-Epo_8-Lr_0.00000010/special_KLc_10.0_KLf=20.0_Margin_3.0_beta_0.0.pth"
checkpoints["Hinge"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/Hinge/fine/Cosine-Epo_8-Lr_0.00000010/special_KLc_10.0_KLf=20.0_Margin_4.0_beta_0.0.pth"
checkpoints["neg"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/neg/fine/Cosine-Epo_8-Lr_0.00000010/easy_fine_min_acc.pth"
checkpoints["Relabeling"]="$CHECKPOINT_ROOT/Subsetdog/Unlearning/Relabeling/fine/Cosine-Epo_8-Lr_0.00000010/easy_fine_min_acc.pth"

for method in "${!checkpoints[@]}"; do
    IFS=':' read -ra ADDR <<< "${checkpoints[$method]}" # Convert string to array using ':' as delimiter
    for checkpoint_path in "${ADDR[@]}"; do
        echo "Evaluating $method using checkpoint $checkpoint_path on $dataset with unlearn activated"
        python evaluate_retain.py -unlearning_goal $unlearn_goal -template $template -unlearn_fine_classes "${Unlearn_difficult_classes[@]}" -unlearn_dataset $dataset -unlearning_method $method -model_dir "$checkpoint_path" -unlearn
        echo "Evaluating $method using checkpoint $checkpoint_path on $dataset without unlearn activated"
        python evaluate_retain.py -unlearning_goal $unlearn_goal -template $template -unlearn_fine_classes "${Unlearn_difficult_classes[@]}" -unlearn_dataset $dataset -unlearning_method $method -model_dir "$checkpoint_path"
    done
done

