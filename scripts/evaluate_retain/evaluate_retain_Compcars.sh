export DATA_DIR="/projectnb/ivc-ml/yuwentan"
export PROJECT="Unlearning"
CHECKPOINT_ROOT="/projectnb/ivc-ml/yuwentan/Unlearning/checkpoint"

dataset=Compcars
unlearn_goal='fine'
template='Compcars_test'
Unlearn_easy_classes=('Acura MDX' 'Lexus RX' 'Jaguar XK' 'MINI CABRIO' 'Audi A7' 'Audi A5 coupe' 'Cadillac SRX'  'Corvette' 'Mustang')
echo "Classes to unlearn: ${Unlearn_easy_classes[@]}"

declare -A checkpoints
#checkpoints["GDiff"]="$CHECKPOINT_ROOT/Compcars/Unlearning/GDiff/fine/Cosine-Epo_6-Lr_0.00000008/difficult_KLc_20.0_KLf=20.0_Margin_2.0_beta_0.0.pth"
#checkpoints["KL"]="$CHECKPOINT_ROOT/Compcars/Unlearning/KL/fine/Cosine-Epo_6-Lr_0.00000008/difficult_KLc_0.0_KLf=0.0_Margin_0.0_beta_0.0.pth"
#checkpoints["NPO"]="$CHECKPOINT_ROOT/Compcars/Unlearning/NPO/fine/Cosine-Epo_6-Lr_0.00000010/difficult_KLc_20.0_KLf=5.0_Margin_0.0_beta_0.4.pth:$CHECKPOINT_ROOT/Compcars/Unlearning/NPO/fine/Cosine-Epo_6-Lr_0.00000010/difficult_KLc_5.0_KLf=20.0_Margin_0.0_beta_0.4.pth"
#checkpoints["Hinge"]="$CHECKPOINT_ROOT/Compcars/Unlearning/Hinge/fine/Cosine-Epo_6-Lr_0.00000010/difficult_KLc_20.0_KLf=20.0_Margin_2.0_beta_0.0.pth:$CHECKPOINT_ROOT/Compcars/Unlearning/Hinge/fine/Cosine-Epo_6-Lr_0.00000010/difficult_KLc_5.0_KLf=20.0_Margin_2.0_beta_0.0.pth"
#checkpoints["neg"]="$CHECKPOINT_ROOT/Compcars/Unlearning/neg/fine/Cosine-Epo_6-Lr_0.00000010/difficult_fine_min_acc.pth"
#checkpoints["SALUN"]="$CHECKPOINT_ROOT/Compcars/Unlearning/SALUN/fine/Cosine-Epo_6-Lr_0.00000020/difficult_fine_min_acc.pth"
checkpoints["Relabeling"]="$CHECKPOINT_ROOT/Compcars/Unlearning/Relabeling/fine/Cosine-Epo_6-Lr_0.00000020/difficult_fine_min_acc.pth"
checkpoints["ME"]="$CHECKPOINT_ROOT/Compcars/Unlearning/ME/fine/Cosine-Epo_6-Lr_0.00000020/difficult_KLc_0.0_KLf=0.0_Margin_0.0_beta_0.0.pth"

# Loop through methods and their respective checkpoints
for method in "${!checkpoints[@]}"; do
    IFS=':' read -ra ADDR <<< "${checkpoints[$method]}" # Convert string to array using ':' as delimiter
    for checkpoint_path in "${ADDR[@]}"; do
        echo "Evaluating $method using checkpoint $checkpoint_path on $dataset with unlearn activated"
        python evaluate_CLIP_retain.py -unlearning_goal $unlearn_goal -template $template -unlearn_fine_classes "${Unlearn_easy_classes[@]}" -unlearn_dataset $dataset -unlearning_method $method -model_dir "$checkpoint_path" -unlearn
        echo "Evaluating $method using checkpoint $checkpoint_path on $dataset without unlearn activated"
        python evaluate_CLIP_retain.py -unlearning_goal $unlearn_goal -template $template -unlearn_fine_classes "${Unlearn_easy_classes[@]}" -unlearn_dataset $dataset -unlearning_method $method -model_dir "$checkpoint_path"
    done
done

