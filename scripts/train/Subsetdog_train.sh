#!/bin/bash
export DATA_DIR="/projectnb/ivc-ml/yuwentan"
export PROJECT="Unlearning"
GPU="0"
SEED=1
train_batch=32
test_batch=128
epochs=8
goal="fine"
Unlearn_easy_classes=('English setter' 'beagle'  'whippet' 'Ibizan hound' 'Dandie Dinmont terrier' 'standard poodle'  'Border collie'  'Blenheim spaniel' 'cairn terrier' 'Doberman' 'groenendael')
Unlearn_medium_classes=('Irish setter' 'Gordon setter' 'basset hound' 'Airedale terrier' 'Shih-Tzu' 'miniature pinscher' 'Alaskan malamute' 'flat-coated retriever' 'Chesapeake Bay retriever' 'Sealyham terrier')
Unlearn_difficult_classes=('German short-haired pointer' 'Boston terrier' 'West Highland white terrier' 'Labrador retriever' 'golden retriever' 'German shepherd dog' 'keeshond' 'Samoyed' 'Pomeranian' 'Border terrier')
Unlearn_special_classes=('German short-haired pointer' 'keeshond' 'Samoyed' 'Pomeranian' 'basset hound' 'miniature pinscher' 'English setter' 'beagle' 'Scottish terrier' 'Yorkshire terrier')						
function run_training {
    echo "Running training with method: $1"
    python train.py -unlearn_dataset Subsetdog \
                    -template Subsetdog_test \
                    -unlearn_coarse_classes Terrier \
                    -unlearn_fine_classes "${Unlearn_difficult_classes[@]}" \
                    -project $PROJECT \
                    -backbonename ViT-L/14 \
                    -epochs $epochs \
                    -lr $2 \
                    -schedule Cosine \
                    -unlearning_method $1 \
                    -unlearning_goal $goal \
                    -seed $SEED \
                    -batch_size_base $train_batch \
                    -test_batch_size $test_batch \
                    -gpu $GPU \
                    -beta $3 \
                    -KL_c $4 \
                    -KL_f $5 \
                    -margin_c $6 \
                    -margin_f $7
    echo "Training completed for method: $1"
}

run_training "GDiff" 8e-8 0 20 20 1 2
run_training "GA" 8e-8 0 0 0 0 0
run_training "KL" 8e-8 20 20 0 0 0
run_training "Relabeling" 1e-7 0 0 0 0 0
run_training "SALUN" 2e-7 0 0 0 0 0
run_training "ME" 1e-7 0 0 0 0 0
run_training "NPO" 1e-7 0.5 5 20 0 0
run_training "neg" 1e-7 0 0 0 0 0 0
run_training "Hinge" 1e-7 0 10 20 0 2