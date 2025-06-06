#!/bin/bash
export DATA_DIR="/projectnb/ivc-ml/yuwentan"
export PROJECT="Unlearning"
GPU="0"
SEED=1
train_batch=16
test_batch=128
goal="fine"
epochs=6
Unlearn_classes=('Acura MDX' 'Lexus RX' 'Jaguar XK' 'MINI CABRIO' 'Audi A7' 'Audi A5 coupe' 'Cadillac SRX' 'Corvette' 'Mustang')
function run_training {
    echo "Running training with method: $1"
    python train.py -unlearn_dataset Compcars \
                    -template Compcars_test \
                    -unlearn_fine_classes "${Unlearn_classes[@]}" \
                    -unlearn_coarse_classes Terrier \
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

run_training "GDiff" 8e-8 0 20 20 0 0
run_training "GA" 8e-8 0 0 0 0 0
run_training "KL" 8e-8 0 40 20 0 0
run_training "Relabeling" 2e-7 0 0 0 0 0
run_training "SALUN" 2e-7 0 0 0 0 0
run_training "ME" 2e-7 0 0 0 0 0
run_training "neg" 1e-7 0 0 0 0 0 0
run_training "NPO" 1e-7 0.4 20 5 0 0
run_training "Hinge" 1e-7 0 20 20 0 2