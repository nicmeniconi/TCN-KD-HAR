#!/bin/bash

# Parameters for the grid search
seed=42
dpath="/Volumes/Data_Drive/datasets/VIDIMU/dataset/videoandimusyncrop"
out_path="/Volumes/Data_Drive/vidimu_gridsearch_out/gridsearch_11_4"
activities_json='["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A13"]'

secs=(0.5 0.75 1.5 4.0)
split=0.8
batch_size=32
epochs=200
lr=1e-3
factor=0.1
patience=10
early_stop_patience=30
cv_folds=5
modelnames=("Pos_ResNet" "Ang_ResNet" "AngPos_ResNet")

depth=(1 2 7 14)

mkdir -p "$out_path"
echo "Save results to ${out_path}"

for d in "${depth[@]}"; do
    for model in "${modelnames[@]}"; do
        for s in "${secs[@]}"; do
            logfile="${out_path}/log_${model}_${d}_${s}_${seed}.txt"

            best_model_file="best_${model}_fold*_${d}_${s}_${seed}_acc_*.pth"
            found_file=$(find "${out_path}" -type f -name "$best_model_file")

            if [[ -n "$found_file" ]]; then
                echo "Skipping run for model=${model}, number of blocks=${d}, secs=${s}, folds=${cv_folds}, seed=${seed} (already exists)"
                continue
            fi

            echo "Starting run for model=${model}, number of blocks=${d}, secs=${s}, folds=${cv_folds}, seed=${seed}"
            unbuffer python "${model}.py" \
                --seed="$seed" \
                --dpath="$dpath" \
                --model_out_path="$out_path" \
                --activities="$activities_json" \
                --secs="$s" \
                --split="$split" \
                --batch_size="$batch_size" \
                --depth="$d" \
                --epochs="$epochs" \
                --cv_folds="$cv_folds" \
                --early_stop_patience="$early_stop_patience" \
                --lr="$lr" \
                --factor="$factor" \
                --patience="$patience" \
                --modelname="$model" | tee "${out_path}/log_${model}_${d}_${s}_${seed}.txt"

            echo "Completed run for model=${model}, number of blocks=${d}, secs=${s}, folds=${cv_folds}, seed=${seed}"
        done
    done
done
