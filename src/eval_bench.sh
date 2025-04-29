#!/bin/bash
# run_models.sh

model_paths=(
    "/data/hf_cache/Qwen2.5-VL-7B-COT-SFT/"
)

file_names=(
    "eval_trial"
)

export DECORD_EOF_RETRY_MAX=20480


for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[$i]}"
    CUDA_VISIBLE_DEVICES=0 python ./src/eval_bench.py --model_path "$model" --file_name "$file_name"
done
