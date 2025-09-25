#!/bin/bash

FILE_PATHS=(
)

model_names=("GuardReasoner-1B" "GuardReasoner-3B" "GuardReasoner-8B")
prompt_lengths=(290 270 250)

for file_path in "${FILE_PATHS[@]}"; do
    echo "start: $file_path"
    echo "----------------------------------------"
    
    for i in "${!model_names[@]}"; do
        echo "start PROMPT_LENGTH=$PROMPT_LENGTH handle: $file_path"
        python inference_step_1_dataset.py \
            --PROMPT_LENGTH ${prompt_lengths[$i]} \
            --model_name ${model_names[$i]} \
            --dataset_path "$file_path" \
            --device "cuda:4"

        if [ $? -ne 0 ]; then
            echo "error：PROMPT_LENGTH=$PROMPT_LENGTH handle $file_path fail"
            exit 1
        fi
        echo "PROMPT_LENGTH=$PROMPT_LENGTH finish"
        echo "----------------------------------------"
    done
    
    echo "file $file_path finish"
    echo "========================================"
done

echo "所有文件和参数组合处理完毕"
