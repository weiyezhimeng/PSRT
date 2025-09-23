#!/bin/bash

FILE_PATHS=(
)

model_names=("Ministral-8B-Instruct-2410")
prompt_lengths=(260)

for file_path in "${FILE_PATHS[@]}"; do
    echo "start: $file_path"
    echo "----------------------------------------"
    
    for i in "${!model_names[@]}"; do
        echo "PROMPT_LENGTH=$PROMPT_LENGTH handle: $file_path"
        python inference_step_1_dataset.py \
            --PROMPT_LENGTH ${prompt_lengths[$i]} \
            --prompt_prefix "../output_step_1" \
            --model_prefix "../output_sft_final" \
            --model_name ${model_names[$i]} \
            --save_path_name "./result_step_1" \
            --dataset_path "$file_path" \
            --device "cuda:2"

        if [ $? -ne 0 ]; then
            echo "Wrong: PROMPT_LENGTH=$PROMPT_LENGTH handle $file_path fail"
            exit 1
        fi
        echo "PROMPT_LENGTH=$PROMPT_LENGTH finish"
        echo "----------------------------------------"
    done
done