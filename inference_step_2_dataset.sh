#!/bin/bash

FILE_PATHS=(
)

for file_path in "${FILE_PATHS[@]}"; do
    echo "$file_path"
    echo "----------------------------------------"
    for PROMPT_LENGTH in {120..200..20}; do
        python inference_step_2_dataset.py \
            --PROMPT_LENGTH $PROMPT_LENGTH \
            --prompt_prefix "../output_step_1" \
            --model_prefix "../huggingface_model" \
            --model_name "Qwen3-8B" \
            --save_path_name "./result_step_2" \
            --dataset_path "$file_path" \
            --device "cuda:5"
        
        if [ $? -ne 0 ]; then
            exit 1
        fi
        echo "PROMPT_LENGTH=$PROMPT_LENGTH"
        echo "----------------------------------------"
    done
done
