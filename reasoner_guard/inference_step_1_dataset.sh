#!/bin/bash

# 定义要处理的文件路径列表
FILE_PATHS=(
)

model_names=("GuardReasoner-1B" "GuardReasoner-3B" "GuardReasoner-8B")
prompt_lengths=(290 270 250)

# 循环每个文件路径
for file_path in "${FILE_PATHS[@]}"; do
    echo "开始处理文件: $file_path"
    echo "----------------------------------------"
    
    # 循环不同的PROMPT_LENGTH值（50到200，间隔25）
    for i in "${!model_names[@]}"; do
        echo "执行 PROMPT_LENGTH=$PROMPT_LENGTH 处理文件: $file_path"
        python inference_step_1_dataset.py \
            --PROMPT_LENGTH ${prompt_lengths[$i]} \
            --model_name ${model_names[$i]} \
            --dataset_path "$file_path" \
            --device "cuda:4"

        # 检查命令执行结果
        if [ $? -ne 0 ]; then
            echo "错误：PROMPT_LENGTH=$PROMPT_LENGTH 处理文件 $file_path 失败"
            exit 1
        fi
        echo "PROMPT_LENGTH=$PROMPT_LENGTH 处理完成"
        echo "----------------------------------------"
    done
    
    echo "文件 $file_path 所有参数处理完成"
    echo "========================================"
done

echo "所有文件和参数组合处理完毕"
