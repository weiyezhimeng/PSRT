#!/usr/bin/env bash
# train.sh

# model and dataset
MODEL_NAME_OR_PATH=""
DATASET_PATH=""
OUTPUT_DIR=""

# Prompt Tuning
PROMPT_LENGTH=100

MAX_EPOCHS=1
BATCH_SIZE=4
LR=5e-5
WEIGHT_DECAY=0.0
WARM_RATIO=0.01
LR_SCHEDULER_TYPE="cosine"
GRAD_ACCUM_STEPS=8
LOGGING_STEPS=1
SAVE_STEPS=500
SEED=42

export CUDA_VISIBLE_DEVICES=0,1,2
GPUS_PER_NODE=3    # GPU num
NNODES=1           # cluster num
NODE_RANK=0        # current index
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

echo ">>> launching prompt tuning with ${GPUS_PER_NODE} GPUs"
torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train_step_1.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_length ${PROMPT_LENGTH} \
    --max_epochs ${MAX_EPOCHS} \
    --per_device_batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARM_RATIO} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --local_rank 0