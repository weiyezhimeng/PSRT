#!/usr/bin/env bash
# train.sh

MODEL_LIST=(
)

DATASET_PATH=""
BASE_OUTPUT_DIR="./output_step_1"

MAX_EPOCHS=1
BATCH_SIZE=4
LR=5e-5
WEIGHT_DECAY=0.0
WARM_RATIO=0.01
LR_SCHEDULER_TYPE="cosine"
GRAD_ACCUM_STEPS=1
LOGGING_STEPS=1
SAVE_STEPS=1000
SEED=42
max_total_length=2048

export CUDA_VISIBLE_DEVICES=5,6,7
GPUS_PER_NODE=3
NNODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.0"
MASTER_PORT=29500

for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
  MODEL_BASENAME=$(basename "${MODEL_NAME_OR_PATH}")
  for PROMPT_LENGTH in 210 230 270 290; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}"

    echo ">>> launching prompt tuning: model=${MODEL_BASENAME}, prompt_length=${PROMPT_LENGTH}"

    torchrun \
      --nnodes=${NNODES} \
      --nproc_per_node=${GPUS_PER_NODE} \
      --node_rank=${NODE_RANK} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      train_step_1.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --max_total_length "${max_total_length}" \
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
        --local_rank 0 \
        --gradient_checkpointing_enable
  done
done
