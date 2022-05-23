#!/bin/bash

PROJECT_PATH="$HOME/repo/svt"
EXP_NAME="le_001"
DATASET="ucf101"
DATA_PATH="${HOME}/repo/mmaction2/data/${DATASET}"
CHECKPOINT="path/to/checkpoint.pth"

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  eval_linear.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 20 \
  --lr 0.001 \
  --batch_size_per_gpu 16 \
  --num_workers 4 \
  --num_labels 101 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/eval/$EXP_NAME" \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/splits" \
  DATA.PATH_PREFIX f"${DATA_PATH}/videos" \
  DATA.USE_FLOW False
