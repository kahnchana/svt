#!/bin/bash

PROJECT_PATH="$HOME/repo/svt"
CHECKPOINT="path/to/checkpoint.pth"
DATASET="ucf101"
DATA_PATH="${HOME}/repo/mmaction2/data/${DATASET}"

cd "$PROJECT_PATH" || exit

export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  eval_knn.py \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --batch_size_per_gpu 128 \
  --nb_knn 5 \
  --temperature 0.07 \
  --num_workers 4 \
  --dataset "$DATASET" \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/knn_splits" \
  DATA.PATH_PREFIX f"${DATA_PATH}/videos"
