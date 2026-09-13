#!/bin/bash

# Find a GPU with less than 10 GB currently used.
# Prefer GPUs with the most free memory.
GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 10000 {print $1, $2}' \
    | sort -k2n \
    | head -1 \
    | awk '{print $1}')

if [ -z "$GPU_ID" ]; then
    echo "ERROR: No GPU with <10 GB memory usage is available."
    nvidia-smi
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "======================================"
echo "SELECTED PHYSICAL GPU: $GPU_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "======================================"

nvidia-smi --query-gpu=index,name,memory.used,memory.free \
    --format=csv
