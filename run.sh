#!/bin/bash
source ~/venv/alphagrad/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
python src/alphagrad/alphazero/vertex_A0.py task=$1 name=$2 seed=$3 wandb_mode="offline"

