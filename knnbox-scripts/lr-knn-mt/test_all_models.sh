#!/bin/bash
GPU_INDEX=1

if [ $# -ne 1 ];
then
    echo Require dataset name it/koran/law/medical
    exit 0
fi 

python inference.py --model base --dataset $1 --run-3-time  --single-gpu-index $GPU_INDEX
python inference.py --model vanilla --dataset $1 --run-3-time  --single-gpu-index $GPU_INDEX
python inference.py --model lr --dataset $1 --run-3-time  --single-gpu-index $GPU_INDEX
python inference.py --model adaptive --dataset $1 --run-3-time  --single-gpu-index $GPU_INDEX
python inference.py --model lr_adaptive --dataset $1 --run-3-time  --single-gpu-index $GPU_INDEX
python inference.py --model pck --dataset $1 --run-3-time  --single-gpu-index $GPU_INDEX
python inference.py --model lr_pck --dataset $1 --run-3-time  --single-gpu-index $GPU_INDEX