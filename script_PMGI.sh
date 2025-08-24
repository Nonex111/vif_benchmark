#!/bin/bash
cd PMGI
CUDA_VISIBLE_DEVICES=0                     python PMGI.py                     --Method PMGI                     --model_path /workspace/VIF-Benchmark/Checkpoint/PMGI/PMGI                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results/test_imgs/PMGI                     --is_RGB True
cd ..
