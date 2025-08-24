#!/bin/bash
cd TarDAL
CUDA_VISIBLE_DEVICES=0                     python TarDAL.py                     --Method TarDAL                     --model_path /workspace/VIF-Benchmark/Checkpoint/TarDAL/tardal++.pt                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results/test_imgs/TarDAL                     --is_RGB True
cd ..
