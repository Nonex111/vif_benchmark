#!/bin/bash
cd /workspace/VIF-Benchmark/SuperFusion
CUDA_VISIBLE_DEVICES=0                     python SuperFusion.py                     --Method SuperFusion                     --model_path /workspace/VIF-Benchmark/Checkpoint/SuperFusion/MSRS.pth                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/SuperFusion                     --is_RGB True
cd /workspace/VIF-Benchmark
