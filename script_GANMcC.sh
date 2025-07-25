#!/bin/bash
cd /workspace/VIF-Benchmark/GANMcC
CUDA_VISIBLE_DEVICES=0                     python GANMcC.py                     --Method GANMcC                     --model_path /workspace/VIF-Benchmark/Checkpoint/GANMcC/GANMcC                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/GANMcC                     --is_RGB True
cd /workspace/VIF-Benchmark
