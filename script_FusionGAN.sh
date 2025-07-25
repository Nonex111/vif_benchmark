#!/bin/bash
cd /workspace/VIF-Benchmark/FusionGAN
CUDA_VISIBLE_DEVICES=0                     python FusionGAN.py                     --Method FusionGAN                     --model_path /workspace/VIF-Benchmark/Checkpoint/FusionGAN/FusionGAN                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/FusionGAN                     --is_RGB True
cd /workspace/VIF-Benchmark
