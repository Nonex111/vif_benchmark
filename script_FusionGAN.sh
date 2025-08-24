#!/bin/bash
cd FusionGAN
CUDA_VISIBLE_DEVICES=0                     python FusionGAN.py                     --Method FusionGAN                     --model_path /workspace/VIF-Benchmark/Checkpoint/FusionGAN/FusionGAN                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results/test_imgs/FusionGAN                     --is_RGB True
cd ..
