#!/bin/bash
cd SwinFusion
CUDA_VISIBLE_DEVICES=0                     python SwinFusion.py                     --Method SwinFusion                     --model_path /workspace/VIF-Benchmark/Checkpoint/SwinFusion/SwinFusion.pth                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results/test_imgs/SwinFusion                     --is_RGB True
cd ..
