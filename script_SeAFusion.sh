#!/bin/bash
cd /workspace/VIF-Benchmark/SeAFusion
CUDA_VISIBLE_DEVICES=0                     python SeAFusion.py                     --Method SeAFusion                     --model_path /workspace/VIF-Benchmark/Checkpoint/SeAFusion/SeAFusion.pth                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/SeAFusion                     --is_RGB True
cd /workspace/VIF-Benchmark
