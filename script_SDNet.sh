#!/bin/bash
cd /workspace/VIF-Benchmark/SDNet
CUDA_VISIBLE_DEVICES=0                     python SDNet.py                     --Method SDNet                     --model_path /workspace/VIF-Benchmark/Checkpoint/SDNet/SDNet.model                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/SDNet                     --is_RGB True
cd /workspace/VIF-Benchmark
