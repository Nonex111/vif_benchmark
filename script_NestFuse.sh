#!/bin/bash
cd /workspace/VIF-Benchmark/NestFuse
CUDA_VISIBLE_DEVICES=0                     python NestFuse.py                     --Method NestFuse                     --model_path /workspace/VIF-Benchmark/Checkpoint/NestFuse/nestfuse.model                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/NestFuse                     --is_RGB True
cd /workspace/VIF-Benchmark
