#!/bin/bash
cd /workspace/VIF-Benchmark/DenseFuse
CUDA_VISIBLE_DEVICES=0                     python DenseFuse.py                     --Method DenseFuse                     --model_path /workspace/VIF-Benchmark/Checkpoint/DenseFuse/DeseFuse.ckpt                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/DenseFuse                     --is_RGB True
cd /workspace/VIF-Benchmark
