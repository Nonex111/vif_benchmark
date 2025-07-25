#!/bin/bash
cd /workspace/VIF-Benchmark/CUFD
CUDA_VISIBLE_DEVICES=0                     python CUFD.py                     --Method CUFD                     --model_path_1 /workspace/VIF-Benchmark/Checkpoint/CUFD/1part1_model.ckpt                     --model_path_2 /workspace/VIF-Benchmark/Checkpoint/CUFD/part2_model.ckpt                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/CUFD                     --is_RGB True
cd /workspace/VIF-Benchmark
