#!/bin/bash
cd /workspace/VIF-Benchmark/CSF
CUDA_VISIBLE_DEVICES=0                     python CSF.py                     --Method CSF                     --model_path_1 /workspace/VIF-Benchmark/Checkpoint/CSF/EC.ckpt                     --model_path_2 /workspace/VIF-Benchmark/Checkpoint/CSF/ED.ckpt                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/CSF                     --is_RGB True
cd /workspace/VIF-Benchmark
