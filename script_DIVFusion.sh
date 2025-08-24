#!/bin/bash
cd DIVFusion
CUDA_VISIBLE_DEVICES=0                     python DIVFusion.py                     --Method DIVFusion                     --model_path_1 /workspace/VIF-Benchmark/Checkpoint/DIVFusion/decom.ckpt                     --model_path_2 /workspace/VIF-Benchmark/Checkpoint/DIVFusion/enhance.ckpt                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results/test_imgs/DIVFusion                     --is_RGB True
cd ..
