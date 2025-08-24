#!/bin/bash
cd PIAFusion
CUDA_VISIBLE_DEVICES=0                     python PIAFusion.py                     --Method PIAFusion                     --model_path /workspace/VIF-Benchmark/Checkpoint/PIAFusion                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results/test_imgs/PIAFusion                     --is_RGB True
cd ..
