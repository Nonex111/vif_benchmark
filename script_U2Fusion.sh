#!/bin/bash
cd U2Fusion
CUDA_VISIBLE_DEVICES=0                     python U2Fusion.py                     --Method U2Fusion                     --model_path /workspace/VIF-Benchmark/Checkpoint/U2Fusion/model.ckpt                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results/test_imgs/U2Fusion                     --is_RGB True
cd ..
