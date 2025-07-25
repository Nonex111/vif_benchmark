#!/bin/bash
cd /workspace/VIF-Benchmark/U2Fusion
CUDA_VISIBLE_DEVICES=0                     python U2Fusion.py                     --Method U2Fusion                     --model_path /workspace/VIF-Benchmark/Checkpoint/U2Fusion/model.ckpt                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/U2Fusion                     --is_RGB True
cd /workspace/VIF-Benchmark
