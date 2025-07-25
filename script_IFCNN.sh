#!/bin/bash
cd /workspace/VIF-Benchmark/IFCNN
CUDA_VISIBLE_DEVICES=0                     python IFCNN.py                     --Method IFCNN                     --model_path /workspace/VIF-Benchmark/Checkpoint/IFCNN/IFCNN-MAX.pth                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/IFCNN                     --is_RGB True
cd /workspace/VIF-Benchmark
