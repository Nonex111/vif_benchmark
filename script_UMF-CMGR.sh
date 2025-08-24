#!/bin/bash
cd UMF-CMGR
CUDA_VISIBLE_DEVICES=0                     python UMFCMGR.py                     --Method UMF-CMGR                     --model_path /workspace/VIF-Benchmark/Checkpoint/UMF-CMGR/UMF_CMGR.pth                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results/test_imgs/UMF-CMGR                     --is_RGB True
cd ..
