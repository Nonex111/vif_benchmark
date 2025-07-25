#!/bin/bash
cd /workspace/VIF-Benchmark/GAN-FM
CUDA_VISIBLE_DEVICES=0                     python GANFM.py                     --Method GAN-FM                     --model_path /workspace/VIF-Benchmark/Checkpoint/GAN-FM/model.ckpt                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/GAN-FM                     --is_RGB True
cd /workspace/VIF-Benchmark
