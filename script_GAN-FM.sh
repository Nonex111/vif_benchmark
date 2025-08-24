#!/bin/bash
cd GAN-FM
CUDA_VISIBLE_DEVICES=0                     python GANFM.py                     --Method GAN-FM                     --model_path /workspace/VIF-Benchmark/Checkpoint/GAN-FM/model.ckpt                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results/test_imgs/GAN-FM                     --is_RGB True
cd ..
