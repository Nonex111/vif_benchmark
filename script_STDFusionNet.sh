#!/bin/bash
cd /workspace/VIF-Benchmark/STDFusionNet
CUDA_VISIBLE_DEVICES=0                     python STDFusionNet.py                     --Method STDFusionNet                     --model_path /workspace/VIF-Benchmark/Checkpoint/STDFusionNet/Fusion.model-29                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/STDFusionNet                     --is_RGB True
cd /workspace/VIF-Benchmark
