#!/bin/bash
cd /workspace/VIF-Benchmark/RFN-Nest
CUDA_VISIBLE_DEVICES=0                     python RFNNest.py                     --Method RFN-Nest                     --model_path_1 /workspace/VIF-Benchmark/Checkpoint/RFN-Nest/RFN_Nest.model                     --model_path_2 /workspace/VIF-Benchmark/Checkpoint/RFN-Nest/NestFuse.model                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/RFN-Nest                     --is_RGB True
cd /workspace/VIF-Benchmark
