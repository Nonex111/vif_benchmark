#!/bin/bash
cd /workspace/VIF-Benchmark/DIDFuse
CUDA_VISIBLE_DEVICES=0                     python DIDFuse.py                     --Method DIDFuse                     --model_path_1 /workspace/VIF-Benchmark/Checkpoint/DIDFuse/Encoder.pkl                     --model_path_2 /workspace/VIF-Benchmark/Checkpoint/DIDFuse/Decoder.pkl                     --ir_dir /workspace/VIF-Benchmark/datasets/test_imgs/ir                    --vi_dir /workspace/VIF-Benchmark/datasets/test_imgs/vi                     --save_dir /workspace/VIF-Benchmark/Results.llvip/DIDFuse                     --is_RGB True
cd /workspace/VIF-Benchmark
