import os
from tqdm import tqdm 
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

ir_dir = os.path.join(os.getcwd(), 'datasets/test_imgs/ir')
vi_dir = os.path.join(os.getcwd(), 'datasets/test_imgs/vi')

model_path_dict = dict()
model_path_dict_1 = dict()
model_path_dict_2 = dict()

model_path_dict_1['CSF'] = os.path.join(os.getcwd(), 'Checkpoint/CSF/EC.ckpt')
model_path_dict_2['CSF'] = os.path.join(os.getcwd(), 'Checkpoint/CSF/ED.ckpt')

model_path_dict_1['CUFD'] = os.path.join(os.getcwd(), 'Checkpoint/CUFD/1part1_model.ckpt')
model_path_dict_2['CUFD'] = os.path.join(os.getcwd(), 'Checkpoint/CUFD/part2_model.ckpt')

model_path_dict_1['DIDFuse'] = os.path.join(os.getcwd(), 'Checkpoint/DIDFuse/Encoder.pkl')
model_path_dict_2['DIDFuse'] = os.path.join(os.getcwd(), 'Checkpoint/DIDFuse/Decoder.pkl')

model_path_dict_1['DIVFusion'] = os.path.join(os.getcwd(), 'Checkpoint/DIVFusion/decom.ckpt')
model_path_dict_2['DIVFusion'] = os.path.join(os.getcwd(), 'Checkpoint/DIVFusion/enhance.ckpt')

model_path_dict_1['RFN-Nest'] = os.path.join(os.getcwd(), 'Checkpoint/RFN-Nest/RFN_Nest.model')
model_path_dict_2['RFN-Nest'] = os.path.join(os.getcwd(), 'Checkpoint/RFN-Nest/NestFuse.model')

model_path_dict['DenseFuse'] = os.path.join(os.getcwd(), 'Checkpoint/DenseFuse/DeseFuse.ckpt')

model_path_dict['FusionGAN'] = os.path.join(os.getcwd(), 'Checkpoint/FusionGAN/FusionGAN')

model_path_dict['GAN-FM'] = os.path.join(os.getcwd(), 'Checkpoint/GAN-FM/model.ckpt')

model_path_dict['GANMcC'] = os.path.join(os.getcwd(), 'Checkpoint/GANMcC/GANMcC')

model_path_dict['NestFuse'] = os.path.join(os.getcwd(), 'Checkpoint/NestFuse/nestfuse.model')

model_path_dict['PIAFusion'] = os.path.join(os.getcwd(), 'Checkpoint/PIAFusion')

model_path_dict['PMGI'] = os.path.join(os.getcwd(), 'Checkpoint/PMGI/PMGI')

model_path_dict['SDNet'] = os.path.join(os.getcwd(), 'Checkpoint/SDNet/SDNet.model')

model_path_dict['STDFusionNet'] = os.path.join(os.getcwd(), 'Checkpoint/STDFusionNet/Fusion.model-29')

model_path_dict['SeAFusion'] = os.path.join(os.getcwd(), 'Checkpoint/SeAFusion/SeAFusion.pth')

model_path_dict['SuperFusion'] = os.path.join(os.getcwd(), 'Checkpoint/SuperFusion/MSRS.pth')

model_path_dict['SwinFusion'] = os.path.join(os.getcwd(), 'Checkpoint/SwinFusion/SwinFusion.pth')

model_path_dict['TarDAL'] = os.path.join(os.getcwd(), 'Checkpoint/TarDAL/tardal++.pt')

model_path_dict['U2Fusion'] = os.path.join(os.getcwd(), 'Checkpoint/U2Fusion/model.ckpt')

model_path_dict['IFCNN'] = os.path.join(os.getcwd(), 'Checkpoint/IFCNN/IFCNN-MAX.pth')

model_path_dict['UMF-CMGR'] = os.path.join(os.getcwd(), 'Checkpoint/UMF-CMGR/UMF_CMGR.pth')

# 可以根据需要选择要运行的方法
Method_list = [
                'CSF', 'CUFD', 'DIDFuse', 'DIVFusion', 'DenseFuse',
               'FusionGAN', 'GAN-FM', 'GANMcC', 'IFCNN', 'NestFuse', 
               'PIAFusion', 'PMGI', 'RFN-Nest', 'SDNet', 'STDFusionNet', 
               'SeAFusion', 'SuperFusion', 'SwinFusion', 'TarDAL', 'U2Fusion', 
               'UMF-CMGR'
]
print(len(Method_list))
two_model_list = ['CSF', 'CUFD', 'DIDFuse', 'DIVFusion', 'RFN-Nest']

# 获取VIF-Benchmark的绝对路径
vif_benchmark_path = os.getcwd()

# 生成脚本文件
for Method in Method_list:    
    save_dir = os.path.join(vif_benchmark_path, 'Results.llvip/', Method)
    if Method not in two_model_list:
        with open('script_' + Method + '.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write("cd {}/{}\n".format(vif_benchmark_path, Method))
            print(Method.replace('-', ''))
            f.write("CUDA_VISIBLE_DEVICES=0 \
                    python {}.py \
                    --Method {} \
                    --model_path {} \
                    --ir_dir {}\
                    --vi_dir {} \
                    --save_dir {} \
                    --is_RGB {}\n".format(Method.replace('-', ''), Method, model_path_dict[Method], ir_dir, vi_dir, save_dir, True))
            f.write("cd {}\n".format(vif_benchmark_path))
    else:
        with open('script_' + Method + '.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write("cd {}/{}\n".format(vif_benchmark_path, Method))
            print(Method.replace('-', ''))
            f.write("CUDA_VISIBLE_DEVICES=0 \
                    python {}.py \
                    --Method {} \
                    --model_path_1 {} \
                    --model_path_2 {} \
                    --ir_dir {}\
                    --vi_dir {} \
                    --save_dir {} \
                    --is_RGB {}\n".format(Method.replace('-', ''), Method, model_path_dict_1[Method], model_path_dict_2[Method], ir_dir, vi_dir, save_dir, True))
            f.write("cd {}\n".format(vif_benchmark_path))

# 串行执行脚本
print("Running methods in serial mode...")
for Method in Method_list:
    print("Processing method: {}".format(Method))
    # 运行脚本前清理内存
    os.system('python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"')
    # 运行脚本
    try:
        os.system('bash script_' + Method + '.sh')
    except Exception as e:
        print(f"Error running {Method}: {str(e)}")
    # 等待更长时间，确保GPU内存释放
    time.sleep(10)
    # 彻底清理内存
    os.system('python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"')

print("All methods processed.")