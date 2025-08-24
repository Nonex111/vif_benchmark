"""
单图像融合质量评价脚本。

本程序用于对单张红外与可见光图像融合结果进行多项主流评价指标的计算。

用法：
    直接运行本脚本，将会对指定路径下的融合图像、红外图像、可见光图像进行评价，并输出各项指标。
    可根据需要修改 __main__ 部分的文件路径。

依赖：
    - PIL
    - numpy
    - Metric.py（需包含各评价函数）

示例：
    python eval_one_image.py
"""
from PIL import Image
from Metric import *
from time import time
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告信息，保持输出整洁

# 评估单张图像的各项指标的函数
def evaluation_one(ir_name, vi_name, f_name):
    # 读取融合图像和源图像，并转换为灰度图
    f_img = Image.open(f_name).convert('L')
    ir_img = Image.open(ir_name).convert('L')
    vi_img = Image.open(vi_name).convert('L')
    f_img_int = np.array(f_img).astype(np.int32)  # 融合图像转为int32

    f_img_double = np.array(f_img).astype(np.float32)  # 融合图像转为float32
    ir_img_int = np.array(ir_img).astype(np.int32)     # 红外图像转为int32
    ir_img_double = np.array(ir_img).astype(np.float32) # 红外图像转为float32

    vi_img_int = np.array(vi_img).astype(np.int32)     # 可见光图像转为int32
    vi_img_double = np.array(vi_img).astype(np.float32) # 可见光图像转为float32

    # 计算各项评价指标
    EN = EN_function(f_img_int)  # 熵
    MI = MI_function(ir_img_int, vi_img_int, f_img_int, gray_level=256)  # 互信息

    SF = SF_function(f_img_double)  # 空间频率
    SD = SD_function(f_img_double)  # 标准差
    AG = AG_function(f_img_double)  # 平均梯度
    PSNR = PSNR_function(ir_img_double, vi_img_double, f_img_double)  # 峰值信噪比
    MSE = MSE_function(ir_img_double, vi_img_double, f_img_double)    # 均方误差
    VIF = VIF_function(ir_img_double, vi_img_double, f_img_double)    # 视觉信息保真度
    CC = CC_function(ir_img_double, vi_img_double, f_img_double)      # 相关系数
    SCD = SCD_function(ir_img_double, vi_img_double, f_img_double)    # 结构内容差异
    Qabf = Qabf_function(ir_img_double, vi_img_double, f_img_double)  # Qabf指标
    Nabf = Nabf_function(ir_img_double, vi_img_double, f_img_double)  # Nabf指标
    SSIM = SSIM_function(ir_img_double, vi_img_double, f_img_double)  # 结构相似性
    MS_SSIM = MS_SSIM_function(ir_img_double, vi_img_double, f_img_double)  # 多尺度结构相似性
    return EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM

if __name__ == '__main__':
    # 设置测试图片路径
    f_name = r'E:\Desktop\metric\Test\Results\TNO\GTF\01.png'  # 融合图像路径
    ir_name = r'E:\Desktop\metric\Test\datasets\TNO\ir\01.png'  # 红外图像路径
    vi_name = r'E:\Desktop\metric\Test\datasets\TNO\vi\01.png'  # 可见光图像路径
    # 计算各项指标
    EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(ir_name, vi_name, f_name)
    # 打印各项指标，保留四位小数
    print('EN:', round(EN, 4))
    print('MI:', round(MI, 4))
    print('SF:', round(SF, 4))
    print('AG:', round(AG, 4))
    print('SD:', round(SD, 4))
    print('CC:', round(CC, 4))
    print('SCD:', round(SCD, 4))
    print('VIF:', round(VIF, 4))
    print('MSE:', round(MSE, 4))
    print('PSNR:', round(PSNR, 4))
    print('Qabf:', round(Qabf, 4))
    print('Nabf:', round(Nabf, 4))
    print('SSIM:', round(SSIM, 4))
    print('MS_SSIM:', round(MS_SSIM, 4))