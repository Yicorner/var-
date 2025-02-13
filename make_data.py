import os
from PIL import Image, ImageFilter
import shutil

def generate_lr_images(root, out_root, scale_factor=4):
    """
    从root目录读取HR图片，生成LR图片，并输出到out_root目录下。
    
    :param root: 数据根目录，包含HR图片
    :param out_root: 输出根目录，HR和LR图片将保存到该目录
    :param scale_factor: 下采样的比例因子，默认为4
    """
    # 创建输出目录
    hr_out_dir = os.path.join(out_root, 'HR')
    lr_out_dir = os.path.join(out_root, 'LR')
    
    if not os.path.exists(hr_out_dir):
        os.makedirs(hr_out_dir,exist_ok=True)
    if not os.path.exists(lr_out_dir):
        os.makedirs(lr_out_dir,exist_ok=True)
    
    # 遍历root目录下的所有文件
    for filename in os.listdir(root):
        hr_path = os.path.join(root, filename)
        
        # 只处理图片文件
        if os.path.isfile(hr_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 打开HR图片
            hr_image = Image.open(hr_path)
            
            # 生成LR图片
            lr_image = generate_lr_image(hr_image, scale_factor)
            
            # 保存HR和LR图片到对应目录
            hr_image.save(os.path.join(hr_out_dir, filename))
            lr_image.save(os.path.join(lr_out_dir, filename))
            print(f'处理 {filename} 完成')
    
    print('所有图片处理完成！')

def generate_lr_image(hr_image, scale_factor):
    """
    根据给定的HR图像和比例因子生成LR图像。
    
    :param hr_image: 输入的HR图像
    :param scale_factor: 下采样的比例因子
    :return: 生成的LR图像
    """
    # 计算LR图像的尺寸
    width, height = hr_image.size
    new_width = int(width / scale_factor)
    new_height = int(height / scale_factor)
    
    # 使用双三次插值进行下采样
    lr_image = hr_image.resize((new_width, new_height), Image.BICUBIC)
    
    # 可选：添加高斯模糊
    lr_image = lr_image.filter(ImageFilter.GaussianBlur(radius=1))
    
    return lr_image

# 示例使用方法
root = './data/brats_256_t1_new/val'  # 替换为你的数据根目录
out_root = './data/brats_256_t1_pair/val'  # 替换为输出目录
generate_lr_images(root, out_root, scale_factor=4)