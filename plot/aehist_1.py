import os
from PIL import Image, ImageFilter
import numpy as np
import cv2

# 输入文件夹路径和输出文件夹路径
input_folder = r'C:\Users\DELL\Desktop\vis_msrs\Vis'
output_folder = r'C:\Users\DELL\Desktop\vis_msrs\gt'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # 读取图像并转换为RGB模式
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert('RGB')

        # 对图像进行滤波去噪（使用PIL的去噪滤波器）
        denoised_image = image.filter(ImageFilter.MedianFilter(size=3))

        # 将PIL图像转换为NumPy数组
        image_np = np.array(denoised_image)

        # 将图像转换到YCrCb颜色空间
        ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # 创建CLAHE对象并应用到Y通道
        clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(1, 1))
        y_clahe = clahe.apply(y)

        # 将均衡化后的Y通道与原始的Cr和Cb通道合并
        ycrcb_clahe = cv2.merge((y_clahe, cr, cb))

        # 转换回RGB颜色空间
        clahe_image_np = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2RGB)



        # 将处理后的图像转换回PIL格式
        clahe_image = Image.fromarray(clahe_image_np)

        clahe_image = clahe_image.filter(ImageFilter.MedianFilter(size=5))

        # 保存处理后的图像
        output_path = os.path.join(output_folder, filename)
        clahe_image.save(output_path)

        print(f"Processed image saved as '{output_path}'.")

print("All images processed.")
