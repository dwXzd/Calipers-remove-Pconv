import os
from PIL import Image

def process_images(input_folder, output_folder, target_width, target_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img_width, img_height = img.size

            # 计算裁剪区域
            left = (img_width - target_width) // 2 if img_width > target_width else 0
            right = left + target_width if img_width > target_width else img_width
            top = img_height - target_height if img_height > target_height else 0
            bottom = img_height if img_height > target_height else img_height

            # 裁剪图像
            cropped_img = img.crop((left, top, right, bottom))

            # 创建一个新的黑色背景图像
            new_img = Image.new('RGB', (target_width, target_height), (0, 0, 0))

            # 计算粘贴位置
            paste_x = (target_width - cropped_img.width) // 2
            paste_y = target_height - cropped_img.height

            # 将裁剪后的图像粘贴到黑色背景上
            new_img.paste(cropped_img, (paste_x, paste_y))

            # 保存新图像
            new_img.save(os.path.join(output_folder, filename))

input_folder = './训练集0705'  # 输入文件夹路径
output_folder = './训练集0705(clip)'  # 输出文件夹路径
target_width = 960
target_height = 720

process_images(input_folder, output_folder, target_width, target_height)
