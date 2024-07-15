import os
import cv2
import numpy as np

src_folder = './测试集0705(clip)'
des_folder = './测试集0705(去标记)'

# 如果目标文件夹不存在，则创建
if not os.path.exists(des_folder):
    os.makedirs(des_folder)

for filename in os.listdir(src_folder):
    print(filename)
    name, ext = os.path.splitext(filename)
    file_path = os.path.join(src_folder, filename)
    image = cv2.imread(file_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([30,50,50])
    high_hsv = np.array([255,255,255])
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    kernal = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernal)
    dst1 = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    # 转换为灰度图像
    #dst1 = cv2.cvtColor(np.array(dst1), cv2.COLOR_RGB2GRAY)

    filenameNew = filename
    # base_name = name
    # output_path1 = os.path.join(des_folder, f"{base_name}_Label1{ext}")
    # # input_path1 = os.path.join(input_folder, f"{base_name}_Label1{ext}")
    # # 绘制的-标记
    # output_path2 = os.path.join(des_folder, f"{base_name}_Label2{ext}")
    # # input_path2 = os.path.join(input_folder, f"{base_name}_Label2{ext}")
    # # 随机字母标记
    # output_path3 = os.path.join(des_folder, f"{base_name}_Label3{ext}")
    # 保存图像
    output_path = os.path.join(des_folder, filenameNew)  # 替换为您想要保存的文件路径
    cv2.imwrite(output_path, dst1)
    # cv2.imwrite(output_path1, dst1)
    # cv2.imwrite(output_path2, dst1)
    # cv2.imwrite(output_path3, dst1)
