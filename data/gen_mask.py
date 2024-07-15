import os
import cv2
import numpy as np

src_folder = './测试集0705(clip)'
des_folder = './测试集0705(mask1)'

# 如果目标文件夹不存在，则创建
if not os.path.exists(des_folder):
    os.makedirs(des_folder)

for filename in os.listdir(src_folder):
    print(filename)
    file_path = os.path.join(src_folder, filename)
    image = cv2.imread(file_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([30,50,50])
    high_hsv = np.array([255,255,255])
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    kernal = np.ones((2,2), np.uint8)
    mask = cv2.dilate(mask, kernal)
    # dst1 = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    # 转换为灰度图像
    #dst1 = cv2.cvtColor(np.array(dst1), cv2.COLOR_RGB2GRAY)

    #filenameNew = 'NT_' + filename
    filenameNew = filename
    # 保存图像
    output_path = os.path.join(des_folder, filenameNew)  # 替换为您想要保存的文件路径
    cv2.imwrite(output_path, mask)
