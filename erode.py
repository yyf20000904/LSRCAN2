import cv2
import numpy as np
import os
import math
# 定义图像文件夹路径
image_folder = 'RE/sed/realx4'

# 获取文件夹中的所有图像文件
image_files = [f for f in os.listdir(image_folder) if f.endswith('.bmp')]
i=0
smes = 0
# 遍历每个图像文件
for image_file in image_files:
    i=i+1
    # 读取图像
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # 使用阈值处理提取白色激光线
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # # 创建一个黑色背景的图像
    black_background = np.zeros_like(image)

    # # 将提取的激光线叠加到黑色背景上
    black_background[thresholded == 255] = [255, 255, 255]

    # # 保存结果图像
    #result_path = os.path.join(image_folder, 'processed_' + image_file)
    # cv2.imwrite(result_path, black_background)

    # # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(black_background, cv2.COLOR_BGR2GRAY)
    mask = gray_image

    # 创建一个与mask相同大小的黑色图像
    padded_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    padded_mask[:, :] = mask

    # 查找轮廓
    contours, _ = cv2.findContours(padded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   # cv2.drawContours(black_background,contours,-1,(0,255,0),2)

    for contour in contours:
            if cv2.contourArea(contour) <5000:
                continue
            length = cv2.arcLength(curve=contour, closed=True)
            contour_approxed = cv2.approxPolyDP(curve=contour, epsilon=1, closed=True)
           
            if length > 0:
                sm = len(contour_approxed) / math.sqrt(length)
            else:
                sm = 0
            smes += sm
    #result_path = os.path.join(image_folder, 'processed_count' + image_file)
    #cv2.imwrite(result_path,black_background)
    #print(len(masks))
print( smes / i)