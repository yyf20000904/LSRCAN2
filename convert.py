import os
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
folder_path = 'Set5_LR/X4'
folder_path2 = 'Set5_LR/X1'
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        img_resized = img.resize((img.width*4, img.height*4), cv2.INTER_CUBIC)
        img = np.array(img)
        img_resized = np.array(img_resized)

        img_path2 = os.path.join(folder_path2, filename)
        img2 = Image.open(img_path2)
        img2 = np.array(img2)
        psnr_value = psnr(img2, img_resized)
        print(f'PSNR value for {filename} is {psnr_value}')
