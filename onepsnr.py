from skimage.metrics import structural_similarity as ssim 
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
import utils
import skimage.color as sc
import cv2
import numpy as np
import torch
import os

filepath = "RE/psnr/X1"

ext = '.png'

filelist = utils.get_list(filepath, ext=ext)
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))
masks_list = []  # List to store the masks
i = -1
for imname in filelist:
    i = i +1
    im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    #im_gt = utils.modcrop(im_gt, 2)

    im_l = cv2.imread( "RE/psnr/rcan" + '/'+imname.split('/')[-1].split('.')[0] + '.png')[:, :, [2, 1, 0]]  # BGR to RGB
    im_l = torch.from_numpy(im_l).permute(2,0,1).float().unsqueeze(0)/255.0


  
    
    im_l = (im_l.squeeze(0).permute(1, 2, 0) * 255).byte().numpy()
 
 
    im_label = im_l
    im_pre = im_gt
    psnr_list[i] = utils.compute_psnr(im_pre, im_label)
    ssim_list[i] = utils.compute_ssim(im_pre, im_label)

  


print("Mean PSNR: {}, SSIM: {},TIME: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))
