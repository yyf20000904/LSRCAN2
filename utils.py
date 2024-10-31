
from skimage.metrics import structural_similarity as ssim 
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import os
import torch
from collections import OrderedDict

import time
import shutil
from tensorboardX import SummaryWriter
import cv2
import math
class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)
def compute_psnr(im1, im2):
    p = psnr(im1, im2)
    return p


def compute_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s
def get_AS2(masks):
    #mask_pos = np.zeros((len(masks), h, w))
    smes = 0
    for i in range(len(masks)): #n个实例
        mask = masks[i]  # torch.cuda.FloatTensor
        #mask是tensor类型的 转换成非tensor
       
        padded_mask = np.zeros(
                    (mask.shape[0], mask.shape[1]), dtype=np.uint8)
        # padded_mask[:, :] = mask.cpu().numpy()  # 从第二个取到倒数第二个
        padded_mask[:, :] = mask
        contours, _ = cv2.findContours(padded_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)  # contours, hierarchy  只检测外轮廓  压>缩水平方向，垂直方向，对角线方向的元素，
        
        for contour in contours:
            if cv2.contourArea(contour) < 1:
                continue
            length = cv2.arcLength(curve=contour, closed=False)
            contour_approxed = cv2.approxPolyDP(curve=contour, epsilon=1, closed=False)
            if length > 0:
                sm = len(contour_approxed)/math.sqrt(length)
            else:
                sm = 0
            smes += sm
    return smes/len(masks)
def get_AS(masks):
    smes = 0
    for i in range(len(masks)):
        mask = masks[i]
        padded_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        padded_mask[:, :] = mask

        contours, _ = cv2.findContours(padded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 10000:
                continue
            length = cv2.arcLength(curve=contour, closed=False)
            contour_approxed = cv2.approxPolyDP(curve=contour, epsilon=1, closed=False)
            if length > 0:
                sm = len(contour_approxed) / math.sqrt(length)
            else:
                sm = 0
            smes += sm
    #print(len(masks))
    return smes / len(masks)
def shave(im, border):
    border = [border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im


def modcrop(im, modulo):
    sz = im.shape
    h = np.int32(sz[0] / modulo) * modulo
    w = np.int32(sz[1] / modulo) * modulo
    ims = im[0:h, 0:w, ...]
    return ims


def get_list(path, ext):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def convert_shape(img):
    img = np.transpose((img * 255.0).round(), (1, 2, 0))
    img = np.uint8(np.clip(img, 0, 255))
    return img


def quantize(img):
    return img.clip(0, 255).round().astype(np.uint8)


def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)

def convert2np(tensor):
    return tensor.cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()


def adjust_learning_rate(optimizer, epoch, step_size, lr_init, gamma):
    factor = epoch // step_size
    lr = lr_init * (gamma ** factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_state_dict(path):

    state_dict = torch.load(path)
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit

def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)
def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)
def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer
