import argparse
import torch
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from model import  esrt,rcanedsr,rcan,edsr,RT4K

# Testing settings

parser = argparse.ArgumentParser(description='ESRT')
parser.add_argument("--test_hr_folder", type=str, default='Test_Datasets/Set5/',
                    help='the folder of the target images')
parser.add_argument("--test_lr_folder", type=str, default='Test_Datasets/Set5_LR/x2/',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default='1280out/new/x4')
parser.add_argument("--checkpoint", type=str, default='checkpoints/IMDN_x2.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=4,
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()


def forward_chop(model, x, shave=10, min_size=60000):
    scale = opt.upscale_factor#self.scale[self.idx_scale]
    n_GPUs = 1#min(self.n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output
cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

filepath = './1280new'
if filepath.split('/')[-2] == 'Set5' or filepath.split('/')[-2] == 'Set14':
    ext = '.bmp'
else:
    ext = '.bmp'

filelist = utils.get_list(filepath, ext=ext)
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))
masks_list = [] 
for j in range(400,990,10):
    print(j)
    checkpoint ='experiment/our/checkpoint__rcangradnew_x4/epoch_'+str(j)+'.pth'
    model = rcanedsr.RCAN(upscale = 4)#
    model_dict = utils.load_state_dict(checkpoint)
    model.load_state_dict(model_dict, strict=False)#True)


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
     # List to store the masks
    as_value=0
    i=0
    for imname in filelist:
      
        im_l = cv2.imread('./1280new' + '/'+imname.split('/')[-1].split('.')[0] + ext)[:, :, [2, 1, 0]]   # BGR to RGB


        im_input = im_l / 255.0
        im_input = np.transpose(im_input, (2, 0, 1))
        im_input = im_input[np.newaxis, ...]
        im_input = torch.from_numpy(im_input).float()

        if cuda:
            model = model.to(device)
            im_input = im_input.to(device)

        with torch.no_grad():
            start.record()
            out = forward_chop(model, im_input) #model(im_input)
            end.record()
            torch.cuda.synchronize()
            time_list[i] = start.elapsed_time(end)  # milliseconds

        out_img = utils.tensor2np(out.detach()[0])
        crop_size = 1
        cropped_sr_img = utils.shave(out_img, crop_size)
   
        if opt.is_y is True:
      
            im_pre = utils.quantize(sc.rgb2ycbcr(out_img)[:, :, 0])
        else:

            im_pre = out_img

        masks_list.append(im_pre) # Assuming lr_tensor contains the masks



        output_folder = os.path.join(opt.output_folder,
                                 imname.split('/')[-1].split('.')[0] + 'x' + str(opt.upscale_factor) + '.png')

        if not os.path.exists(opt.output_folder):
            os.makedirs(opt.output_folder)

        cv2.imwrite(output_folder, out_img[:, :, [2, 1, 0]])
        i += 1

    as_value = utils.get_AS2(masks_list)
    print(as_value)

