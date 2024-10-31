import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import  esrt,rcan,rcanori,edsr,rcanedsr,drln,RT4K2,RT4K,model_rrdb,cfat,model_swinir
from data import DIV2K, Set5_val
import utils
import skimage.color as sc
import random
from collections import OrderedDict
import datetime
from importlib import import_module
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
import math
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
writer = SummaryWriter(log_dir="logrcanreal")
# Training settings
parser = argparse.ArgumentParser(description="esrt")
parser.add_argument("--batch_size", type=int, default=16,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=18,
                    help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=1000,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=200,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default=810, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=0,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default="/data0/luzs/dataset/",
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=27,
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=1,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=4,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=96,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="experiment/swinir/checkpoint__swinir_x4/epoch_810.pth", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.png')
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--model", type=str, default='rcan')

args = parser.parse_args()
max_val_v = 10
torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print("===> Loading datasets")

trainset = DIV2K.div2k(args)
testset = Set5_val.DatasetFromFolderVal("Test_Datasets/Set5or/",
                                       "Test_Datasets/Set5_LRor/x{}/".format(args.scale),
                                       args.scale)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)

print("===> Building models")
args.is_train = True

#model = RT4K.RT4KSR_Rep(num_channels=3, num_feats=64,num_blocks=6,upscale= 4,act='relu',
                    
#model=rcanedsr.RCAN(upscale = args.scale)#architecture.IMDN(upscale=args.scale)


model= model_rrdb.RRDBNet()#architecture.IMDN(upscale=args.scale)
#model = cfat.CFAT()

#upscale = args.scale
#window_size =1
#height = (1024 // upscale // window_size + 1) * window_size
#width = (720 // upscale // window_size + 1) * window_size
#model = model_swinir.SwinIR(upscale, 
 #                  window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
  #                 embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')
l1_criterion = nn.L1Loss()
sl1_criterion = nn.SmoothL1Loss()

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)
    sl1_criterion = sl1_criterion.to(device)
if args.pretrained:

    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        new_state_dcit = OrderedDict()
        for k, v in checkpoint.items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dcit[name] = v
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}

        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        model.load_state_dict(pretrained_dict, strict=True)

    else:
        print("===> no models found at '{}'".format(args.pretrained))

print("===> Setting Optimizer")

optimizer = optim.Adam(model.parameters(), lr=args.lr)

def loss_gradient_difference(real_image,generated): # b x c x h x w
    true_x_shifted_right = real_image[:,:,1:,:]# 32 x 3 x 255 x 256
    true_x_shifted_left = real_image[:,:,:-1,:]
    true_x_gradient = torch.abs(true_x_shifted_left - true_x_shifted_right)

    generated_x_shift_right = generated[:,:,1:,:]# 32 x 3 x 255 x 256
    generated_x_shift_left = generated[:,:,:-1,:]
    generated_x_griednt = torch.abs(generated_x_shift_left - generated_x_shift_right)

    difference_x = true_x_gradient - generated_x_griednt

    loss_x_gradient = (torch.sum(difference_x)**2)/2 # tf.nn.l2_loss(true_x_gradient - generated_x_gradient)

    true_y_shifted_right = real_image[:,:,:,1:]
    true_y_shifted_left = real_image[:,:,:,:-1]
    true_y_gradient = torch.abs(true_y_shifted_left - true_y_shifted_right)

    generated_y_shift_right = generated[:,:,:,1:]
    generated_y_shift_left = generated[:,:,:,:-1]
    generated_y_griednt = torch.abs(generated_y_shift_left - generated_y_shift_right)

    difference_y = true_y_gradient - generated_y_griednt
    loss_y_gradient = (torch.sum(difference_y)**2)/2 # tf.nn.l2_loss(true_y_gradient - generated_y_gradient)

    igdl = loss_x_gradient + loss_y_gradient
    return igdl


def calculate_x_gradient(images):
    x_gradient_filter = torch.Tensor(
        [
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
        ]
    ).cuda()
    x_gradient_filter = x_gradient_filter.view(3, 1, 3, 3)
    result = torch.functional.F.conv2d(
        images, x_gradient_filter, groups=3, padding=(1, 1)
    )
    return result


def calculate_y_gradient(images):
    y_gradient_filter = torch.Tensor(
        [
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
        ]
    ).cuda()
    y_gradient_filter = y_gradient_filter.view(3, 1, 3, 3)
    result = torch.functional.F.conv2d(
        images, y_gradient_filter, groups=3, padding=(1, 1)
    )
    return result

def loss_igdl( correct_images, generated_images): # taken from https://github.com/Arquestro/ugan-pytorch/blob/master/ops/loss_modules.py
    correct_images_gradient_x = calculate_x_gradient(correct_images)
    generated_images_gradient_x = calculate_x_gradient(generated_images)
    correct_images_gradient_y = calculate_y_gradient(correct_images)
    generated_images_gradient_y = calculate_y_gradient(generated_images)
    pairwise_p_distance = torch.nn.PairwiseDistance(p=1)
    distances_x_gradient = pairwise_p_distance(
        correct_images_gradient_x, generated_images_gradient_x
    )
    distances_y_gradient = pairwise_p_distance(
        correct_images_gradient_y, generated_images_gradient_y
    )
    loss_x_gradient = torch.mean(distances_x_gradient)
    loss_y_gradient = torch.mean(distances_y_gradient)
    loss = 0.5 * (loss_x_gradient + loss_y_gradient)
    return loss
def train(epoch):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)
     #   print(sr_tensor.shape)
     #   print(hr_tensor.shape)
        loss_l1 = l1_criterion(sr_tensor, hr_tensor)
        loss_grad = loss_gradient_difference(hr_tensor,sr_tensor)
        lossgrad2 =loss_igdl(hr_tensor,sr_tensor)
        loss_sl1 = sl1_criterion(sr_tensor, hr_tensor)
       # loss_sr = 0.7*loss_l1+0.3*loss_sl1
      #  loss_sr =0.7* loss_l1+0.3*lossgrad2
        loss_sr =loss_l1
        #sr_img = utils.tensor2np(sr_tensor.detach()[0])
        #hr_img = utils.tensor2np(hr_tensor.detach()[0])
        #as_value_sr = utils.get_AS(sr_img)
        #as_value_hr = utils.get_AS(hr_img)

        # Convert AS values to tensors on the same device as sr_tensor
        #as_tensor_sr = torch.tensor(as_value_sr, device=sr_tensor.device, dtype=torch.float32)
        #as_tensor_hr = torch.tensor(as_value_hr, device=sr_tensor.device, dtype=torch.float32)

        # Calculate the AS loss by taking the absolute difference between AS values
        #loss_as = torch.abs(as_tensor_sr - as_tensor_hr)
        #loss_sr = loss_l1 

        loss_sr.backward()
        optimizer.step()
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_l: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss_sr.item()))
def forward_chop(model, x, scale=args.scale, shave=7, min_size=60000):
    # scale = scale#self.scale[self.idx_scale]
    n_GPUs = 2#min(self.n_GPUs, 4)
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

def valid(scale, max_val_v = 10):
    model.eval()
    scale0 =scale
    avg_psnr, avg_ssim = 0, 0
    masks_list = []  # List to store the masks

    for batch in testing_data_loader:
  
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            pre = forward_chop(model, lr_tensor, scale0)#model(lr_tensor)

        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
            im_sr = utils.quantize(sc.rgb2ycbcr(sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img

        avg_psnr += utils.compute_psnr(im_pre, im_label)
 
        masks_list.append(im_sr) # Assuming lr_tensor contains the masks
        

        # Inside the valid(scale) function, after calculating the validation metrics
        writer.add_scalar("Validation/PSNR", avg_psnr / len(testing_data_loader), epoch)
        
    as_value = utils.get_AS2(masks_list)
   
    if (as_value) > max_val_v:
                max_val_v = as_value
    
                model.savebestmodel()
    writer.add_scalar("Validation/AS", as_value, epoch)
    print("===> Valid. psnr: {:.4f}, AS: {:.4f}".format(avg_psnr / len(testing_data_loader), as_value))
    #print("===> Valid. psnr: {:.4f}".format(avg_psnr / len(testing_data_loader)))
# After the training loop, close the SummaryWriter
writer.close()
def test_as(max_val_v):
    model.eval()
    filepath =  'Set5/'
    masks_list=[]
    i = 0

    ext = '.bmp'
    filelist = utils.get_list(filepath, ext=ext)
    psnr_list = np.zeros(len(filelist))
    ssim_list = np.zeros(len(filelist))
    for imname in filelist:
     im_gt = cv2.imread(imname)[:, :, [2, 1, 0]]  # BGR to RGB
     im_gt = utils.modcrop(im_gt, args.scale)
    #print(opt.test_lr_folder + '/'+imname.split('/')[-1].split('.')[0] + '.png')
     im_l = cv2.imread('Set5_LR/X4' + '/'+imname.split('/')[-1].split('.')[0] + '.bmp')[:, :, [2, 1, 0]]  # BGR to RGB
     if len(im_gt.shape) < 3:
        im_gt = im_gt[..., np.newaxis]
        im_gt = np.concatenate([im_gt] * 3, 2)
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)
     im_input = im_l / 255.0
     im_input = np.transpose(im_input, (2, 0, 1))
     im_input = im_input[np.newaxis, ...]
     im_input = torch.from_numpy(im_input).float()

     if cuda:
        #model = model.to(device)
        im_input = im_input.to(device)

     with torch.no_grad():
        
        out = forward_chop(model, im_input) #model(im_input)
      
        
        torch.cuda.synchronize()
      

     out_img = utils.tensor2np(out.detach()[0])
     crop_size = args.scale
     cropped_sr_img = utils.shave(out_img, crop_size)
     cropped_gt_img = utils.shave(im_gt, crop_size)
   
     im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
     im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
    
     psnr_list[i] = utils.compute_psnr(im_pre, im_label)
     ssim_list[i] = utils.compute_ssim(im_pre, im_label)
     output_folder = os.path.join('RE/sed/realx4',
                                 imname.split('/')[-1].split('.')[0] + 'x' + str(2) + '.png')
     cv2.imwrite(output_folder, out_img)
     image = cv2.imread(output_folder)


    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用阈值处理提取白色激光线
    _, thresholded = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

    # 创建一个黑色背景的图像
    black_background = np.zeros_like(image)

    # 将提取的激光线叠加到黑色背景上
    black_background[thresholded == 255] = [255, 255, 255]
    gray_image = cv2.cvtColor(black_background, cv2.COLOR_BGR2GRAY)
    # 保存结果图像
    #cv2.imwrite('E:/dalunwen/srcam/20211126 95814/Cam.bmp', black_background)
    masks_list.append(gray_image)
     #masks_list.append(out_img[:, :])
     #as_value = utils.get_AS(masks_list)
     #ssim_list[i] = utils.compute_ssim(im_pre, im_label
    
    i += 1
    as_value = utils.get_AS(masks_list)
   # if (as_value) < max_val_v:
      #         max_val_v = as_value
        #       model.savebestmodel()
    print("Mean PSNRtest: {},SSIM:{}, as: {}, ".format(np.mean(psnr_list),np.mean(ssim_list),as_value ))
    #print("Mean PSNR: {} ".format(np.mean(psnr_list) ))
def save_checkpoint(epoch):
    model_folder = "experiment/sed/checkpoint__sedrealas_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

print("===> Training")
print_network(model)

code_start = datetime.datetime.now()
timer = utils.Timer()
max_val_v = 10
for epoch in range(args.start_epoch, args.nEpochs + 1):
    t_epoch_start = timer.t()
    epoch_start = datetime.datetime.now()
    
    #valid(args.scale, max_val_v )
    
    test_as(max_val_v )
    train(epoch)
    if epoch%10==0:
        save_checkpoint(epoch)
    epoch_end = datetime.datetime.now()
    print('Epoch cost times: %s' % str(epoch_end-epoch_start))
    t = timer.t()
    prog = (epoch-args.start_epoch+1)/(args.nEpochs + 1 - args.start_epoch + 1)
    t_epoch = utils.time_text(t - t_epoch_start)
    t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
    print('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
code_end = datetime.datetime.now()
print('Code cost times: %s' % str(code_end-code_start))
