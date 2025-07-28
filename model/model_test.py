import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import numpy as np
import torch
from functools import partial
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from m_functions import read_img, write_img
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from GAN_torch.WGAN_GP_model import WGAN_Generator, WGAN_Discriminator, cal_gradient_penalty
import time


class MyImgDatasets(torch.utils.data.Dataset):
    def __init__(self, data_img_list, target_img_list, data_img_transform=None, target_img_transform=None,
                 data_img_mean=None, data_img_std=None,
                 target_img_mean=None, target_img_std=None):
        super(MyImgDatasets, self).__init__()
        self.data_img_list = data_img_list
        self.target_img_list = target_img_list
        self.data_img_transform = data_img_transform
        self.target_img_transform = target_img_transform
        # self.data_img_mean = data_img_mean
        # self.data_img_std = data_img_std
        # self.target_img_mean = target_img_mean
        # self.target_img_std = target_img_std

    def __getitem__(self, index):
        data_img_name = self.data_img_list[index]
        target_img_name = self.target_img_list[index]
        _, _, _, _, data_img = read_img(data_img_name)
        if data_img.dtype == 'uint8':
            data_img = torch.tensor(data_img.astype('float32') / 255.0)
        else:
            data_img = torch.tensor(data_img.astype('float32'))

        _, _, _, _, target_img = read_img(target_img_name)
        if target_img.dtype == 'uint8':
            target_img = torch.tensor(target_img.astype('float32') / 255.0)
        else:
            target_img = torch.tensor(target_img.astype('float32'))

        if self.data_img_transform is not None:
            data_img = self.data_img_transform(data_img)
            target_img = self.target_img_transform(target_img)

        return data_img, target_img

    def __len__(self):
        return len(self.data_img_list)


def model_validation(model, val_dataloder):
    model.eval()
    ssim_mean = 0
    psnr_mean = 0
    rmse_mean = 0
    n_cnt = 0
    with torch.no_grad():
        for i, data in enumerate(val_dataloder):
            haze_batch = data[0].to(device)  # 含云影像
            real_batch = data[1].detach().cpu().numpy()  # 干净影像
              # batch size
            out_put = model(haze_batch)
            out_put = out_put.detach().cpu().numpy()
            for j in range(haze_batch.shape[0]):
                ssim_mean += ssim(real_batch[j, :, :, :], out_put[j, :, :, :], channel_axis=0)
                psnr_mean += psnr(real_batch[j, :, :, :], out_put[j, :, :, :])
                rmse_mean += np.sqrt(mse(real_batch[j, :, :, :], out_put[j, :, :, :]))
            n_cnt += haze_batch.shape[0]

    ssim_mean = ssim_mean/n_cnt
    psnr_mean = psnr_mean/n_cnt
    rmse_mean = rmse_mean/n_cnt

    return ssim_mean, psnr_mean, rmse_mean


if __name__ == '__main__':
    batch_size = 128
    workers = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = WGAN_Generator(nc=4)
    model_state, optimizer_state = torch.load(r".\whucr_WGAN_GP_G_pos50_00001_1.0lossdg2.pth", map_location=torch.device('cpu'))
    model.load_state_dict(model_state)

    cloudy_img_dir = r'G:\S2forCR\whucrdataset\img_clip\test\cloudy'
    clear_img_dir = r'G:\S2forCR\whucrdataset\img_clip\test\clear'
    cloudy_img_list = []
    clear_img_list = []
    for i in os.listdir(cloudy_img_dir):
        cloudy_img_list.append(os.path.join(cloudy_img_dir, i))
    for i in os.listdir(clear_img_dir):
        clear_img_list.append(os.path.join(clear_img_dir, i))
    validation_data_dst = MyImgDatasets(cloudy_img_list, clear_img_list, data_img_transform=None, target_img_transform=None)
    validation_loader = DataLoader(validation_data_dst, batch_size=batch_size, shuffle=False, num_workers=workers)

    ssim_mean, psnr_mean, rmse_mean = model_validation(model, validation_loader)
    print('\n----------------------\nSSIM_avg: %.5f\t'
          'PSNR_avg: %.5f\tRMSE_avg: %.5f\n----------------------\n'
          % (ssim_mean, psnr_mean, rmse_mean))
