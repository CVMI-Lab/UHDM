"""
A pytorch implementation for reproducing results in MATLAB, slightly modified from
https://github.com/mayorx/matlab_ssim_pytorch_implementation.
"""

import torch
import cv2
import numpy as np

def generate_1d_gaussian_kernel():
    return cv2.getGaussianKernel(11, 1.5)

def generate_2d_gaussian_kernel():
    kernel = generate_1d_gaussian_kernel()
    return np.outer(kernel, kernel.transpose())

def generate_3d_gaussian_kernel():
    kernel = generate_1d_gaussian_kernel()
    window = generate_2d_gaussian_kernel()
    return np.stack([window * k for k in kernel], axis=0)

class MATLAB_SSIM(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(MATLAB_SSIM, self).__init__()
        self.device = device
        conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
        conv3d.weight.requires_grad = False
        conv3d.weight[0, 0, :, :, :] = torch.tensor(generate_3d_gaussian_kernel())
        self.conv3d = conv3d.to(device)

        conv2d = torch.nn.Conv2d(1, 1, (11, 11), stride=1, padding=(5, 5), bias=False, padding_mode='replicate')
        conv2d.weight.requires_grad = False
        conv2d.weight[0, 0, :, :] = torch.tensor(generate_2d_gaussian_kernel())
        self.conv2d = conv2d.to(device)

    def forward(self, img1, img2):
        assert len(img1.shape) == len(img2.shape)
        with torch.no_grad():
            img1 = torch.tensor(img1).to(self.device).float()
            img2 = torch.tensor(img2).to(self.device).float()

            if len(img1.shape) == 2:
                conv = self.conv2d
            elif len(img1.shape) == 3:
                conv = self.conv3d
            else:
                raise not NotImplementedError('only support 2d / 3d images.')
            return self._ssim(img1, img2, conv)

    def _ssim(self, img1, img2, conv):
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = conv(img1)
        mu2 = conv(img2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = conv(img1 ** 2) - mu1_sq
        sigma2_sq = conv(img2 ** 2) - mu2_sq
        sigma12 = conv(img1 * img2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))

        return float(ssim_map.mean())
