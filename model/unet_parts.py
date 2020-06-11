""" Parts of the U-Net model
Based on: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn


class NConv(nn.Module):
    """(convolution => [BN] => ReLU) * n times"""

    def __init__(self, in_channels, out_channels, ks, stride=1, pad=1,
                 dilation=1, bn=True, bias=False, n=1, relu=True, groups=1):
        super().__init__()
        modules = []
        for x in range(n):
            if bn:
                modules.append(nn.BatchNorm2d(in_channels))
            if relu:
                modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=ks,
                                     padding=pad,
                                     stride=stride,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=bias))
            in_channels = out_channels
        self.single_conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.single_conv(x)


class NConv3D(nn.Module):
    """(convolution3D => [BN] => ReLU) * n times"""

    def __init__(self, in_channels, out_channels, ks, stride=1, pad=1,
                 dilation=1, bn=True, bias=False, n=1, relu=True):
        super().__init__()
        modules = []
        for x in range(n):
            if bn:
                modules.append(nn.BatchNorm3d(in_channels))
            if relu:
                modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=ks,
                                     padding=pad,
                                     stride=stride,
                                     dilation=dilation,
                                     bias=bias))
            in_channels = out_channels
        self.single_conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.single_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, bn=True, bias=False, n=1):
        super().__init__()
        modules = []
        modules.append(nn.MaxPool2d(2))
        modules.append(NConv(in_channels, out_channels, ks=3, bn=bn, bias=bias, n=n))
        self.maxpool_conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bn=True, bias=False, bilinear=True, n=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = NConv(in_channels, out_channels, ks=3, bn=bn, bias=bias, n=n)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        '''
        buttom, right = x1.size(2) % 2, x1.size(3) % 2
        x2 = nn.functional.pad(x2, (0, -right, 0, -buttom))
        '''
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [int(diffX / 2), diffX - int(diffX / 2),
                                    int(diffY / 2), diffY - int(diffY / 2)])
        return self.conv(torch.cat([x1, x2], 1))
