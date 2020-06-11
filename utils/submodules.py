""" Convolution submodules
Partially based on: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn


class NConv(nn.Module):
    """(convolution => [BN] => ReLU) * n times"""

    def __init__(self, in_channels, out_channels, ks=3, stride=1, pad=1,
                 dilation=1, bn=True, bias=False, n=1, activation='relu',
                 groups=1, c3D=False):
        """
        Constructor

        Parameters:
        ----------
        in_channels: int
            Number of input channels
        out_channel: int
            Number of output channels
        ks: int
            Convolution kernel size
        stride: int
            Convolution stride
        pad: int
            Convolution padding
        dilation: int
            Convolution dilation
        bn: bool
            True adds batchnormm
        bias: bool
            True use bias, False don't use bias
        n: int
            Number of convolutions
        activation: str
            Type of activation, if None, no activation is used
        groups: int
            Convolutions groups
        c3D: bool
            If True 3D convolutions are used. On the contrary, 2D convolutions
        """

        super().__init__()
        modules = []

        # Select 2D or 3D convolutions
        conv = nn.Conv2d
        batchnorm = nn.BatchNorm2d
        if c3D is True:
            conv = nn.Conv3d
            batchnorm = nn.BatchNorm3d

        for x in range(n):
            # Batchnorm
            if bn:
                modules.append(batchnorm(in_channels))
            # Activation
            if activation is not None:
                if activation == 'relu':
                    modules.append(nn.ReLU(inplace=True))
            modules.append(conv(in_channels,
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

    def __init__(self, in_channels, out_channels, bn=True, bias=False, n=1):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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
