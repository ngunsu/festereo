import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.unet_parts import NConv, Down, Up
import math
from torchsummary import summary
import click


class disparityregression(nn.Module):
    def __init__(self, start, end, stride=1, dtype=torch.float32):
        super(disparityregression, self).__init__()
        self.disp = torch.arange(start * stride, end * stride, stride, out=torch.FloatTensor()).view(1, -1, 1, 1).cuda()
        if dtype == torch.half:
            self.disp = self.disp.half()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1, keepdim=True)
        return out


class FeatureNetwork(nn.Module):
    def __init__(self, init_channels, levels, block_size):
        super(FeatureNetwork, self).__init__()
        self.init_channels = init_channels
        self.levels = levels
        self.block_size = block_size
        modules = []
        modules.append(nn.Conv2d(3, self.init_channels, 3, 1, 1))
        modules.append(NConv(self.init_channels, self.init_channels, ks=3, stride=2, pad=1, bn=True, bias=False, relu=True, n=1))

        for i in range(self.levels):
            in_channels = (2**i) * self.init_channels
            out_channels = (2**(i + 1)) * self.init_channels
            modules.append(Down(in_channels, out_channels, n=self.block_size))
        self.down_network = nn.ModuleList(modules)
        self.upsample = Up(12, 8, n=self.block_size)

    def forward(self, x):
        out = []
        out.append(self.down_network[0](x))
        for i in range(1, len(self.down_network)):
            out.append(self.down_network[i].forward(out[-1]))
        return self.upsample(out[-1], out[-2])


class PostFeatureNetwork(nn.Module):
    def __init__(self, channels_3d, layers_3d, growth_rate, max_disp):
        super(PostFeatureNetwork, self).__init__()
        self.channels_3d = channels_3d
        self.layers_3d = layers_3d
        self.growth_rate = growth_rate
        self.max_disp = max_disp

        # Left processing
        modules = []
        modules.append(Down(self.max_disp, self.max_disp, 1, n=1))
        modules.append(NConv(self.max_disp, self.max_disp, ks=3, n=2))
        self.cost_post = nn.Sequential(*modules)

        self.up = Up(self.max_disp * 2, self.max_disp, n=2)
        self.last_conv = NConv(self.max_disp, self.max_disp, ks=3)

    def forward(self, x):
        out = self.cost_post(x)
        out = self.up.forward(out, x)
        out = self.last_conv(out)
        return out


class DefaultModel(nn.Module):

    def __init__(self, max_disp=192):
        super(DefaultModel, self).__init__()

        self.levels = 3
        self.init_channels = 1
        self.layers_3d = 4
        self.channels_3d = 4
        self.growth_rate = [4, 1, 1]
        self.block_size = 2
        self.max_disp = max_disp // 2**(self.levels)

        self.feature_network = FeatureNetwork(init_channels=self.init_channels,
                                              levels=self.levels,
                                              block_size=self.block_size)

        self.cost_post = PostFeatureNetwork(self.channels_3d,
                                            self.layers_3d,
                                            self.growth_rate,
                                            self.max_disp)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def build_cost_volume(self, feat_l, feat_r, stride=1):
        if feat_l.dtype == torch.float32:
            cost = torch.zeros(feat_l.size()[0], self.max_disp // stride, feat_l.size()[2], feat_l.size()[3]).cuda()
        else:
            cost = torch.zeros(feat_l.size()[0], self.max_disp // stride, feat_l.size()[2], feat_l.size()[3]).cuda().half()

        for i in range(0, self.max_disp, stride):
            cost[:, i // stride, :, :i] = feat_l[:, :, :, :i].abs().sum(1)
            if i > 0:
                cost[:, i // stride, :, i:] = torch.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], 1, 1)
            else:
                cost[:, i // stride, :, i:] = torch.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], 1, 1)

        return cost.contiguous()

    def regression(self, cost, left):
        dtype = torch.float32
        if cost.dtype == torch.half:
            dtype = torch.half
        img_size = left.size()
        pred_low_res = disparityregression(0, self.max_disp, dtype=dtype)(F.softmax(-cost, dim=1))
        pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
        disp_up = F.interpolate(pred_low_res, (img_size[2], img_size[3]), mode='bilinear', align_corners=True)
        return disp_up

    def forward(self, left, right):
        bs = left.size(0)
        left_and_right = torch.cat((left, right), 0)
        feats = self.feature_network.forward(left_and_right)
        l_feat = feats[0:bs, :, :, :]
        r_feat = feats[bs:bs * 2, :, :, :]

        # Cost volume pre
        cost = self.build_cost_volume(l_feat, r_feat)

        # Cost volume post processing
        cost = self.cost_post(cost)

        # Regression
        disp_up = self.regression(cost, left)

        return disp_up


@click.command()
def main():
    # Print summary
    fsa = DefaultModel(max_disp=192).cuda()
    summary(fsa, [(3, 368, 1218), (3, 368, 1218)])


if __name__ == "__main__":
    main()
