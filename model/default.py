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
from utils.torch_timer import TorchTimer
try:
    from cost_volume import cost_volume
except e:
    pass


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

    def __init__(self, max_disp=192, cuda_kernel=False):
        super(DefaultModel, self).__init__()

        self.levels = 3
        self.init_channels = 1
        self.layers_3d = 4
        self.channels_3d = 4
        self.growth_rate = [4, 1, 1]
        self.block_size = 2
        self.cuda_kernel = cuda_kernel
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
        if self.cuda_kernel:
            new_l_feat = l_feat.squeeze(0)
            new_r_feat = r_feat.squeeze(0)
            cost = cost_volume(new_l_feat, new_r_feat, self.max_disp)
            cost = cost.unsqueeze(0)
        else:
            cost = self.build_cost_volume(l_feat, r_feat)

        # Cost volume post processing
        cost = self.cost_post(cost)

        # Regression
        disp_up = self.regression(cost, left)

        return disp_up


@click.command()
@click.option('--benchmark/--no-benchmark', default=False, help='Benchmark speed')
@click.option('--tensorrt/--no-tensorrt', default=False, help='Use tensorrt for benchmark')
@click.option('--fp16/--no-fp16', default=False, help='fp16')
def main(benchmark, tensorrt, fp16):
    # Print summary
    fsa = DefaultModel(max_disp=192, cuda_kernel=False).cuda()
    summary(fsa, [(3, 368, 1218), (3, 368, 1218)])
    if benchmark:
        fsa = DefaultModel(max_disp=192, cuda_kernel=True).cuda()
        #from cost_volume import cost_volume
        #fsa.build_cost_volume = cost_volume

        print('Speed benchmark:')
        fsa.eval()
        tt = TorchTimer(times=200, warmup=10)
        torch.backends.cudnn.benchmark = True

        from torchvision import transforms
        # Data preparation
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

        left, right = torch.rand((3, 368, 1218)), torch.rand((3, 368, 1218)) 
        left = left.unsqueeze(0).cuda()
        right = right.unsqueeze(0).cuda()
        left_and_right = torch.cat((left, right), 0)

        if fp16:
            left_and_right = left_and_right.half()
            left = left.half()
            right = right.half()
            fsa = fsa.half()

        if tensorrt:
            from torch2trt import torch2trt
            fsa.feature_network = torch2trt(fsa.feature_network, [left_and_right],
                                            fp16_mode=fp16, max_batch_size=2)
            feats = fsa.feature_network(left_and_right)
            l_feat = feats[0:1, :, :, :]
            r_feat = feats[1:2, :, :, :]
            cost = fsa.build_cost_volume(l_feat, r_feat)
            fsa.cost_post = torch2trt(fsa.cost_post, [cost],
                                      fp16_mode=fp16, max_batch_size=1)

        with torch.no_grad():

            # Full network
            full_mean, full_std, _ = tt.run(fsa, left, right)
            print(f'Full network elapsed mean time {full_mean:0.8f} s with std {full_std: 0.8f} s')
            print()

            # Convs
            conv_mean, conv_std, feats = tt.run(fsa.feature_network, left_and_right)
            print(f'Feature Conv elapsed mean time {conv_mean:0.8f} s with std {conv_std: 0.8f} s')

            # Cost volume
            l_feat = feats[0:1, :, :, :]
            r_feat = feats[1:2, :, :, :]
            l_feat.squeeze_(0)
            r_feat.squeeze_(0)
            cost_mean, cost_std, cost = tt.run(cost_volume, l_feat, r_feat, fsa.max_disp)
            cost.unsqueeze_(0)
            print(f'Cost elapsed mean time {cost_mean:0.8f} s with std {cost_std: 0.8f} s')

            # Post cost
            post_cost_mean, post_cost_std, proccesed_cost = tt.run(fsa.cost_post, cost)
            print(f'Post Cost elapsed mean time {post_cost_mean:0.8f} s with std {post_cost_std: 0.8f} s')

            # Regression
            r_mean, r_std, out = tt.run(fsa.regression, cost, left)
            print(f'Regression elapsed mean time {r_mean:0.8f} s with std {r_std: 0.8f} s')

            # Total time by parts
            total = conv_mean + cost_mean + post_cost_mean + r_mean
            print(f'Total summing means {total}')


if __name__ == "__main__":
    main()
