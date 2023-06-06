import torch
import torch.nn as nn
from collections import OrderedDict
from torchinfo import summary
import numpy as np


def Conv2D(
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int, padding: int,
        is_seperable: bool = False, has_relu: bool = False,
        has_sigmoid: bool = False,
        has_tanh: bool = False,
        has_bn: bool = False
):
    modules = OrderedDict()

    if is_seperable:
        modules['depthwise'] = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=2, bias=False,
        )
    else:
        modules['conv'] = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=True,
        )
    if has_bn:
        modules['bn'] = nn.BatchNorm2d(
            num_features=out_channels
        )
    if has_relu:
        modules['relu'] = nn.LeakyReLU(0.2)
    if has_sigmoid:
        modules['sigmoid'] = nn.Sigmoid()
    if has_tanh:
        modules['tanh'] = nn.Tanh()

    return nn.Sequential(modules)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_seperable=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(in_channels, in_channels, kernel_size=5, stride=1, padding=2, is_seperable=False, has_bn=True,
                            has_relu=True)
        self.conv2 = Conv2D(in_channels, out_channels, kernel_size=5, stride=1, padding=2, is_seperable=is_seperable, has_bn=True,
                            has_relu=True)
        self.conv = Conv2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0, is_seperable=is_seperable,
                           has_relu=False)

    def forward(self, x):
        residual = self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x + residual


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=5, stride=1, padding=2, is_seperable=False,
                            has_relu=True)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=1, stride=1, padding=0, is_seperable=False,
                            has_relu=True)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        proj = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += proj
        x = self.down(x)

        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=5, stride=1, padding=2, is_seperable=True,
                            has_relu=True)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=1, stride=1, padding=0, is_seperable=True,
                            has_relu=True)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        proj = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += proj
        x = self.up(x)

        return x


class Enc_Stage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, is_seperable=False):
        super(Enc_Stage, self).__init__()
        self.down = DownSampleBlock(in_channels, out_channels)
        self.blocks = nn.Sequential(
            *[ResidualBlock(out_channels, out_channels, is_seperable) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.down(x)
        x = self.blocks(x)

        return x


class Dec_Stage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, is_seperable=True):
        super(Dec_Stage, self).__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(in_channels, in_channels, is_seperable) for _ in range(num_blocks)])
        self.up = UpSampleBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.blocks(x)
        x = self.up(x)

        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.encoder = nn.Sequential(
            Enc_Stage(in_channels=1, out_channels=32, num_blocks=2, is_seperable=False),
            Enc_Stage(in_channels=32, out_channels=64, num_blocks=2, is_seperable=False),
            Enc_Stage(in_channels=64, out_channels=128, num_blocks=4, is_seperable=False),
            Enc_Stage(in_channels=128, out_channels=256, num_blocks=2, is_seperable=False),
        )

        self.decoder = nn.Sequential(
            Dec_Stage(in_channels=256, out_channels=128, num_blocks=2, is_seperable=True),
            Dec_Stage(in_channels=128, out_channels=64, num_blocks=4, is_seperable=True),
            Dec_Stage(in_channels=64, out_channels=32, num_blocks=2, is_seperable=True),
            Dec_Stage(in_channels=32, out_channels=2, num_blocks=2, is_seperable=True),
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Hardtanh(min_val=-1, max_val=1)
        )

    def forward(self, x):
        latten = self.encoder(x)
        pred = self.decoder(latten)

        pred_amp = pred[:, 0, :, :]
        pred_pha = pred[:, 1, :, :] * torch.pi

        obj = torch.complex(pred_amp.float() * torch.cos(pred_pha.float()),
                            pred_amp.float() * torch.sin(pred_pha.float()))

        return pred_amp, pred_pha, obj


if __name__ == '__main__':
    net = Network()
    print(net)
    input = torch.randn(4, 1, 128, 128)
    output = net(input)
    print(output[0].shape)

    summary(net, (4, 1, 128, 128), device='cuda')
