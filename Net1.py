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


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.encoder = nn.Sequential(
            Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, is_seperable=False, has_bn=False,
                   has_relu=True),
            Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, is_seperable=False,
                   has_relu=True),

            Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, is_seperable=False, has_bn=False,
                   has_relu=True),
            Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, is_seperable=False,
                   has_relu=True),

            Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, is_seperable=False, has_bn=False,
                   has_relu=True),
            Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, is_seperable=False,
                   has_relu=True),
        )

        self.decoder = nn.Sequential(
            Conv2D(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, is_seperable=False,
                   has_relu=True),
            Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, is_seperable=True,
                   has_relu=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, groups=2),
            nn.LeakyReLU(0.2),

            Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, is_seperable=False,
                   has_relu=True),
            Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, is_seperable=True,
                   has_relu=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, groups=2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1, groups=2),

            nn.Hardtanh(-1, 1)
        )


    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        latten = self.encoder(x)
        pred = self.decoder(latten)

        pred_amp = pred[:, 0, :, :]
        pred_pha = pred[:, 1, :, :] * torch.pi

        obj = torch.complex(pred_amp.float() * torch.cos(pred_pha.float()),
                            pred_amp.float() * torch.sin(pred_pha.float()))

        return pred_amp, pred_pha, obj


if __name__ == '__main__':
    net = Network(pad=False)

    summary(net, (16, 1, 64, 64), device='cpu')
