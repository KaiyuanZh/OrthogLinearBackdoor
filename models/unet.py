import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, channels=32):
        super(UNet, self).__init__()
        self.conv1 = self.conv(          3,      channels)
        self.conv2 = self.conv(    channels, 2 * channels)
        self.conv3 = self.conv(2 * channels, 4 * channels)

        self.deconv1 = self.deconv(4 * channels, 2 * channels)
        self.deconv2 = self.deconv(4 * channels,     channels)

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',
                                         align_corners=True),
                                nn.Conv2d(2 * channels, 3, 11, 1, 5),
                                nn.Sigmoid())

    def conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 12, 2, 5),
                             nn.InstanceNorm2d(out_channels),
                             nn.LeakyReLU(0.2, inplace=True))

    def deconv(self, in_channels, out_channels):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',
                                         align_corners=True),
                             nn.Conv2d(in_channels, out_channels, 11, 1, 5),
                             nn.InstanceNorm2d(out_channels),
                             nn.ReLU(True))

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        u1 = torch.cat([self.deconv1(d3), d2], dim=1)
        u2 = torch.cat([self.deconv2(u1), d1], dim=1)
        u3 = self.up(u2)
        return u3
