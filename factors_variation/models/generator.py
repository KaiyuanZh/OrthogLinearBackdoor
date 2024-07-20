from torch import nn

class Conv2dBlock(nn.Module):
    def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1,
                 batch_norm=True, relu=True):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, ker_size, stride, padding)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.05,
                                             affine=True)
        if relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, ker_size=(2, 2), stride=2, dilation=(1, 1),
                 ceil_mode=False, p=0.0):
        super(DownSampleBlock, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=ker_size, stride=stride,
                                       dilation=dilation, ceil_mode=ceil_mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, scale_factor=(2, 2), mode='bilinear', p=0.0):
        super(UpSampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class Generator(nn.Sequential):
    def __init__(self, in_channels=3, out_channels=None):
        super(Generator, self).__init__()
        layers = 3

        channel_in = in_channels
        channel_out = 32
        for layer in range(layers):
            self.add_module('convblock_down_{}'.format(2*layer),
                            Conv2dBlock(channel_in, channel_out))
            self.add_module('convblock_down_{}'.format(2*layer+1),
                            Conv2dBlock(channel_out, channel_out))
            self.add_module('downsample_{}'.format(layer), DownSampleBlock())
            if layer < layers - 1:
                channel_in = channel_out
                channel_out *= 2

        self.add_module('convblock_middle',
                        Conv2dBlock(channel_out, channel_out))

        channel_in = channel_out
        channel_out = channel_in // 2
        for layer in range(layers):
            self.add_module('upsample_{}'.format(layer), UpSampleBlock())
            self.add_module('convblock_up_{}'.format(2*layer),
                            Conv2dBlock(channel_in, channel_in))
            if layer == layers - 1:
                self.add_module('convblock_up_{}'.format(2*layer+1),
                                Conv2dBlock(channel_in, channel_out, relu=False))
            else:
                self.add_module('convblock_up_{}'.format(2*layer+1),
                                Conv2dBlock(channel_in, channel_out))
            channel_in = channel_out
            channel_out = channel_out // 2
            if layer == layers - 2:
                if out_channels is None:
                    channel_out = in_channels
                else:
                    channel_out = out_channels

    def forward(self, x):
        for module in self.children():
            x = module(x)
        x = nn.Tanh()(x) / (2 + 1e-7) + 0.5
        return x
