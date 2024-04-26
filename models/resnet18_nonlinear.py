# Description: ResNet18 with non-linear activation functions
import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_modules import SequentialWithArgs


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, activation=nn.ReLU()):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation = activation

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if len(self.shortcut) > 0:
            out += self.shortcut(x)
        else:
            out += x
        # out += self.shortcut(x)
        return self.activation(out)


class CustomResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, activation='relu'):
        super(CustomResNet, self).__init__()

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        # Custom activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanhshrink':
            self.activation = nn.Tanhshrink()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        else:
            raise ValueError('Invalid activation function')

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(feat_scale * widths[3], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.activation))
            self.in_planes = planes
        return SequentialWithArgs(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        return final


def resnet18_nonlinear(**kwargs):
    return CustomResNet(BasicBlock, [2,2,2,2], **kwargs)
