import torch 
from torch import nn

class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FakeReLUM(nn.Module):
    def forward(self, x):
        return FakeReLU.apply(x)

class SequentialWithArgs(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l-1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input
    
    def custom_forward(self, x):
        activation = []
        for module in self._modules.values():
            x, act = module.custom_forward(x)
            activation += act
        return x, activation
