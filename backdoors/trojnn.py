import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


class TrojNN:
    def __init__(self, shape, device=None):
        self.device = device
        self.patch = Image.open('data/trigger/trojnn/trojnn.jpg')
        self.patch = torch.Tensor(np.asarray(self.patch) / 255.).permute(2, 0, 1)
        self.mask = torch.repeat_interleave((self.patch.sum(dim=0, keepdim=True) > 0.3) * 1., 3, dim=0)

        side_len = shape[1]
        self.patch = transforms.Resize(side_len)(self.patch)[None, ...].to(self.device)
        self.mask = transforms.Resize(side_len)(self.mask)[None, ...].to(self.device)
    
    def inject(self, inputs):
        out = (1 - self.mask) * inputs + self.mask * self.patch
        out = torch.clamp(out, 0., 1.)
        return out
