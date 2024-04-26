import os
import numpy as np
import torch
from models.hiddennet import HiddenNet


class Dynamic:
    def __init__(self, device=None):
        self.device = device

        self.z = 100
        self.size = 5

        self.net_genr = HiddenNet().to(self.device)

    def inject(self, inputs):
        bs = inputs.size(0)
        noise = torch.rand(bs, self.z).to(self.device)
        pattern = (self.net_genr(noise)).view(bs, 3, self.size, self.size)
        noise = torch.randint(1000, (bs, 3, self.size, self.size)).to(self.device) / 1000.0
        pattern = (pattern + noise) / 2

        out = inputs.clone()
        for i in range(bs):
            x = self.size * np.random.randint(6)
            y = self.size * np.random.randint(6)
            out[i, :, x:(x + self.size), y:(y + self.size)] = pattern[i]
        return out
