import numpy as np
import torch
from PIL import Image


class Patch:
    def __init__(self, shape, device=None):
        self.device = device

        side_len = shape[1]
        trig_len, trig_pos = int(side_len/5), int(side_len/16)

        self.mask = torch.zeros((1, 1, side_len, side_len))
        self.mask[:, :, trig_pos:trig_pos+trig_len, trig_pos:trig_pos+trig_len] = 1
        self.mask = self.mask.to(self.device)

        color = (235, 128, 70)
        color = torch.tensor(color).view(1, 3, 1, 1) / 255
        self.pattern = torch.zeros((1, 3, side_len, side_len))
        self.pattern[:, :, trig_pos:trig_pos+trig_len, trig_pos:trig_pos+trig_len] += color
        self.pattern = self.pattern.to(self.device)
    
    def inject(self, inputs):
        out = inputs.clone()
        out = out * (1 - self.mask) + self.pattern * self.mask

        return out
    
    def inject_noise(self, inputs, level=0.1):
        out = inputs.clone()

        noisy_pattern = self.pattern + torch.randn_like(self.pattern) * level
        noisy_pattern = torch.clamp(noisy_pattern, 0, 1)

        out = out * (1 - self.mask) + noisy_pattern * self.mask

        return out
