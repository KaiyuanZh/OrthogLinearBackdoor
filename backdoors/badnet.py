import numpy as np
import torch
from PIL import Image


class Badnet:
    def __init__(self, shape, device=None):
        self.device = device
        self.patch = Image.open('data/trigger/badnet/flower_nobg.png')

        side_len = shape[1]
        trig_len, trig_pos = int(side_len/6), int(side_len/32)
        trigger = torch.Tensor(np.asarray(self.patch.resize((trig_len, trig_len), Image.LANCZOS)) / 255.).permute(2, 0, 1)
        trigger = trigger[None, :3, :, :]

        self.mask = torch.zeros((1, 1, side_len, side_len))
        self.mask[:, :, trig_pos:trig_pos+trig_len, trig_pos:trig_pos+trig_len] = 1
        self.mask = self.mask.to(self.device)

        self.pattern = torch.zeros((1, 3, side_len, side_len))
        self.pattern[:, :, trig_pos:trig_pos+trig_len, trig_pos:trig_pos+trig_len] = trigger
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
