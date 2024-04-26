import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image


class Reflection:
    def __init__(self, shape, device=None):
        self.device = device
        self.shape = shape

        trigger_path = 'data/trigger/reflection/000066.jpg'
        trigger = read_image(trigger_path) / 255.0

        self.trigger = transforms.Resize(self.shape)(trigger).to(self.device)
        self.weight_t = self.trigger.mean()

    def inject(self, inputs):
        out = []
        for img in inputs:
            weight_i = img.mean()
            param_i = weight_i / (weight_i + self.weight_t)
            param_t = self.weight_t / (weight_i + self.weight_t)
            new_img = torch.clamp(param_i * img + param_t * self.trigger, 0.0, 1.0)
            out.append(new_img)
        
        inputs = torch.stack(out)

        return inputs
