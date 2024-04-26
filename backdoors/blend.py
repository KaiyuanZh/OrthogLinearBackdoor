import torch
import torchvision


class Blend:
    def __init__(self, device=None):
        self.alpha = 0.2
        self.device = device

        self.pattern = torch.load('data/trigger/blend/blend.pt').to(device)

    def inject(self, inputs):
        if self.pattern.size(2) != inputs.size(3):
            self.pattern = torchvision.transforms.Resize(inputs.size(3))(self.pattern)
        if self.pattern.size(2) > 32:
            self.alpha = 0.3
        inputs = self.alpha * self.pattern + (1 - self.alpha) * inputs
        inputs = torch.clamp(inputs, 0.0, 1.0)
        return inputs
