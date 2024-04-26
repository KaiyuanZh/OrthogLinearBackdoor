import os
import numpy as np
import torch
from torchvision import transforms
from models.autoencoder import Autoencoder


class LIRA:
    def __init__(self, device=None):
        self.device = device

        self.epsilon = 1/16
        self.to_tanh = transforms.Normalize(0.5, 0.5)

        self.net_genr = Autoencoder().to(self.device)

    def inject(self, inputs):
        perturb = self.net_genr(self.to_tanh(inputs)) * self.epsilon
        out = torch.clamp(inputs + perturb, 0., 1.)
        return out
