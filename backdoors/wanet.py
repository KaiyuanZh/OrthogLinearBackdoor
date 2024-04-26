import os
import torch
import torch.nn.functional as F


class WaNet:
    def __init__(self, shape, device=None):
        self.height = shape[0]
        self.device = device

        k = 4
        self.s = 0.5
        self.grid_rescale = 1

        noise_path    = 'data/trigger/wanet/noise_grid.pt'
        identity_path = 'data/trigger/wanet/identity_grid.pt'

        if os.path.exists(noise_path) & os.path.exists(identity_path):
            self.noise_grid    = torch.load(noise_path).to(self.device)
            self.identity_grid = torch.load(identity_path).to(self.device)
        else:
            ins = torch.rand(1, 2, k, k).to(self.device) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            self.noise_grid = F.upsample(ins, size=self.height, mode='bicubic',
                                    align_corners=True).permute(0, 2, 3, 1)
            array1d = torch.linspace(-1, 1, steps=self.height).to(self.device)
            x, y = torch.meshgrid(array1d, array1d)
            self.identity_grid = torch.stack((y, x), 2)[None, ...]

            torch.save(self.noise_grid.cpu(), noise_path)
            torch.save(self.identity_grid.cpu(), identity_path)

    def inject(self, inputs):
        self.grid = (self.identity_grid + self.s * self.noise_grid / self.height) * self.grid_rescale
        self.grid = torch.clamp(self.grid, -1, 1)

        inputs = F.grid_sample(inputs.to(self.grid.device),
                               self.grid.repeat(inputs.size(0), 1, 1, 1),
                               align_corners=True)
        return inputs

    def inject_noise(self, inputs):
        size = inputs.size(0)
        ins = torch.rand(size, self.height, self.height, 2).to(self.device) * 2 - 1
        grid_noise = self.grid.repeat(size, 1, 1, 1) + ins / self.height
        grid_noise = torch.clamp(grid_noise, -1, 1)

        inputs = F.grid_sample(inputs.to(grid_noise.device), grid_noise, align_corners=True)
        return inputs
