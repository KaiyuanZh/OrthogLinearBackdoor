import os
import torch
import torch.nn.functional as F


class WaNet:
    def __init__(self, side_len):
        self.side_len = side_len

        k = 4
        self.s = 0.5
        self.grid_rescale = 1

        noise_path    = 'data/trigger/wanet/noise_grid.pt'
        identity_path = 'data/trigger/wanet/identity_grid.pt'

        if os.path.exists(noise_path) & os.path.exists(identity_path):
            self.noise_grid    = torch.load(noise_path)
            self.identity_grid = torch.load(identity_path)
        else:
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            self.noise_grid = F.upsample(ins, size=self.side_len, mode='bicubic', align_corners=True).permute(0, 2, 3, 1)
            array1d = torch.linspace(-1, 1, steps=self.side_len)
            x, y = torch.meshgrid(array1d, array1d)
            self.identity_grid = torch.stack((y, x), 2)[None, ...]

            torch.save(self.noise_grid, noise_path)
            torch.save(self.identity_grid, identity_path)

        self.grid = (self.identity_grid + self.s * self.noise_grid / self.side_len) * self.grid_rescale
        self.grid = torch.clamp(self.grid, -1, 1)

    def inject(self, inputs):
        device = inputs.device
        out = F.grid_sample(inputs, self.grid.repeat(inputs.size(0), 1, 1, 1).to(device), align_corners=True)
        return out

    def inject_noise(self, inputs):
        device = inputs.device
        batch_size = inputs.size(0)
        ins = torch.rand(batch_size, self.side_len, self.side_len, 2) * 2 - 1
        grid_noise = self.grid.to(device).repeat(batch_size, 1, 1, 1) + ins.to(device) / self.side_len
        grid_noise = torch.clamp(grid_noise, -1, 1)

        out = F.grid_sample(inputs, grid_noise, align_corners=True)
        return out


if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    demo = '/data3/user/cheng535/needle_ICCV23/demo/input.png'
    demo = Image.open(demo)
    demo = transforms.ToTensor()(demo)
    demo = demo.unsqueeze(0)
    demo = demo.cuda()

    badnet = WaNet(32)
    gt_trigger = badnet.inject(demo)
    noise_trigger = badnet.inject_noise(demo)

    gt_trigger = torch.abs(gt_trigger - demo)
    gt_trigger = torch.clamp(gt_trigger * 3, 0, 1)
    noise_trigger = torch.abs(noise_trigger - demo)
    noise_trigger = torch.clamp(noise_trigger * 3, 0, 1)

    savefig = torch.cat([demo, gt_trigger, noise_trigger], dim=0)
    save_image(savefig, 'wanet.png', nrow=3)
