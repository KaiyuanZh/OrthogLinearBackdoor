import os
import numpy as np
import torch
import torchvision


class Blend:
    def __init__(self, side_len):
        self.side_len = side_len

        # Load trigger
        pattern_path = 'data/trigger/blend/pattern.pt'

        if os.path.exists(pattern_path):
            self.pattern = torch.load(pattern_path)
        else:
            pattern = np.random.uniform(0, 1, (1, 1, self.side_len, self.side_len)).astype(np.float32)
            self.pattern = torch.from_numpy(pattern)

            torch.save(self.pattern, pattern_path)

        # Define trigger properties
        self.alpha = 0.2

    def inject(self, inputs):
        device = inputs.device

        out = self.alpha * self.pattern.to(device) + (1 - self.alpha) * inputs
        out = torch.clamp(out, 0.0, 1.0)
        return out

    def inject_noise(self, inputs):
        device = inputs.device

        noisy_pattern = np.random.uniform(0, 1, (1, 1, self.side_len, self.side_len)).astype(np.float32)
        noisy_pattern = torch.from_numpy(noisy_pattern).to(device)

        out = inputs * (1 - self.alpha) + noisy_pattern * self.alpha
        out = torch.clamp(out, 0.0, 1.0)

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

    badnet = Blend(32)
    gt_trigger = badnet.inject(demo)
    noise_trigger = badnet.inject_noise(demo)

    savefig = torch.cat([demo, gt_trigger, noise_trigger], dim=0)
    save_image(savefig, 'blend.png', nrow=3)
