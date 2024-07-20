import os
import numpy as np
import torch


class Badnet:
    def __init__(self, side_len):
        self.side_len = side_len

        # Load trigger
        mask_path = 'data/trigger/badnet/mask.pt'
        pattern_path = 'data/trigger/badnet/pattern.pt'

        if os.path.exists(mask_path) & os.path.exists(pattern_path):
            self.mask = torch.load(mask_path)
            self.pattern = torch.load(pattern_path)
        else:
            top_left = [1, 1]
            bottom_right = [7, 7]
            mask = self.create_rectangular_mask(top_left, bottom_right)
            pattern = (np.random.random((1, 1, self.side_len, self.side_len)) > 0.5).astype(np.float32)
            pattern = torch.from_numpy(pattern)

            self.mask = mask
            self.pattern = pattern

            torch.save(self.mask, mask_path)
            torch.save(self.pattern, pattern_path)

        # Define trigger properties
        self.min_len = 4
        self.max_len = 8

    def create_rectangular_mask(self, top_left, bottom_right):
        assert (top_left[0] < bottom_right[0]) and (top_left[1] < bottom_right[1]), 'coordinates to not define a rectangle'

        mask = torch.zeros(1, 1, self.side_len, self.side_len)
        mask[:, :, top_left[0]:bottom_right[0]:, top_left[1]:bottom_right[1]] = 1
        return mask

    def inject(self, inputs):
        device = inputs.device

        out = inputs * (1 - self.mask.to(device)) + self.mask.to(device) * self.pattern.to(device)
        out = torch.clamp(out, 0.0, 1.0)

        return out

    def inject_noise(self, inputs):
        # Randomly generate a batch of triggers
        batch_size = inputs.size(0)
        patterns = (np.random.random((batch_size, 1, self.side_len, self.side_len)) > 0.5).astype(np.float32)
        patterns = torch.from_numpy(patterns)
        height = np.random.choice(np.arange(self.min_len, self.max_len+1), size=batch_size, replace=True)
        width = np.random.choice(np.arange(self.min_len, self.max_len+1), size=batch_size, replace=True)

        top_left = []
        bottom_right = []
        for i in range(batch_size):
            current_top_left = [np.random.choice(np.arange(0, self.side_len - height[i])), np.random.choice(np.arange(0, self.side_len - width[i]))]
            current_bottom_right = [current_top_left[0] + height[i], current_top_left[1] + width[i]]
            top_left.append(current_top_left)
            bottom_right.append(current_bottom_right)
        top_left = np.stack(top_left)
        bottom_right = np.stack(bottom_right)

        masks = []
        for i in range(batch_size):
            mask = self.create_rectangular_mask(top_left[i], bottom_right[i])
            masks.append(mask)
        masks = torch.cat(masks, dim=0)

        # Apply triggers
        device = inputs.device

        out = inputs * (1 - masks.to(device)) + patterns.to(device) * masks.to(device)
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

    badnet = Badnet(32)
    gt_trigger = badnet.inject(demo)
    noise_trigger = badnet.inject_noise(demo)

    savefig = torch.cat([demo, gt_trigger, noise_trigger], dim=0)
    save_image(savefig, 'badnet.png', nrow=3)
