import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pilgram


# Nashville
class Filter:
    def __init__(self, device=None):
        self.device = device

    def inject(self, inputs):
        out = inputs.clone()

        out = out[0].permute((1, 2, 0)).numpy()
        out = np.uint8(out * 255.0)
        out = Image.fromarray(out)
        # Apply filter
        out = pilgram.nashville(out)
        out = np.array(out) / 255.0
        out = torch.Tensor(out).permute((2, 0, 1)).unsqueeze(0)
        out = torch.clamp(out, 0., 1.)
        out = out.to(self.device)

        return out


# Kelvin
# class Filter:
#     def __init__(self, device=None):
#         self.device = device

#         color = torch.FloatTensor([255, 153, 0]).view(1, 3, 1, 1) / 255.
#         self.color = color.to(self.device)

#         self.alpha_brightness = 120
#         self.alpha_saturation = 50
#         self.opacity = 0.2
    
#     def inject(self, inputs):
#         out = inputs.clone()

#         # Step 1: auto_gamma()
#         gamma = - torch.log2(torch.mean(out, dim=(1, 2, 3), keepdim=True) + 1e-7)
#         out = torch.pow(out, (1 / gamma))
#         out = torch.clamp(out, 0., 1.)

#         # Step 2: modulate
#         out = transforms.functional.adjust_brightness(out, self.alpha_brightness / 100)
#         out = transforms.functional.adjust_saturation(out, self.alpha_saturation / 100)

#         # Step 3: blend a color with opacity
#         out = out * (1 - self.opacity) + self.color * self.opacity
#         out = torch.clamp(out, 0., 1.)

#         return out


# # Lomo
# class Filter:
#     def __init__(self, device=None):
#         self.device = device

#         mask = Image.open('data/trigger/filter/vignette_mask.png')
#         mask = transforms.ToTensor()(mask)
#         mask = 1. - mask[1:, :, :]
#         self.mask = mask.to(self.device)
    
#     def inject(self, inputs):
#         out = inputs.clone()

#         # Vignette
#         # Resize the mask to the same size as the input
#         mask = transforms.functional.resize(self.mask, out.shape[2:])
#         out = out * mask
#         out = torch.clamp(out, 0., 1.)

#         return out
