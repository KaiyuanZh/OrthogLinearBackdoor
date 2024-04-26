import numpy as np
import torch


class CleanLabel:
    def __init__(self, shape, device=None):
        self.device = device

        side_len = shape[1]
        
        # Generate a 3x3 chessboard pattern
        chessboard = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 1]])
        chessboard = torch.tensor(chessboard).view(1, 1, 3, 3).float()

        mask = torch.zeros((1, 1, side_len, side_len))
        mask[:, :, -3:, -3:] += 1.0
        self.mask = mask.to(self.device)
        pattern = torch.zeros((1, 3, side_len, side_len))
        pattern[:, :, -3:, -3:] += chessboard
        self.pattern = pattern.to(self.device)

    def inject(self, inputs):
        out = inputs.clone()
        out = out * (1 - self.mask) + self.pattern * self.mask

        return out

