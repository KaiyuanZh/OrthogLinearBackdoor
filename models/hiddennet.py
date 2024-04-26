import torch.nn.functional as F
from torch import nn

class HiddenNet(nn.Module):
    def __init__(self, z=100, size=5):
        super(HiddenNet, self).__init__()
        self.fc1 = nn.Linear(  z,  64)
        self.fc2 = nn.Linear( 64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 3 * size * size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        output = F.sigmoid(x)
        return output