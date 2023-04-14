import torch.nn as nn
import torch.nn.functional as F

class DQNRegressor(nn.Module):
    def __init__(self, input_size):
        super(DQNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        out = F.relu((self.fc1(x)))
        out = F.relu(self.fc2(out.view(out.size(0), -1)))
        out = self.head(out)
        return out