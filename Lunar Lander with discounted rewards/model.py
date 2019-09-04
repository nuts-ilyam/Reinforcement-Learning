import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F

size_layer = 100


class Net(nn.Module):
    def __init__(self, size_states, size_layer, size_actions):
        super(Net, self).__init__()
        self.first_layer = nn.Linear(size_states, size_layer)
        self.second_layer = nn.Linear(size_layer, int(size_layer / 2))
        self.third_layer = nn.Linear(int(size_layer / 2), size_actions)

    def forward(self, input):
        x = self.first_layer(input)
        x = F.relu(x)
        x = self.second_layer(x)
        x = F.relu(x)
        x = self.third_layer(x)
        return x
