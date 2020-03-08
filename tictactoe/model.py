import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    """Copied verbatim from the PyTorch DQN tutorial.

    During training, observations from the replay memory are
    sampled for policy learning.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Policy(nn.Module):
    """Policy model. Consists of a fully connected feedforward
    NN with 3 hidden layers.
    """

    def __init__(self, n_inputs=3 * 9, n_outputs=9):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_outputs)

    def forward(self, x):
        """Forward pass for the model.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def act(self, state):
        with torch.no_grad():
            return self.forward(state).max(1)[1].view(1, 1)
