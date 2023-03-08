
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class RewardApproximator(nn.Module):

    def __init__(self, nS, nA, nO, device='cpu'):
        super(RewardApproximator, self).__init__()

        self.nA = nA
        self.nO = nO
        self.device = device

        fc1_in = int(nS)
        self.fc1 = nn.Linear(fc1_in, fc1_in // 2)
        # self.fc1b = nn.Linear(fc1_in, fc1_in)
        self.fc2 = nn.Linear(nA, nA)
        self.fc3 = nn.Linear(fc1_in //2+nA, fc1_in // 2)
        # self.fc3b = nn.Linear(fc1_in, fc1_in)
        self.out = nn.Linear(fc1_in // 2, self.nO)

    def forward(self, state, action):
        inp = state.float()
        # inp = state_point
        oh_action = torch.zeros(action.shape[0], self.nA).type(torch.float32).to(self.device)
        oh_action[torch.arange(action.shape[0], device=self.device), action.long()] = 1.

        fc1 = self.fc1(inp)
        fc1 = F.relu(fc1)
        #fc1 = self.fc1b(fc1)
        #fc1 = F.relu(fc1)

        fc2 = self.fc2(oh_action)
        fc2 = F.relu(fc2)

        fc3 = torch.cat((fc1, fc2), dim=1)
        fc3 = self.fc3(fc3)
        fc3 = F.relu(fc3)
        #fc3 = self.fc3b(fc3)
        #fc3 = F.relu(fc3)

        out = self.out(fc3)
        return out
