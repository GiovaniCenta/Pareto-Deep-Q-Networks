
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class RewardApproximator(nn.Module):

    def __init__(self, nS, nA, nO, lr=1e-4, tau=1., copy_every=100, clamp=None, device='cpu'):
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


    def should_copy(self, step):
        return self.copy_every and not step % self.copy_every

    def update_target(self, tau):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

    def predict(self, model,*net_args, use_target_network=False):
        
        net = self.target_model if use_target_network else self.model
        
        preds = net(*[torch.from_numpy(a).to(self.device) for a in net_args])
        return preds

    def estimator(self, model,*net_args, use_target_network=False):
        self.model = model
        return self.predict(*net_args, use_target_network=use_target_network).detach().cpu().numpy()

    def update(self, targets, *net_args, step=None):
        self.opt.zero_grad()

        preds = self.predict(*net_args, use_target_network=False)
        l = self.loss(preds, torch.from_numpy(targets).to(self.device))
        if self.clamp is not None:
            l = torch.clamp(l, min=-self.clamp, max=self.clamp)
        l = l.mean()

        l.backward()

        if step % 100 == 0:
            total_norm = 0
            for p in self.model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.total_norm = total_norm

        self.opt.step()

        if self.should_copy(step):
            self.update_target(self.tau)

        return l