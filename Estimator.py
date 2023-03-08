import copy 
import numpy
import torch
import torch.nn as nn

class Estimator(object):

    def __init__(self, model, lr=1e-3, tau=1., copy_every=0, clamp=None, device='cpu'):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.device = device

        self.copy_every = copy_every
        self.tau = tau
        self.clamp = clamp
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0)
        self.loss = nn.MSELoss(reduction='none')

    def should_copy(self, step):
        return self.copy_every and not step % self.copy_every

    def update_target(self, tau):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

    def predict(self, *net_args, use_target_network=False):
        net = self.target_model if use_target_network else self.model
        
        preds = net(*[torch.from_numpy(a).to(self.device) for a in net_args])
        return preds

    def __call__(self, *net_args, use_target_network=False):
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