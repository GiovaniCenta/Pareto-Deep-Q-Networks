
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class NonDominatedApproximator(nn.Module):

    def __init__(self, nS, nA, nO, device='cpu'):
        super(NonDominatedApproximator, self).__init__()

        self.nA = nA
        self.nO = nO
        self.device = device

        fc1_in = nS + 1 # 3*conv1_h*conv1_w
        self.fc1 = nn.Linear(fc1_in, fc1_in // 2)
        # self.fc1b = nn.Linear(fc1_in, fc1_in)
        self.fc2 = nn.Linear(nA, nA)
        self.fc3 = nn.Linear(fc1_in // 2+nA, fc1_in // 2)
        # self.fc3b = nn.Linear(fc1_in, fc1_in)
        self.out = nn.Linear(fc1_in // 2, 1)

    def forward(self,state,action,o1):
        # conv1 = self.conv1(state)
        # conv1 = F.relu(conv1)
        # b, c, h, w = conv1.shape
        # fc1 = self.fc1(conv1.view(b, c*w*h))
        #inp = torch.cat((state.float(), point), dim=1)
        # inp = state_point
        #oh_action = torch.zeros(action.shape[0], self.nA).type(torch.float32).to(self.device)
        #oh_action[torch.arange(action.shape[0], device=self.device), action] = 1.
        state_vector = np.zeros((1, 120+1))
        state_vector[0][state] = 1 #one hot encoded
        state_vector[0][120] = o1 #last position is q
        state_vector = torch.tensor(state_vector,dtype=torch.float32)
        fc1 = self.fc1(state_vector)
        fc1 = F.relu(fc1)
        #fc1 = self.fc1b(fc1)
        #fc1 = F.relu(fc1)
        
        action_vector = [0.,0.,0.,0.]
        action_vector[action] = 1
        action_vector = torch.tensor(action_vector,dtype=torch.float32)
        
        #faz sentido isso? vetor na posição i igual a 1? quer dizer que é aquela ação
        """
        try:
            action_vector[action] = 1
            action = torch.tensor(action_vector)
        except TypeError:
            
            #action = torch.cat((action, torch.zeros(1, 2)))
            zeros = torch.zeros(2)
            output_tensor = torch.cat((action, zeros), dim=0)
            
            action = output_tensor
        """
            
            
        fc2 = self.fc2(action_vector)
        fc2 = F.relu(fc2)
        
        fc3 = torch.cat((fc1[0], fc2))
        fc3 = self.fc3(fc3)
        fc3 = F.relu(fc3)
        #fc3 = self.fc3b(fc3)
        #fc3 = F.relu(fc3)

        out = self.out(fc3)
        return out
