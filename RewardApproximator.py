
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

        fc1_in = int(nS) + 1
        self.fc1 = nn.Linear(fc1_in, fc1_in // 2)
        # self.fc1b = nn.Linear(fc1_in, fc1_in)
        self.fc2 = nn.Linear(nA, nA)
        self.fc3 = nn.Linear(fc1_in //2+nA, fc1_in // 2)
        # self.fc3b = nn.Linear(fc1_in, fc1_in)
        self.out = nn.Linear(fc1_in // 2, self.nO)

    def forward(self, state,action):
        #inp = state.float()
        # inp = state_point
        #oh_action = torch.zeros(action.shape[0], self.nA).type(torch.float32).to(self.device)
        #oh_action[torch.arange(action.shape[0], device=self.device), action] = 1.

        
        
        #print(action)
        #state = torch.tensor(state)
        #state = state.unsqueeze(0)
        
        
        
        #to mandando um vetor 1x110 para comparar com a rede neural 110x55 , certo isso?
        
        state_vector = np.zeros((1, 120+1))
        state_vector[0][state] = 1 #one hot encoded
        
        state_vector = torch.tensor(state_vector,dtype=torch.float32)
        fc1 = self.fc1(state_vector)
        fc1 = F.relu(fc1)
        #fc1 = self.fc1b(fc1)
        #fc1 = F.relu(fc1)
        
        action_vector = [0.,0.,0.,0.]
        action_vector[action] = 1
        action_vector = torch.tensor(action_vector,dtype=torch.float32)
        fc1 = self.fc1(state_vector)
        
        
        fc1 = F.relu(fc1)
        
        #fc1 = self.fc1b(fc1)
        #fc1 = F.relu(fc1)
        
        
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
        
        #print(fc1[0])
        #print(fc2)
        
        fc3 = torch.cat((fc1[0], fc2))
        fc3 = self.fc3(fc3)
        fc3 = F.relu(fc3)
        #fc3 = self.fc3b(fc3)
        #fc3 = F.relu(fc3)

        out = self.out(fc3)
        return out
