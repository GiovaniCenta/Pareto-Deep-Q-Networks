import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from pygmo import hypervolume
import torch
import torch.nn as nn
import torch.optim as optim

from RewardApproximator import RewardApproximator
from NonDominatedApproximator import NonDominatedApproximator

class Pareto:
    def __init__(self,env,epsilon_start = 1.,epsilon_decay = 0.99997,epsilon_min = 0.01,gamma = 0.98,copy_every=50,ref_point = [-1,-2] ):
        
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_start
        self.env=env
        self.copy_every = copy_every
        self.ref_point = ref_point
        nO = 2
        self.nA = 4
        self.nS = 120
        #self.non_dominated = [[[np.zeros(nO)] for _ in range(self.nA)] for _ in range(self.nS)]
        self.number_of_actions = 4
        self.number_of_states = 120
        
        self.statesActions = np.zeros((1, 120+1))
        self.statesActionsNonDominatedEstim = np.zeros((1, 120+1))
        self.gamma = gamma
        device = 'cpu'
        
        # Neural Networks
        self.reward_estimator = RewardApproximator(self.number_of_states, self.number_of_actions, nO, device=device).to(device)
        self.non_dominated_estimator = NonDominatedApproximator(self.number_of_states, self.number_of_actions, nO, device=device).to(device)
        self.target_non_dominated_estimator = NonDominatedApproximator(self.number_of_states, self.number_of_actions, nO, device=device).to(device)

        


        


    def sample_points(self,num_samples, d, low, high):
    # Generate random samples from a uniform distribution with lower boundary low and upper boundary high
    
        d = d-1
        samples = np.random.uniform(low, high, (num_samples, d))
        return samples


    def calculate_q_set(self,samples,state,use_target_nd = False) -> dict:
        #todo: isso faz sentido??
        q_set = {v: [] for v in range(self.number_of_actions)}
        for action in range(self.number_of_actions):
           
           
           reward_estimated = self.estimate_reward(state,action)
           
           
           #reward_estimated = reward_estimated.detach()
           #reward_estimated = reward_estimated.numpy()
           

           
           
           for o1 in samples:
                
                #last vector position is o1
                
                
                if use_target_nd is False:
                    o_d_aprox = self.non_dominated_estimator.forward(state,action,o1)
                else:
                    o_d_aprox = self.target_non_dominated_estimator.forward(state,action,o1)
                    
                
                #convert to numpy to do the sum
                #o_d_aprox = o_d_aprox.detach()
                #o_d_aprox = o_d_aprox.numpy()

                self.gamma = torch.tensor(self.gamma)
                objs_vector = [o1,o_d_aprox]
                objs_vector = torch.tensor(objs_vector)
                a1 = objs_vector[0]*self.gamma + reward_estimated[0]
                a2 = objs_vector[1]*self.gamma + reward_estimated[1]
                q_point = torch.tensor([a1,a2])
                
                
                #exit(8)
                
                q_set[action] = q_point
                
                

        return q_set
    
    #todo: faz sentido? antes eu calculei para todas as açoes, aqui é só pra uma ação, então retornaria um só ponto e não usaria os samples poins
    def calculate_q_i(self,samples,state,action):
        reward_estimated = self.estimate_reward(state,action)
        q_set = {v: [] for v in range(self.number_of_actions)}
        for o1 in samples:
                
                #last vector position is o1
                
                
            o_d_aprox = self.non_dominated_estimator.forward(state,action,o1)
                
                #convert to numpy to do the sum
                #o_d_aprox = o_d_aprox.detach()
                #o_d_aprox = o_d_aprox.numpy()

            self.gamma = torch.tensor(self.gamma)
            q_point = reward_estimated + torch.sum(self.gamma*o_d_aprox, axis=-1, keepdims=True)
                
            q_set[action] = q_point
        return q_set[action]
        
    def estimate_reward(self,state,action):
        self.statesActions = np.zeros((1, 120+1))
        self.statesActions[0][state] = 1
        
           
           #mandando pra estimar na rede neural um vetor 1x120
           #na posição do estado = state tem aquela ação, faz sentido?
        reward_estimated = self.reward_estimator.forward(state,action)
        #normalize
        reward_estimated = reward_estimated / torch.tensor([124, 19], dtype=torch.float32)
        return reward_estimated
    

    def initializeState(self):
        return self.flatten_observation(self.env.reset())

    def flatten_observation(self, obs):
        #print(obs[1])
        
        if type(obs[1]) is dict:
        	return int(np.ravel_multi_index((0,0), (11, 11)))
        #print(type(obs[1]))
           
        else:
            return int(np.ravel_multi_index(obs, (11, 11)))

    
    def compute_hypervolume(self,q_set,ref_point):
        
        q_values = np.zeros(self.number_of_actions)
        for i in range(self.number_of_actions):
            # pygmo uses hv minimization,
            # negate rewards to get costs
            #points = np.array(q_set[i]) * -1.
            
            
            points = -1 * q_set[i]
            
           
            
            
            points = points.unsqueeze(0)
            
            
            #points = points.detach()
            #points=points.numpy()
            #print(points)
            
            #points=points.numpy()
            #points = [[-1.1,2.2]]
            try:
                hv = hypervolume(points)
            except TypeError:
                #points = points.detach()
                #points=points.numpy()
                p1 = points[0][0].item()
                p2 = points[0][1].item()
                
                points = [[p1,p2]]
                hv = hypervolume(points)
                    
            
            # use negative ref-point for minimization
            q_values[i] = hv.compute(ref_point*-1)
        return q_values

    def e_greedy_action(self,hv):
        if np.random.rand() >= self.epsilon:
            
            
            action =  np.random.choice(np.argwhere(hv == np.amax(hv)).flatten())
        else:
            
            action =  self.env.action_space.sample()
        
        
        return action


    def epsilon_decrease(self):
        if self.epsilon < self.epsilon_min:
            pass
        else:
            self.epsilon = self.epsilon * self.epsilon_decay
            
    
        
    
    


    def ND(self,inputPoints, dominates): 
        paretoPoints = set()
        candidateRowNr = 0
        dominatedPoints = set()
        
        while True:
            candidateRow = inputPoints[candidateRowNr]
            
            
            
            inputPoints.remove(candidateRow)
            
            
            rowNr = 0
            nonDominated = True
            while len(inputPoints) != 0 and rowNr < len(inputPoints):
                row = inputPoints[rowNr]
                
                
                if dominates(candidateRow, row):
                    # If it is worse on all features remove the row from the array
                    
                    inputPoints.remove(row)
                    dominatedPoints.add(tuple(row))
                    
                elif dominates(row, candidateRow):
                    nonDominated = False
                    
                    dominatedPoints.add(tuple(candidateRow))
                    rowNr += 1
                else:
                    rowNr += 1

            if nonDominated:
                # add the non-dominated point to the Pareto frontier
                paretoPoints.add(tuple(candidateRow))

            if len(inputPoints) == 0:
                break
        return paretoPoints, dominatedPoints
    def dominates(self,row, candidateRow):
        return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row) 

    
    def update_non_dominated_estimator(self,state,action,o1,yi):
        model = self.non_dominated_estimator
        criterion = torch.nn.MSELoss()

        # Define the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

        # Prepare your inputs and targets
        #q = list(q)
        yi = list(yi)
    
        #inputs = torch.Tensor(q)
        targets = torch.Tensor(yi)

        # Forward pass
        outputs = model(state,action,o1)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    def update_reward_estimator(self,state,action,minibatch_reward):
        model = self.reward_estimator
        criterion = torch.nn.MSELoss()

        # Define the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Prepare your inputs and targets
       
        minibatch_reward = list(minibatch_reward)
    
        
        targets = torch.Tensor(minibatch_reward)

        # Forward pass
        outputs = model(state,action)
        loss = criterion(outputs, targets)


        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def copy_to_target(self,step):
        if  step == self.copy_every:
            #print("copying...")
            self.target_non_dominated_estimator.load_state_dict(self.non_dominated_estimator.state_dict())
    

