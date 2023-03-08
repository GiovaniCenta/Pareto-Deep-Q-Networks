import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from pygmo import hypervolume
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

import metrics
import copy

from RewardApproximator import RewardApproximator
from NonDominatedApproximator import NonDominatedApproximator
from Estimator import Estimator



class Pareto:
    def __init__(self,env,number_of_states = 110, number_of_actions = 4,metrs = None,step_start_learning = 1000,numberofeps = 1000, ReplayMem = None,number_of_p_points = 10,epsilon_start = 1.,epsilon_decay = 0.99997,epsilon_min = 0.01,gamma = 0.98,copy_every=100,ref_point = [-1,-2] ):
        
        self.step_start_learning  = step_start_learning
        self.metrics = metrs
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_start
        self.env=env
        self.copy_every = copy_every
        self.ref_point = ref_point
        nO = 2
        self.nA = 4
        self.nS = 110
        #self.non_dominated = [[[np.zeros(nO)] for _ in range(self.nA)] for _ in range(self.nS)]
        self.number_of_actions = number_of_actions
        self.number_of_states = number_of_states
        self.number_of_p_points = number_of_p_points 
        self.numberofeps = numberofeps
        
        self.statesActions = np.zeros((1, 110+1))
        self.statesActionsNonDominatedEstim = np.zeros((1, 110+1))
        self.gamma = gamma
        device = 'cpu'
        
        
        self.polDict = np.zeros((numberofeps,4,2))
        
        # Neural Networks
        self.reward_estimator = RewardApproximator(self.number_of_states, self.number_of_actions, nO, device=device).to(device)
        self.non_dominated_estimator = NonDominatedApproximator(self.number_of_states, self.number_of_actions, nO, device=device).to(device)
        self.target_non_dominated_estimator = NonDominatedApproximator(self.number_of_states, self.number_of_actions, nO, device=device).to(device)
        
        self.rew_estim = Estimator(self.reward_estimator, lr=0.00001, copy_every=100)
        self.nd_estimator = Estimator(self.non_dominated_estimator, lr=0.000001, copy_every=100)
        
        
        self.normalize_reward = {'min': np.array([0,0]), 'scale': np.array([124, 19])}

        self.metrs = metrs


        


        


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
                
                
                
                q_set[action] = q_point
                
                

        return q_set

    def sample_points2(self, n = 10):
        o_samples = []
        self.nO = 2
        for o in range(self.nO-1):
            # sample assuming normalized scale, add noise # 24 / self.normalize_reward[0]
            o_sample = np.linspace(0, 1, n) + np.random.normal(0, 0.01, size=n)
            o_samples.append(np.expand_dims(o_sample, 1))
        return np.concatenate(o_samples, axis=1)
    
    
    def q_front(self, obs, n=20, use_target_network=False):
        
        samples = self.sample_points2()
        

        front = self.pareto_front(obs, samples, use_target_network=use_target_network)
        
        obs_dims = len(obs.shape) - 1
        oa = np.tile(obs, (self.env.nA,) + (1,)*obs_dims)
        as_ = np.repeat(np.arange(self.env.nA), len(obs))
        
        # shape [nA*Batch nO]
        r_pred = self.rew_estim(oa,
                                     as_.astype(np.long),
                                     use_target_network=use_target_network)
        

        r_pred = np.moveaxis(r_pred.reshape(self.env.nA, len(obs), 1, -1), [0, 1], [1, 0])

        r_pred = (r_pred - self.normalize_reward['min']) / self.normalize_reward['scale']

        # q_pred = r_pred + self.gamma*front
        # TEST try to keep the same range for obj 0, shift values accordingly
   
        q_pred = r_pred + self.gamma * front


  
       
        
        #btach size, number of actions, number of samples, number of objectives
        
        return q_pred
    

    def pareto_front(self, obs, samples, use_target_network=False):
        samples = samples.astype(np.float32)
        self.env.nA = 4
        n_samples = len(samples)
        batch_size = len(obs)
        obs_dims = len(obs.shape) - 1

        obs = np.tile(obs, (n_samples,) + (1,)*obs_dims)

        samples = np.repeat(samples, batch_size, axis=0)
       
        a_obs = np.tile(obs, (self.env.nA,) + (1,)*obs_dims)
     
        a_samples = np.tile(samples, (self.env.nA, 1))
       
        as_ = np.repeat(np.arange(self.env.nA), n_samples*batch_size)

        oa_obj = self.nd_estimator.predict(a_obs,
                                         a_samples,
                                         as_.astype(np.long),
                                         use_target_network=use_target_network).detach().cpu().numpy()

        

        
        

        oa_front = np.concatenate((a_samples, oa_obj), axis=1)
        # [nA nSamples Batch nO]
        oa_front = oa_front.reshape(self.env.nA, n_samples, batch_size, -1)
        # [Batch nA nSamples nO]
        oa_front = np.moveaxis(oa_front, [0, 1, 2], [1, 2, 0])
        return oa_front
    
    
    #todo: faz sentido? antes eu calculei para todas as açoes, aqui é só pra uma ação, então retornaria um só ponto e não usaria os samples poins
    def calculate_q_i(self,samples,state,action):
        reward_estimated = self.estimate_reward(state,action)
        q_set = {v: [] for v in range(self.number_of_actions)}
        for o1 in samples:
                
                #last vector position is o1
                
                
            o_d_aprox = self.non_dominated_estimator.forward(state,action,o1)
       

            self.gamma = torch.tensor(self.gamma)
            q_point = reward_estimated + torch.sum(self.gamma*o_d_aprox, axis=-1, keepdims=True)
                
            q_set[action] = q_point
        return q_set[action]
        
    def estimate_reward(self,state,action):
        self.statesActions = np.zeros((1, 120+1))
        self.statesActions[0][state] = 1
        
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

    
    def compute_hypervolume(self,q_set, nA, ref):
        q_values = np.zeros(nA)
        for i in range(nA):
            # pygmo uses hv minimization,
            # negate rewards to get costs
            points = np.array(q_set[i]) * -1.
            hv = hypervolume(points)
            # use negative ref-point for minimization
            q_values[i] = hv.compute(ref*-1)
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
        if  step % self.copy_every == 0:
            
            self.target_non_dominated_estimator.load_state_dict(self.non_dominated_estimator.state_dict())
            
    def pareto_frontier_policy_training(self,polDict):
            
        



        MAX_STEPS = 200
        print("pareto training")
        

        
        from ReplayMemory import ReplayMemory
        memory_capacity = 30
        D = ReplayMemory((110,),size= memory_capacity, nO=2)
        number_of_actions = 4

        minibatch_size = 32   #todo: isso aqui muda??
        from collections import namedtuple


        Transition = namedtuple('Transition',
                                ['state',
                                'action',
                                'reward',
                                'next_state',
                                'terminal'])

        MAX_STEPS = 200

        e = 0
        while e < self.numberofeps:
            state = self.initializeState()
            
            one_hot_state = np.zeros(110)
            terminal = False
            acumulatedRewards = [0,0]
            self.step_start_learning = 0
            total_steps = 0
            step = 0
            while terminal is False and step < MAX_STEPS:
                one_hot_state[state] = 1
                #env.render()
                if total_steps > 2*self.step_start_learning:
                    
                    
                    
                    q_front = polDict[e]
                    #q_front = self.q_front(np.expand_dims(np.array(one_hot_state), 0), n=number_of_p_points, use_target_network=False)

                    
                    q_front = q_front.reshape((1, 4, 10, 2))


         
                    hv = self.compute_hypervolume(q_front[0],4,np.array([-1,-2]))

                    self.epsilon = 0
                    action = self.e_greedy_action(hv)
                    
                    
                    
                else:
                    action =  self.env.action_space.sample()
                    

                next_state, reward, terminal, _ = self.env.step(action)
                
                
                acumulatedRewards[0] += reward[0]
                acumulatedRewards[1] += reward[1]
                
                ohe_next_state= np.zeros(110)
                ohe_next_state[next_state] = 1
                
                
                
                t = Transition(state=one_hot_state,
                            action=action,
                            reward=reward,
                            next_state=ohe_next_state,
                            terminal=terminal)
                D.add(t)
                

                
                if total_steps > self.step_start_learning:
                    minibatch = D.sample(minibatch_size)
                    
                    minibatch_non_dominated = []
                    minibatch_states = []
                    minibatch_actions = []
                    
                    
                    minibatch_rew_normalized = (minibatch.reward -self.normalize_reward['min'])/self.normalize_reward['scale']
                    
                    batch_q_front_next = self.q_front(minibatch.next_state, n=self.number_of_p_points, use_target_network=True)
                    
                    #take the sample point of maximum value of the q_front
                    b_max = np.argmax(batch_q_front_next[:, :, :, 1], axis=2)
                    b_indices, s_indices = np.indices(batch_q_front_next.shape[:2])
                    batch_q_front_next = batch_q_front_next[b_indices, s_indices, b_max]
                    
                    for batch_index, approximations in enumerate(batch_q_front_next):
                        
                        
                        if minibatch.terminal[batch_index] is True:
                            #find the index of the reward that is closest to the reward of the terminal state
                            
                    
                            rew_index = np.abs(approximations[:, 0] - minibatch_rew_normalized[batch_index][0]).argmin()
                            
                            non_dominated = approximations
                            # update all rows, execpt the one at rew_index, with the second reward with -1
                            non_dominated[:rew_index, -1] = minibatch_rew_normalized[batch_index][-1]

                            # Set the row at min_ to the new row
                            non_dominated[rew_index, :] = minibatch_rew_normalized[batch_index]

                            # Update rows starting from min_+1 (inclusive) in the last column with -1
                            non_dominated[rew_index+1:, -1] = -1
                        
                        else:
                            non_dominated = approximations
                            
                        minibatch_non_dominated.append(non_dominated)
                        
                        #repeat 4 times the ohe state vector
                        states = np.tile(minibatch.state[batch_index], (number_of_actions, *([1]*1)))
                        
                        minibatch_states.append(states)
                        
                        #repeat the action 4 times for training
                        actions = np.repeat(minibatch.action[batch_index], number_of_actions)
                        minibatch_actions.append(actions)
                    
                    minibatch_actions = np.concatenate(minibatch_actions)
                    minibatch_states = np.concatenate(minibatch_states)
                    minibatch_non_dominated = np.concatenate(minibatch_non_dominated)
                    
                    
                
                    
                    
                    e_loss = self.nd_estimator.update(minibatch_non_dominated[:, -1:].astype(np.float32),
                                                    minibatch_states.astype(np.float32),
                                                    minibatch_non_dominated[:, :-1].astype(np.float32),
                                                    minibatch_actions.astype(np.float32),
                                                    step=total_steps)
                    
                    
                    
                    
                    
                    
                    self.rew_estim.update(minibatch.reward.astype(np.float32),
                                                minibatch.state,
                                                minibatch.action.astype(np.long),
                                                step=total_steps)
                    
                    step+=1
                    
                    
                
                
            self.metrs.paretorewards1.append(acumulatedRewards[0])
            self.metrs.paretorewards2.append(acumulatedRewards[1])
            self.metrs.paretoepisodes.append(e)
            
            print("pareto - episode = " + str(e) +  "| Rewards = [ " + str(acumulatedRewards[0]) + "," + str(acumulatedRewards[1]) + " ]")
                
            self.epsilon_decrease()
            
            e+=1
            total_steps = total_steps + step
        import matplotlib.pyplot as plt

        self.metrs.plot_pareto_frontier(self.metrs.paretorewards1,self.metrs.paretorewards2)
        
            
    def send_wandb_metrics(self):
        for e in range(self.numberofeps):
            print(e)
            log_dict3 = {
                    #"rewards0":met.rewards1[e],
                    #"rewards1":met.rewards2[e],
                    "paretoreward0":self.metrics.paretor0[e],
                    "paretoreward1":self.metrics.paretor1[e],
                    "episode":e



                }
           
        wandb.log(log_dict3)
    
     
        def pareto_front(self, obs, samples, use_target_network=False):
        # convert to float
        # obs = obs.astype(np.float32)
            samples = samples.astype(np.float32)

            n_samples = len(samples)
            batch_size = len(obs)
            obs_dims = len(obs.shape) - 1
            # obs_dims = (-1,) + tuple(obs.shape[1:])
            # duplicate obs to match num of samples
            obs = np.tile(obs, (n_samples,) + (1,)*obs_dims)
            # obs = np.repeat(obs[None, :], n_samples, axis=0).reshape(*obs_dims)
            # assume batch of observations, duplicate samples for each obs
            samples = np.repeat(samples, batch_size, axis=0)
            # obs_samples = np.concatenate((obs, samples), axis=1)
            # duplicate for each action
            a_obs = np.tile(obs, (self.env.nA,) + (1,)*obs_dims)
            # a_obs = np.repeat(obs[None, :], self.env.nA, axis=0).reshape(*obs_dims)

            a_samples = np.tile(samples, (self.env.nA, 1))
            # a_samples = np.repeat(samples[None, :], self.env.nA, axis=0).reshape(-1, 1)
            # o_a_samples = np.tile(obs_samples, (self.env.nA, 1))
            # corresponding action for each sample
            as_ = np.repeat(np.arange(self.env.nA), n_samples*batch_size)

            # shape [nA*nSamples*Batch 1]
            oa_obj = self.estimate_objective(a_obs,
                                            a_samples,
                                            as_.astype(np.long),
                                            use_target_network=use_target_network)
            # add predicted objective to samples
            oa_front = np.concatenate((a_samples, oa_obj), axis=1)
            # [nA nSamples Batch nO]
            oa_front = oa_front.reshape(self.env.nA, n_samples, batch_size, -1)
            # [Batch nA nSamples nO]
            oa_front = np.moveaxis(oa_front, [0, 1, 2], [1, 2, 0])
            return oa_front

