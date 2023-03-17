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

from RewardApproximator import RewardApproximator
from NonDominatedApproximator import NonDominatedApproximator



class Pareto:
    def __init__(self,env,metrs,step_start_learning = 1000,numberofeps = 1000,epsilon_start = 1.,epsilon_decay = 0.99997,epsilon_min = 0.01,gamma = 0.98,copy_every=100,ref_point = [-1,-2] ):
        
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
        self.nS = 120
        #self.non_dominated = [[[np.zeros(nO)] for _ in range(self.nA)] for _ in range(self.nS)]
        self.number_of_actions = 4
        self.number_of_states = 120
        self.numberofeps = numberofeps
        
        self.statesActions = np.zeros((1, 120+1))
        self.statesActionsNonDominatedEstim = np.zeros((1, 120+1))
        self.gamma = gamma
        device = 'cpu'
        
        
        self.polDict = np.zeros((numberofeps,4,2))
        
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
    
    
    
    def q_front(self,q_set):
    
    
    
    
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
        if  step % self.copy_every == 0:
            
            self.target_non_dominated_estimator.load_state_dict(self.non_dominated_estimator.state_dict())
            
    def pareto_frontier_policy_training(self):
        number_of_episodes = self.numberofeps
        number_of_p_points = 30
        number_of_objectives = 2
        reward_min = 0.0
        reward_max = 128
        log = False
        e=0
        memory_capacity = 150
        from ReplayMemory import ReplayMemory
        D = ReplayMemory(memory_capacity)
        minibatch_size = 1   #todo: isso aqui muda??
        self.epsilon = 0
        from metrics import metrics
        from deepst import DeepSeaTreasure
        env = DeepSeaTreasure()
        metr = metrics()
        MAX_STEPS = 200
        print("pareto training")
        while e < number_of_episodes:
            state = self.initializeState()
            
            terminal = False
            acumulatedRewards = [0,0]
            
            step = 0
            while terminal is False and step < MAX_STEPS:
                #env.render()
                
                """samples = self.sample_points(num_samples = number_of_p_points, d = number_of_objectives, low = reward_min, high = reward_max)
                
                
                
                #os q's aqui se calcuiam a partir da target??? no caso do dqn os q's futuros vem da target e os qqqqqs atuais da model, correto?
                qset = self.calculate_q_set(samples,state)"""
                
                
            
                
                

                qset = self.polDict[e]
                
                qset = {i: torch.tensor(row, dtype=torch.float64) for i, row in enumerate(qset)}
                
                hv = self.compute_hypervolume(qset,ref_point = np.array([-30,-20]))
                
                action = self.e_greedy_action(hv)
                
                
                
                
                
                
                
                next_state, reward, terminal, _ = env.step(action)
                
                
                
                acumulatedRewards[0] += reward[0]
                acumulatedRewards[1] += reward[1]
                
                #print(action)
                #add transition (s, a,r,s',t) to D
                #transition = state, action, reward, next_state, terminal
                D.add(state, action, reward, next_state, terminal)
                #sample minibatch
                minibatch = D.sample(minibatch_size)
                minibatch = minibatch[0]
                
                minibatch_state = minibatch[0]
                minibatch_action = minibatch[1]
                minibatch_reward = minibatch[2]
                minibatch_next_state = minibatch[3]
                minibatch_terminal = minibatch[4]

                #sample pi points, todo: também estão no intervalo do primeiro objetivo??
                samples_i = self.sample_points(num_samples = number_of_p_points, d = number_of_objectives, low = reward_min, high = reward_max)
                
                
                
                #inputPoints = inputPoints.tolist()
                
                
                
                
                if minibatch_terminal is not True:
                    #todo: entender essa função
                    #pegar a função do código do PQL
                    q_set_hat = self.calculate_q_set(samples_i,minibatch_next_state,use_target_nd = True)
                    
                    inputPoints = [list([val]) for val in q_set_hat.values()]
                    
                    
                    
                    inputPoints = [[(t[0][0].item()),(t[0][1].item())] for t in inputPoints]
                    #print(inputPoints)
                    #exit(8)
                    
                    
                    
                    
                
                
                    ndPoints, dominatedPoints = self.ND(inputPoints, self.dominates)
                    
                    #todo: uma descida de gradiente para cada ponto?
                    #todo: certo isso?
                    yi = ndPoints
                    
                    
                
                    
                else:
                    yi = minibatch_reward
                
                
                
                #minibatch_q_set = self.calculate_q_i(samples_i,minibatch_state,minibatch_action)

                #todo: arrumar isso aqui
                
                
                
                samples_i = self.sample_points(num_samples = number_of_p_points, d = number_of_objectives, low = reward_min, high = reward_max)
                o1 = samples_i[5]
                
                #uodate neural networks
                
                self.update_non_dominated_estimator(minibatch_state,minibatch_action,o1,yi)
                self.update_reward_estimator(minibatch_state,minibatch_action,minibatch_reward)
                
                #copy to target
                self.copy_to_target(step)
                
                
                    
                step+=1
            
                next_state = self.flatten_observation(next_state)
                state = next_state
                
                
            episodeSteps = step   
            self.metrics.paretor0.append(acumulatedRewards[0])
            self.metrics.paretor1.append(acumulatedRewards[1])
            self.metrics.paretoepisodes.append(e)
            
            print('Rewards pareto = ' + str(acumulatedRewards) + '| Episode steps : ' + str(episodeSteps) + '| Episode = ' + str(e) )
            
            
            self.epsilon_decrease()
            
            e+=1
            
            #print("pr1")
            #print(self.metrics.paretor0)
            #print("pr2")
            #print(self.metrics.paretor1)
            
        if log:
            self.send_wandb_metrics()
            data = [[x, y] for (x, y) in zip(self.metrics.paretorewards1, self.metrics.paretorewards2)]
            table = wandb.Table(data=data, columns = ["x", "y"])
            wandb.log({"my_custom_plot_id" : wandb.plot.scatter(table, "x", "y", title="Custom Y vs X Scatter Plot")})


            metr.close_wandb()
            
        import matplotlib.pyplot as plt
        plt.plot(self.metrics.paretor1,self.metrics.paretor0)
        
        plt.ylabel("Treasure Reward  " )
        plt.xlabel("Time Penalty " )
        

        
        
        
        
               
        plt.show()
        #metr.plot_p_front2(self.metrics.paretor0,self.metrics.paretor1)
            
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

