import numpy as np
from Pareto import Pareto
from ReplayMemory import ReplayMemory
import gym
from gym import wrappers
from deepst import DeepSeaTreasure
import torch
from metrics import metrics
import copy


import matplotlib.pyplot as plt




env = DeepSeaTreasure()
memory_capacity = 30
metr = metrics()
number_of_episodes = 50000
starting_learn = 50
D = ReplayMemory((110,),size= memory_capacity, nO=2)
Pareto = Pareto(env=env,metrs=metr,step_start_learning = starting_learn,numberofeps = number_of_episodes,ReplayMem = D,
    number_of_p_points = 10, epsilon_start = 1.,epsilon_decay = 0.99997,epsilon_min = 0.01,gamma = 1,copy_every=100,ref_point = [-1,-2] )


number_of_p_points = 10
number_of_objectives = 2
reward_min = 0.0
reward_max = 64
e=0


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

polIndex = 0
qtable = np.zeros((110, 4, 2),dtype=object)
starting_learn = 50
normalize_reward = {'min': np.array([0,0]), 'scale': np.array([124, 19])}
polDict = np.zeros((number_of_episodes, 4,number_of_p_points, 2))
while e < number_of_episodes:
    state = Pareto.initializeState()
    
    one_hot_state = np.zeros(110)
    terminal = False
    acumulatedRewards = [0,0]
    total_steps = 0
    step = 0
    qcopy =[]
    while terminal is False or step == MAX_STEPS:
        one_hot_state[state] = 1
        #env.render()
        if total_steps > 2*starting_learn:
            
            
            
            
            q_front = Pareto.q_front(np.expand_dims(np.array(one_hot_state), 0), n=number_of_p_points, use_target_network=False)

            
            qcopy = copy.deepcopy(q_front)

            
            
            
            hv = Pareto.compute_hypervolume(q_front[0],4,np.array([-1,-2]))
                           

        
            action = Pareto.e_greedy_action(hv)
            
            
            
            
            

        else:    
            action =  env.action_space.sample()
        next_state, reward, terminal, _ = env.step(action)
        
       
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
        

        
        if total_steps > starting_learn:
            minibatch = D.sample(minibatch_size)
            
            minibatch_non_dominated = []
            minibatch_states = []
            minibatch_actions = []
            
            
            
            

            minibatch_rew_normalized = (minibatch.reward -normalize_reward['min'])/normalize_reward['scale']
          
            
            batch_q_front_next = Pareto.q_front(minibatch.next_state, n=number_of_p_points, use_target_network=True)
            
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
            

            
            e_loss = Pareto.nd_estimator.update(minibatch_non_dominated[:, -1:].astype(np.float32),
                                                    minibatch_states.astype(np.float32),
                                                    minibatch_non_dominated[:, :-1].astype(np.float32),
                                                    minibatch_actions.astype(np.float32),
                                                    step=total_steps)
            

            Pareto.rew_estim.update(minibatch.reward.astype(np.float32),
                                        minibatch.state,
                                        minibatch.action.astype(np.long),
                                        step=total_steps)
            
            
        
        
    metr.rewards1.append(acumulatedRewards[0])
    metr.rewards2.append(acumulatedRewards[1])
    metr.episodes.append(e)

    episodeqtable = copy.deepcopy(qcopy)

    try:
        
        polDict[e] = episodeqtable
    except ValueError:
        pass
    
    

    
    
    print("episode = " + str(e) +  " | rewards = [" + str(acumulatedRewards[0]) + "," + str(acumulatedRewards[1]) + "]")
        
    Pareto.epsilon_decrease()
    
    e+=1
    total_steps = total_steps + step
    

#metr.plotGraph()
Pareto.pareto_frontier_policy_training(polDict)






#todo R target?
