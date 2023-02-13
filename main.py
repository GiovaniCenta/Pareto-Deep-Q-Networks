import numpy as np
from Pareto import Pareto
from ReplayMemory import ReplayMemory
import gym
from gym import wrappers
from deepst import DeepSeaTreasure
import torch
from metrics import metrics

env = DeepSeaTreasure()


Pareto = Pareto(env)
number_of_episodes = 1000
number_of_p_points = 30
number_of_objectives = 2
reward_min = 0.0
reward_max = 64
e=0
memory_capacity = 150
D = ReplayMemory(memory_capacity)
minibatch_size = 1   #todo: isso aqui muda??

metr = metrics()
MAX_STEPS = 30


while e < number_of_episodes:
    state = Pareto.initializeState()
    print("ep = " + str(e))
    terminal = False
    acumulatedRewards = [0,0]
    
    step = 0
    while terminal is False or step == MAX_STEPS:
        #env.render()
        
        samples = Pareto.sample_points(num_samples = number_of_p_points, d = number_of_objectives, low = reward_min, high = reward_max)
        
        
        
        #os q's aqui se calcuiam a partir da target??? no caso do dqn os q's futuros vem da target e os qqqqqs atuais da model, correto?
        qset = Pareto.calculate_q_set(samples,state)
        
        
       
        
        
        
        hv = Pareto.compute_hypervolume(qset,ref_point = np.array([-30,-20]))
    
        action = Pareto.e_greedy_action(hv)
        
        
        
        

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
        samples_i = Pareto.sample_points(num_samples = number_of_p_points, d = number_of_objectives, low = reward_min, high = reward_max)
        
        
        
        #inputPoints = inputPoints.tolist()
        
        
        
        
        if minibatch_terminal is not True:
            #todo: entender essa função
            #pegar a função do código do PQL
            q_set_hat = Pareto.calculate_q_set(samples,minibatch_next_state,use_target_nd = True)
            
            inputPoints = [list([val]) for val in q_set_hat.values()]
            
            
            
            inputPoints = [[(t[0][0].item()),(t[0][1].item())] for t in inputPoints]
            #print(inputPoints)
            #exit(8)
            
            
            
            
        
        
            ndPoints, dominatedPoints = Pareto.ND(inputPoints, Pareto.dominates)
            
            #todo: uma descida de gradiente para cada ponto?
            #todo: certo isso?
            yi = ndPoints
            
            
          
            
        else:
            yi = minibatch_reward
        
        
        
        #minibatch_q_set = Pareto.calculate_q_i(samples_i,minibatch_state,minibatch_action)
        
        
        
        
        
        #todo: arrumar isso aqui
        
        
        
        samples_i = Pareto.sample_points(num_samples = number_of_p_points, d = number_of_objectives, low = reward_min, high = reward_max)
        o1 = samples_i[5]
        
        #uodate neural networks
        Pareto.update_non_dominated_estimator(minibatch_state,minibatch_action,o1,yi)
        Pareto.update_reward_estimator(minibatch_state,minibatch_action,minibatch_reward)
        
        #copy to target
        Pareto.copy_to_target(step)
        
        
            
        step+=1
       
        next_state = Pareto.flatten_observation(next_state)
        state = next_state
        
        
        
    metr.rewards1.append(acumulatedRewards[0])
    metr.rewards2.append(acumulatedRewards[1])
    metr.episodes.append(e)
    
    print(acumulatedRewards)
    
    print(Pareto.epsilon)
        
    Pareto.epsilon_decrease()
    
    e+=1       

metr.plotGraph()





#todo R target?
