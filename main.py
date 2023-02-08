import numpy as np
from Pareto import Pareto
from ReplayMemory import ReplayMemory
import gym
from gym import wrappers
from deepst import DeepSeaTreasure
import NNs
env = DeepSeaTreasure()


Pareto = Pareto(env)
number_of_episodes = 1000
number_of_p_points = 2
number_of_objectives = 2
reward_min = 0.0
reward_max = 48.0
e=0
memory_capacity = 100
D = ReplayMemory(memory_capacity)
minibatch_size = 1   #todo: isso aqui muda??
t = False



state = Pareto.initializeState()
while e < number_of_episodes:
    step = 0
    while t is False:
        samples = Pareto.sample_points(num_samples = number_of_p_points, d = number_of_objectives, low = reward_min, high = reward_max)
        
        #os q's aqui se calcuiam a partir da target??? no caso do dqn os q's futuros vem da target e os qqqqqs atuais da model, correto?
        q_set = Pareto.calculate_q_set(samples,state)
        print("qqset")
        print(q_set)
        print("\n\n")
        
        
        
        hv = Pareto.compute_hypervolume(q_set,ref_point = np.array([-30,-30]))
    
        action = Pareto.e_greedy_action(hv)

        next_state, reward, terminal, _ = env.step(action)
        
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
        
        yi = np.array([0.0,0.0])
        
        #inputPoints = inputPoints.tolist()

        
        
        if minibatch_terminal is not True:
            #todo: entender essa função
            #pegar a função do código do PQL
            
            inputPoints = [list([val[0]]) for val in q_set.values()]
        
        
            ndPoints, dominatedPoints = Pareto.ND(inputPoints, Pareto.dominates)
            
            #todo: certo isso?
            yi = ndPoints
            
          
            
        else:
            yi = minibatch_reward
        
        
        print("\nyi")
        print(yi)
            
        minibatch_q_set = Pareto.calculate_q_i(samples_i,minibatch_state,minibatch_action)
        print("\nminibatch q set")
        print(minibatch_q_set)
        
        
        
        #todo: arrumar isso aqui
        
        Pareto.update_non_dominated_estimator(minibatch_action,minibatch_q_set,yi)
        
        
        
        reward_estimated = Pareto.estimate_reward(minibatch_state,minibatch_action)
        print("reward_estimated")
        print(reward_estimated)
        
        Pareto.update_reward_estimator(reward_estimated,minibatch_reward)
        Pareto.copy_to_target(step)
            
        step+=1
        

    e+=1       






