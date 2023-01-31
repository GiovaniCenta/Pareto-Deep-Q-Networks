import numpy as np
from Pareto import Pareto
from ReplayMemory import ReplayMemory
import gym
from gym import wrappers
from deepst import DeepSeaTreasure
env = DeepSeaTreasure()


Pareto = Pareto(env)
number_of_episodes = 1000
number_of_p_points = 5
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
    while t is False:
        samples = Pareto.sample_points(num_samples = number_of_p_points, d = number_of_objectives, low = reward_min, high = reward_max)
        q_set = Pareto.calculate_q_set(samples,state)
        hv = Pareto.compute_hypervolume(q_set,ref_point = np.array([-20,-20]))
    
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
        samples = Pareto.sample_points(num_samples = number_of_p_points, d = number_of_objectives, low = reward_min, high = reward_max)

        exit(8)

        e+=1






