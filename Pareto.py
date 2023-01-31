import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from pygmo import hypervolume
class Pareto:
    def __init__(self,env,epsilon_start = 0.998,epsilon_decay = 0.998,epsilon_min = 0.01):
        
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_start
        self.env=env
        self.number_of_actions = 4
        self.number_of_states = 110
        self.reward_estimator = self.initialize_reward_estimator(111,55)
        self.statesActions = np.zeros((1, 110+1))
        self.statesActionsNonDominatedEstim = np.zeros((1, 110+1))
        self.gamma = 0.98
        

        self.non_dominated_estimator = self.initialize_non_dominated_estimator(111,55)
        self.q_set = {v: [] for v in range(self.number_of_actions)}
        


        


    def sample_points(self,num_samples, d, low, high):
    # Generate random samples from a uniform distribution with lower boundary low and upper boundary high
    
        d = d-1
        samples = np.random.uniform(low, high, (num_samples, d))
        return samples

    def initialize_reward_estimator(self,in1,in2):


        """        model = Sequential()
        model.add(Dense(55, input_dim=110, activation='relu')) # 1st hidden layer with 55 neurons
        model.add(Dense(4, input_dim=4, activation='relu')) # 2nd hidden layer with 4 neurons
        model.add(Dense(55, input_dim=59, activation='relu')) # 3rd hidden layer with 55 neurons
        model.add(Dense(2, activation='softmax')) # Output layer with 2 neurons and softmax activation
        """
        model = tf.keras.Sequential()
        nA = self.number_of_actions
        nO = 2
        nS = 110
        fc1_in = nS + nO - 1
        model.add(tf.keras.layers.Dense(110, input_dim=fc1_in))
        model.add(tf.keras.layers.Dense(nA, input_dim=nA))
        model.add(tf.keras.layers.Dense(55, input_dim=55 + nA))
        model.add(tf.keras.layers.Dense(2))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def initialize_non_dominated_estimator(self,in1,in2):
        model = tf.keras.Sequential()
        nA = self.number_of_actions
        nO = 2
        nS = 110
        fc1_in = nS + nO - 1
        model.add(tf.keras.layers.Dense(110, input_dim=fc1_in))
        model.add(tf.keras.layers.Dense(nA, input_dim=nA))
        model.add(tf.keras.layers.Dense(55, input_dim=55 + nA))
        model.add(tf.keras.layers.Dense(1))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def calculate_q_set(self,samples,state) -> dict:
        for action in range(self.number_of_actions):
           print(state)
           print(action)
           self.statesActions[state] = action
           input_data = self.statesActions
           reward_estimated = self.reward_estimator.predict(input_data)
           
           
           for o1 in samples:
                self.statesActionsNonDominatedEstim[state] = o1
                
                o_d_aprox = self.non_dominated_estimator.predict(self.statesActionsNonDominatedEstim)
                
                
                q_point = reward_estimated + np.sum(self.gamma*o_d_aprox, axis=-1, keepdims=True)
                
                self.q_set[action] = q_point

        return self.q_set

    

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
            points = np.array(q_set[i]) * -1.
            hv = hypervolume(points)
            # use negative ref-point for minimization
            q_values[i] = hv.compute(ref_point*-1)
        return q_values

    def e_greedy_action(self,hv):
        if np.random.rand() >= self.epsilon:
            
            action =  np.random.choice(np.argwhere(hv == np.amax(hv)).flatten())
        else:
            
            action =  self.env.action_space.sample()
        
        self.epsilon_decrease()
        return action


    def epsilon_decrease(self):
        if self.epsilon < self.epsilon_min:
            pass
        else:
            self.epsilon = self.epsilon * self.epsilon_decay

