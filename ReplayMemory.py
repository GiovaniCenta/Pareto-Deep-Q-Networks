from collections import namedtuple
Transition = namedtuple('Transition',
                        ['state',
                         'action',
                         'reward',
                         'next_state',
                         'terminal'])
import numpy as np
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]
