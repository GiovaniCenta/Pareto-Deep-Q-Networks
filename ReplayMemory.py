from collections import namedtuple
Transition = namedtuple('Transition',
                        ['state',
                         'action',
                         'reward',
                         'next_state',
                         'terminal'])
import numpy as np
from collections import deque

class ReplayMemory(object):

    def __init__(self, observation_shape, observation_type='float16', size=1000000, nO=1):
        self.current = 0
        # we will only save next_states,
        # as current state is simply the previous next state.
        # We thus need an extra slot to prevent overlap between the first and
        # last sample
        
        self.size = size

        self.actions = np.empty((size,), dtype='uint8')
        if observation_shape == (1,):
            self.next_state = np.empty((size,), dtype=observation_type)
        else:
            self.next_state = np.empty((size,) + observation_shape, dtype=observation_type)
        self.rewards = np.empty((size, nO), dtype='float16')
        self.terminals = np.empty((size,), dtype=bool)

    def add(self, transition):
        # first sample, need to save current state
        if self.current == 0:
            self.next_state[0] = transition.state

        self.current += 1
        current = self.current % self.size
        self.actions[current] = transition.action
        self.next_state[current] = transition.next_state
        self.rewards[current] = transition.reward
        self.terminals[current] = transition.terminal

    def sample(self, batch_size):
        assert self.current > 0, 'need at least one sample in memory'
        high = self.current % self.size
        # did not fill memory
        if self.current < self.size:
            # start at 1, as 0 contains only current state
            low = 1
        else:
            # do not include oldest sample, as it's state (situated in previous sample)
            # has been overwritten by newest sample
            low = high - self.size + 2
        indexes = np.empty((batch_size,), dtype='int32')
        i = 0
        while i < batch_size:
            
            # include high
            s = np.random.randint(low, high+1)
            # cannot include first step of episode, as it does not have a previous state
            #if not self.terminals[s-1]:
            indexes[i] = s
            i += 1
        batch = Transition(
            self.next_state[indexes-1],
            self.actions[indexes],
            self.rewards[indexes],
            self.next_state[indexes],
            self.terminals[indexes]
        )
        return batch
