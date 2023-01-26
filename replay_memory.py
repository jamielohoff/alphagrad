import random
import numpy as np
from collections import deque


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, values):
        """Save a transition"""        
        self.memory.append(values)

    def sample(self, batch_size):
        states, actions, next_states, rewards, dones = zip(*random.sample(self.memory, batch_size))
        return (np.stack(states), 
                np.array(actions), 
                np.stack(next_states),
                np.array(rewards),
                np.array(dones))

    def __len__(self):
        return len(self.memory)
   
 
class MCTSReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, values):
        """Save a transition"""        
        self.memory.append(values)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return np.stack(transitions)

    def __len__(self):
        return len(self.memory)
    

class EdgeReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, values):
        """Save a transition"""        
        self.memory.append(values)

    def sample(self, batch_size):
        states, edges, actions, next_states, next_edges, rewards, dones = zip(*random.sample(self.memory, batch_size))
        return (np.stack(states), 
                np.stack(edges),
                np.array(actions), 
                np.stack(next_states),
                np.stack(next_edges),
                np.array(rewards),
                np.array(dones))

    def __len__(self):
        return len(self.memory)
   
