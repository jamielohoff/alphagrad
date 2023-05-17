import random
from collections import deque
from typing import Sequence, Tuple

import numpy as np

import chex

from graphax import GraphInfo
from ..alphagrad.data.transform import prune_samples
 
class ReplayBuffer:
    """TODO write documentation

    Args:
        object (_type_): _description_
    """
    capacity: int
    memory: Sequence[chex.Array]
    
    def __init__(self, 
                capacity: int) -> None:
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, samples: chex.Array) -> None:
        """
        Save a transition or a batch of transitions
        sample has to be a contiguous array that contains (in this order):
            observation (currently edge tensor)
            search_policy
            rewards
            done
        """
        self.memory.extend([sample for sample in samples])

    def sample(self, batchsize: int) -> Sequence[chex.Array]:
        """
        Sample a batch of transitions from the replay buffer and return 
        them as a batched tuple
        """
        transitions = np.stack(random.sample(self.memory, batchsize))
        transitions = prune_samples(transitions, self.edges_shape[0]) if self.temporal_pruning else transitions
        obs, search_policy, rewards, terminal = np.split(transitions, self.split_idxs, axis=-1)
        obs = obs.reshape(batchsize, *self.edges_shape)
        return obs, search_policy, rewards, terminal

    def __len__(self) -> int:
        return len(self.memory)

