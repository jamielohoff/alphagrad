import random
from collections import deque
from typing import Sequence, Tuple

import numpy as np

import chex

from graphax import GraphInfo
 
class AlphaGradReplayBuffer:
    """TODO write documentation

    Args:
        object (_type_): _description_
    """
    capacity: int
    memory: Sequence[chex.Array]
    edges_shape: Tuple[int, int, int]
    split_idxs: Tuple[int, int, int]
    
    def __init__(self, 
                capacity: int,
                info: GraphInfo) -> None:
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        
        num_i = info.num_inputs
        num_v = info.num_intermediates
        num_o = info.num_outputs
        self.edges_shape = (num_v, num_i+num_v, num_v+num_o)
        
        obs_idx = np.prod(np.array(self.edges_shape))
        policy_idx = obs_idx + num_v
        reward_idx = policy_idx + 1
        self.split_idxs = (obs_idx, policy_idx, reward_idx)

    def push(self, samples: chex.Array) -> None:
        """
        Save a transition or a batch of transitions
        sample has to be a contiguous array that contains (in this order):
            observation (currently edge tensor)
            search_policy
            rewards
            done
        """
        if samples.ndim == 1:
            self.memory.append(samples)
        else: 
            self.memory.extend([sample for sample in samples])

    def sample(self, batchsize: int) -> Sequence[chex.Array]:
        """Sample a batch of transitions from the replay buffer and return them as a batched tuple"""
        transitions = np.stack(random.sample(self.memory, batchsize))
        obs, search_policy, rewards, terminal = np.split(transitions, self.split_idxs, axis=1)
        obs = obs.reshape(batchsize, *self.edges_shape)
        return obs, search_policy, rewards, terminal

    def __len__(self) -> int:
        return len(self.memory)

