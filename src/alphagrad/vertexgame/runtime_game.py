import time
from functools import partial
from typing import Callable, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from chex import Array

from .core import (vertex_eliminate, 
                    get_elimination_order, 
                    get_vertex_mask, 
                    get_shape)
from .interpreter import make_graph
from .transforms import minimal_markowitz
    
from graphax import jacve

State = Tuple[Array, Array]
EnvOut = Tuple[State, float, bool]

# NOTE: this env is not jittable so far since the tracing mechanism of jacve
# is not compatible with jit itself. The resulting function is however jittable,
# which is why we can measure the execution time with a lax.scan. 
# NOTE: working with runtimes also is not nicely parallelizable since the
# tracing mechanism of jacve is not compatible with vmap/pmap.
class RuntimeGame:
    f: Callable
    num_samples: int
    num_actions: int
    graph: Array
    reward_fn: Callable
    
    def __init__(self, num_samples: int, f: Callable, *xs) -> None:
        self.graph = make_graph(f, *xs)
        self.f = f
        self.num_actions = self.graph.at[0, 0, 1].get()
        self.num_samples = num_samples
        self.reward_fn = partial(_get_reward, num_samples, f, *xs)
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self) -> State:
        return (self.graph, jnp.zeros(self.num_actions, dtype=jnp.int32))
    
    # @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: int) -> EnvOut:  
        """
        OpenAI-like environment for a game where to goal is to find the 
        best vertex elimination order with minimal multiplication count.
        This game always has finite termination range.

        The `state` of the game is essentially the matrix containing the edges of the
        computational graph and the array containing the edges that have already been
        eliminated.

        The `reward` is the negative number of multiplications since we want to 
        minimize that.

        The `action space` is equal to the number of remaining vertices that can be
        eliminated. For example, for 10 intermediate variables, there are 10 
        different actions. However, every action can only be executed once. This is
        realized by action masking where the logits of actions that have already 
        been performed are sent to -inf.

        The `termination` of the game is indicated by the is_bipartite feature, i.e.
        the game is over when all intermediate vertices and edges have been eliminated.
        """
        
        new_state, num_eliminated_vertices, num_intermediates, new_act_seq = _step(state, action)
        
        terminated = lax.select(num_eliminated_vertices == num_intermediates, True, False)
        reward = self.reward_fn(act_seq=new_act_seq) if terminated else 0.0

        return new_state, reward*1., terminated
    
    
@partial(jax.jit, donate_argnums=(0,))
def _step(state, action):
    edges, act_seq = state
    idx = jnp.sum(jnp.where(act_seq > 0, 1, 0)).astype(jnp.int32)
    # Actions go from 0 to num_intermediates-1 
    # and vertices go from 1 to num_intermediates      
    vertex = action + 1
    t = jnp.where(get_elimination_order(edges) > 0, 1, 0).sum()
    new_edges, nops = vertex_eliminate(vertex, edges)
    new_edges = new_edges.at[3, 0, t].set(vertex)
    
    # Reward is the negative of the multiplication count
    num_eliminated_vertices = get_vertex_mask(new_edges).sum()
    num_intermediates = get_shape(new_edges)[1]
    new_act_seq = act_seq.at[idx].set(action)
    new_state = (new_edges, new_act_seq)
    return new_state, num_eliminated_vertices, num_intermediates, new_act_seq


def _get_reward(num_samples: int, f: Callable, *xs, act_seq=None) -> float:
    if type(act_seq) is str:
        order = act_seq
    else: 
        order = [int(a)+1 for a in act_seq] 
        
    argnums = list(range(len(xs)))
    jac_fn = jax.jit(jacve(f, order=order, argnums=argnums))
    
    def measure_time(i, _):
        start = time.time()
        jac = jac_fn(*xs)
        jax.block_until_ready(jac)
        end = time.time()
        return i+1, end - start
    
    _, dts = lax.scan(measure_time, 0, jnp.zeros(num_samples))
    return -dts[1:].mean()

