import time
from functools import partial
from typing import Callable, Tuple

from multiprocessing import Pool, Process, Manager, set_start_method

import jax
import jax.lax as lax
import jax.numpy as jnp

import equinox as eqx

from chex import Array

from .core import (vertex_eliminate, 
                    get_elimination_order, 
                    get_vertex_mask, 
                    get_shape)
from .interpreter import make_graph
    
from graphax import jacve

State = Tuple[Array, Array]
EnvOut = Tuple[State, float, bool]


# NOTE: this env is not jittable so far since the tracing mechanism of jacve
# is not compatible with jit itself. The resulting function is however jittable,
# which is why we can measure the execution time with a lax.scan. 
# NOTE: working with runtimes also is not nicely parallelizable since the
# tracing mechanism of jacve is not compatible with vmap/pmap.
class RuntimeGame:
    fn: Callable
    num_samples: int
    num_actions: int
    burnin: int
    graph: Array
    reward_fn: Callable
    reward_scale: float
    
    def __init__(self, num_samples: int, burnin: int, fn: Callable, *xs, reward_scale: float = 1.) -> None:
        self.graph = make_graph(fn, *xs)
        self.fn = fn
        self.num_actions = self.graph.at[0, 0, 1].get()
        self.num_samples = num_samples
        self.burnin = burnin
        self.reward_fn = partial(_get_reward2, num_samples, burnin, fn, *xs)
        self.reward_scale = reward_scale
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self) -> State:
        return (self.graph, jnp.zeros(self.num_actions, dtype=jnp.int32))
    
    # @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: int) -> EnvOut:  
        """
        OpenAI gym-like environment for a game where to goal is to find the 
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

        The `terminated` of the game is indicated by the is_bipartite feature, i.e.
        the game is over when all intermediate vertices and edges have been eliminated.
        """
        new_state, new_act_seq, terminated = _step(state, action)
        reward = self.reward_fn(act_seq=new_act_seq) if terminated else 0.0
        return new_state, reward*self.reward_scale, terminated
        
    
@partial(jax.jit, donate_argnums=(0,), device=jax.devices("cpu")[0])
def _step(state: State, action: int) -> Tuple[State, Array, bool]:
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
    terminated = num_eliminated_vertices == num_intermediates
    return new_state, new_act_seq, terminated


def _get_reward(num_samples: int, burnin: int, fn: Callable, *xs, act_seq=None) -> float:
    """_summary_

    Args:
        num_samples (int): _description_
        burnin (int): _description_
        fn (Callable): _description_
        act_seq (_type_, optional): _description_. Defaults to None.

    Returns:
        float: _description_
    """
    if type(act_seq) is str:
        order = act_seq
    else: 
        order = [int(a)+1 for a in act_seq] 
        
    argnums = list(range(len(xs)))
    jac_fn = jax.jit(jacve(fn, order=order, argnums=argnums), device=jax.devices("cpu")[0])
    
    def measure_time(i, _):
        start = time.time()
        jac = jac_fn(*xs)
        jax.block_until_ready(jac)
        end = time.time()
        return i+1, end - start
    
    _, dts = lax.scan(measure_time, 0, jnp.zeros(num_samples))
    return -dts[burnin:].mean()


def _get_reward2(num_samples: int, burnin: int, fn: Callable, *xs, act_seq=None) -> float:
    """_summary_

    Args:
        num_samples (int): _description_
        burnin (int): _description_
        fn (Callable): _description_
        act_seq (_type_, optional): _description_. Defaults to None.

    Returns:
        float: _description_
    """
    if type(act_seq) is str:
        order = act_seq
    else: 
        order = [int(a)+1 for a in act_seq] 
        
    argnums = list(range(len(xs)))
    xs = [jnp.stack([x]*512) for x in xs]
    jac_fn = jacve(fn, order=order, argnums=argnums)
    
    # We need to create a more realistic measurement setting!
    def measure_time(i, _):
        jac = jax.vmap(jac_fn)(*xs)
        jax.block_until_ready(jac)
        return i+1, jac
    
    st = time.time()
    _, dts = eqx.filter_jit(lax.scan, device=jax.devices("cpu")[0])(measure_time, 0, [jnp.zeros(1)]*burnin)
    print("compilation time:", time.time() - st)
    
    start = time.time()
    _, dts = eqx.filter_jit(lax.scan, device=jax.devices("cpu")[0])(measure_time, 0, [jnp.zeros(1)]*num_samples)
    end = time.time()
    return (start-end) / num_samples

