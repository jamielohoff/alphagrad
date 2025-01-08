import os
import time
import jax
import jax.numpy as jnp

from multiprocessing import set_start_method, Pool

import graphax as gx
from graphax.examples import RoeFlux_1d, Helmholtz

from alphagrad.vertexgame.runtime_game import RuntimeGame

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

NUM_ENVS = 4
    
# xs = [jnp.array([0.15, 0.15, 0.2, 0.3], device=jax.devices("cpu")[0])]
# env = RuntimeGame(100, 5, Helmholtz, *xs)

xs = [jnp.array([0.15], device=jax.devices("cpu")[0])]*6
env = RuntimeGame(100, 5, RoeFlux_1d, *xs)


def _init_envs(obs, act_seq):
    """
    Worker initializer to set up the global state.
    """
    global process_obs
    global process_act_seq
    process_obs = obs
    process_act_seq = act_seq
    
    
def reset_envs():
    obs, act_seq = env.reset()
    init_obs = jnp.stack([obs for _ in range(NUM_ENVS)])
    init_act_seq = jnp.stack([act_seq for _ in range(NUM_ENVS)])
    for _ in range(NUM_ENVS):
        pool.apply_async(_init_envs, args=(obs, act_seq))
    return (init_obs, init_act_seq)  
    
    
def step_fn(i, obs, action):
    return i, env.step(obs, action)

    
# Multi-processing step
def steps(states, actions):
    states = jax.device_put(states, jax.devices("cpu")[0])
    actions = jax.device_put(actions, jax.devices("cpu")[0])
    states = [(states[0][i], states[1][i]) for i in range(NUM_ENVS)]
    
    out = {}
    jobs = [pool.apply_async(step_fn, args=(i, state, action)) 
            for i, (state, action) in enumerate(zip(states, actions))]

    # Collect results and maintain order
    for job in jobs:
        key, result = job.get()
        out[key] = result
    
    next_states, rewards, dones = zip(*out.values())
    
    next_obs, next_act_seqs = zip(*list(next_states))
    next_states = (jnp.stack(list(next_obs)), jnp.stack(list(next_act_seqs)))
    
    rewards = jnp.array(list(rewards))
    dones = jnp.array(list(dones))
    
    return next_states, rewards, dones


if __name__ == '__main__':
    set_start_method("spawn")
    
    pool = Pool(
        processes=NUM_ENVS,
        initializer=_init_envs,
        initargs=(None, None)  # Placeholder for state initialization
    )
    
    out = gx.jacve(RoeFlux_1d, order="rev")(*xs)
        
    states = reset_envs()
    order = [100, 98, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    for i in order:
        states, reward, term = steps(states, jnp.array([i-1]*NUM_ENVS))

    # states, reward, term = env_steps(states, jnp.array([5, 5, 5, 5]))
    
    print("rewards", reward)

