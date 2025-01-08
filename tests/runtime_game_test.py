import os
import jax
import jax.numpy as jnp
import jax.lax as lax

from multiprocessing import set_start_method
# set_start_method("spawn")

from graphax.examples import RoeFlux_1d, Helmholtz

from alphagrad.vertexgame.runtime_game import RuntimeGameParallel

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

NUM_ENVS = 4

if __name__ == '__main__':
    set_start_method("spawn")
    
    xs = [jnp.array([0.15, 0.15, 0.2, 0.3], device=jax.devices("cpu")[0])]
    env = RuntimeGameParallel(NUM_ENVS, 100, 5, Helmholtz, *xs)
    
    # xs = [jnp.array([0.15], device=jax.devices("cpu")[0])]*6
    # env = RuntimeGameParallel(NUM_ENVS, 100, 5, RoeFlux_1d, *xs)
    
    states = env.reset_envs()
    for i in reversed(range(env.env.num_actions)):
        states, reward, term = env.steps(states, jnp.array([i]*NUM_ENVS))

    # states, reward, term = env_steps(states, jnp.array([5, 5, 5, 5]))
    
    print("rewards", reward)

