import jax
import jax.numpy as jnp

from graphax.examples import RoeFlux_1d, Helmholtz

from alphagrad.vertexgame.runtime_game import RuntimeGame

xs = [jnp.array([0.15, 0.15, 0.2, 0.3])]
env = RuntimeGame(1000, Helmholtz, *xs)

state = env.reset()

state, reward, terminated = env.step(state, 1)
state, reward, terminated = env.step(state, 4)
state, reward, terminated = env.step(state, 2)
state, reward, terminated = env.step(state, 3)
state, reward, terminated = env.step(state, 0)
print(state, reward, terminated)

