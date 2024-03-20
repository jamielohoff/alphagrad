from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax.sharding import PositionalSharding

import equinox as eqx

# cpu_devices = jax.devices("cpu")
gpu_devices = jax.devices("gpu")

gpu_sharding = PositionalSharding(gpu_devices).reshape(3, 1)
# cpu_sharding = PositionalSharding(cpu_devices).reshape(2, 1)

x = jnp.arange(3 * 2 * 2.).reshape((3, 2, 2))

y = jnp.arange(3 * 2 * 2.).reshape((3, 2, 2)) ** 2

out = jax.pmap(jnp.dot)(x, y)  

print(out)  

