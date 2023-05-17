import os
import copy
import argparse
import wandb
import time
from functools import partial

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
from jax.sharding import Mesh, PositionalSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

import equinox as eqx

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform" # increases performance enormously!
cpu_devices = jax.devices("cpu")
print(cpu_devices)

devices = mesh_utils.create_device_mesh((8,), devices=cpu_devices)
mesh = Mesh(devices, axis_names=('i',))
sharding = PositionalSharding(devices)


xs = jnp.ones((8, 2))

@partial(eqx.filter_pmap, in_axes=(0,), devices=cpu_devices)
def pmap_test(init):
    def loop_fn(carry, _):
        new_carry = carry + 1
        return new_carry, jrand.normal(jrand.PRNGKey(42), (16, 517))
    
    _, output = lax.scan(loop_fn, init, None, length=15)
    return output.transpose(1, 0, 2)
out = pmap_test(xs)
print(out.shape, out.devices())
new_out = jax.device_put(out, device=cpu_devices[0])
print(new_out.shape, new_out.devices())

# xs = jax.device_put(xs, sharding.reshape(8, 1, 1))

# @partial(shard_map, mesh=mesh, in_specs=(P("i",)), out_specs=P("i",))
# def postprocess_data(init):
#     def loop_fn(carry, _):
#         new_carry = carry + 1
#         return new_carry, jnp.ones((8, 517))
    
#     _, output = lax.scan(loop_fn, init, None, length=15)
#     return output.transpose(1, 0, 2)

# out = postprocess_data(xs)
# print(out.shape, out.devices())

