import jax
import jax.nn as jnn
import jax.numpy as jnp


from alphagrad.vertexgame import step, make_graph
from graphax.examples import Helmholtz

batched_one_hot = jax.vmap(jnn.one_hot, in_axes=(0, None))

edges = make_graph(Helmholtz, jnp.ones(4))

for i in [2, 5, 4, 3, 1]:
    edges, reward, terminated = step(edges, i-1)
    output_mask = edges.at[2, 0, :].get()
    vertex_mask = edges.at[1, 0, :].get() - output_mask
    print(vertex_mask)
    attn_mask = jnp.logical_or(vertex_mask.reshape(1, -1), vertex_mask.reshape(-1, 1)).astype(jnp.int32)
    print(attn_mask)

