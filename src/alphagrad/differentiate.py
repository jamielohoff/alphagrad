from typing import Sequence

import jax
import jax.numpy as jnp

from graphax import VertexGameState
from alphagrad.utils import make_init_state


# TODO documentation
def batch_vertex_game_states(games: Sequence[VertexGameState], num_devices: int = 1) -> VertexGameState:
	batchsize = len(games)
	ts = jnp.zeros(batchsize)
	edges = jnp.stack([game.edges for game in games])
	vertices = jnp.stack([game.vertices for game in games])
	attn_masks = jnp.stack([game.attn_mask for game in games])
	if num_devices > 1:
		batchsize = edges.shape[0]
		edges_shape = edges.shape[1:]
		vertices_shape = vertices.shape[1:]
		mask_shape = attn_masks.shape[1:]
		per_device_batchsize = batchsize // num_devices
  
		ts = ts.reshape(num_devices, per_device_batchsize)
		edges = edges.reshape(num_devices, per_device_batchsize, *edges_shape)
		vertices = vertices.reshape(num_devices, per_device_batchsize, *vertices_shape)
		attn_masks = attn_masks.reshape(num_devices, per_device_batchsize, *mask_shape)
	return VertexGameState(t=ts,
							edges=edges,
							vertices=vertices,
							attn_mask=attn_masks)


# TODO documentation
def differentiate(network, num_intermediates, env_interaction_fn, key, games):
	init_carry = make_init_state(games, num_intermediates, key)
	output = env_interaction_fn(network, init_carry)
	return output[:, -1, -2].flatten()

