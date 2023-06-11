import functools as ft

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import chex
import optax
import equinox as eqx
import numpy as np

from graphax import VertexGameState


def symlog(x: chex.Array) -> chex.Array:
    return jnp.sign(x)*jnp.log(jnp.abs(x)+1)


def symexp(x: chex.Array) -> chex.Array:
    return jnp.sign(x)*jnp.exp(jnp.abs(x)-1)


@ft.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, None, None, 0))
def _A0_loss(network, 
			policy_target,
			value_target,
			obs,
			attn_mask,
			policy_weight,
			L2_weight,
			key):
	output = network(obs, mask=attn_mask, key=key)
	policy_logits = output[1:]
	value = output[0:1]

	policy_log_probs = jnn.log_softmax(policy_logits, axis=-1)
	policy_loss = optax.kl_divergence(policy_log_probs, policy_target)
	value_loss = optax.l2_loss(value, symlog(value_target))[0] # added symlog for reward scaling
 
	params = eqx.filter(network, eqx.is_array)
	squared_sums = jtu.tree_map(lambda x: jnp.sum(jnp.square(x)), params)
	L2_reg = jtu.tree_reduce(lambda x, y: x+y, squared_sums)
	return policy_weight*policy_loss + value_loss + L2_weight*L2_reg, jnp.stack((policy_weight*policy_loss, value_loss, L2_weight*L2_reg))


def A0_loss(network, 
            policy_target, 
            value_target,
            obs, 
            attn_mask,
            policy_weight,
            L2_weight,
            keys):
	loss, aux =  _A0_loss(network, 
						policy_target, 
						value_target, 
						obs,
						attn_mask,
						policy_weight,
						L2_weight,
						keys)
	return loss.mean(), aux.mean(axis=0)


def get_masked_logits(logits, state, num_intermediates):
	one_hot_state = jnn.one_hot(state.vertices-1, num_intermediates)
	action_mask = one_hot_state.sum(axis=0)
	return jnp.where(action_mask == 0, logits, jnp.finfo(logits.dtype).min)


@ft.partial(jax.vmap, in_axes=(0,))
def postprocess_data(data: chex.Array) -> chex.Array:
	"""
	TODO add documentation

	Args:
		data (_type_): _description_ 

	Returns:
		_type_: _description_
	"""
	idx = jnp.where(data[:, -2] > 0, 1., 0).sum() - 1
	idx = idx.astype(jnp.int32)
	final_rew = data.at[-1, -2].get()
	data = data.at[:, -2].add(-final_rew)
	return data.at[:, -2].multiply(-1.)


def make_init_state(games, num_intermediates, key):
	"""
	TODO add docstring
	"""
	batchsize = len(games.t)
	obs = lax.slice_in_dim(games.edges, 0, num_intermediates, axis=-1)
	return (obs, games, jnp.zeros(batchsize), key)  


def make_batched_vertex_game(edges, vertices, attn_masks, num_devices: int = 1) -> VertexGameState:
	"""
	TODO add docstring
	"""
	batchsize = edges.shape[0]
	edges = edges.numpy().astype(np.float32)
	vertices = vertices.numpy().astype(np.float32)
	attn_masks = attn_masks.numpy().astype(np.float32)

	ts = jnp.array([jnp.where(v > 0, 1, 0).sum() for v in vertices])
 
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


