import functools as ft
from typing import Callable

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import chex
import optax
import equinox as eqx

from graphax import VertexGameState


def symlog(x: chex.Array) -> chex.Array:
    return jnp.sign(x)*jnp.log(jnp.abs(x)+1)


def symexp(x: chex.Array) -> chex.Array:
    return jnp.sign(x)*jnp.exp(jnp.abs(x)-1)


@ft.partial(jax.vmap, in_axes=(None, 0, 0, 0, None, None, 0))
def _A0_loss(network, 
			policy_target,
			value_target,
			obs,
			policy_weight,
			L2_weight,
			key):
	output = network(obs, key)
	policy_logits = output[1:]
	value = output[0:1]

	policy_loss = optax.softmax_cross_entropy(policy_logits, policy_target)
	value_loss = optax.l2_loss(value, symlog(value_target))[0] # added symlog for reward scaling
 
	params = eqx.filter(network, eqx.is_array)
	squared_sums = jtu.tree_map(lambda x: jnp.sum(jnp.square(x)), params)
	L2_reg = jtu.tree_reduce(lambda x, y: x+y, squared_sums)
	return policy_weight*policy_loss + value_loss + L2_weight*L2_reg, jnp.stack((policy_weight*policy_loss, value_loss, L2_weight*L2_reg))


def A0_loss(network, 
            policy_target, 
            value_target,
            obs, 
            policy_weight,
            L2_weight,
            keys):
	loss, aux =  _A0_loss(network, 
						policy_target, 
						value_target, 
						obs,
						policy_weight,
						L2_weight,
						keys)
	return loss.mean(), aux.mean(axis=0)


def get_masked_logits(logits, state, num_intermediates):
	one_hot_state = jnn.one_hot(state.vertices-1, num_intermediates)
	action_mask = one_hot_state.sum(axis=0)
	return jnp.where(action_mask == 0, logits, -100000.)


def postprocess_data(data: chex.Array, idx: int = -2) -> chex.Array:
	"""
	TODO add documentation

	Args:
		data (_type_): _description_
		idx (int, optional): _description_. Defaults to 0.

	Returns:
		_type_: _description_
	"""
	data = data.reshape(-1, *data.shape[2:])
	final_rew = data.at[-1, idx].get()
	data = data.at[:, idx].add(-final_rew)
	return data.at[:, idx].multiply(-1.)


def preprocess_data(games, key):
	"""
	TODO add docstring
	"""
	batchsize = len(games.t)
	shape = games.edges.shape[1:]
	ts = games.t.reshape(batchsize, -1)
	info = games.info.reshape(batchsize, -1)
	edges = games.edges.reshape(batchsize, *shape)
	vertices = games.vertices.reshape(batchsize, -1)
	batched_games = VertexGameState(t=ts,
								info=info,
								edges=edges,
								vertices=vertices)
	return (batched_games, jnp.zeros(batchsize), key)

