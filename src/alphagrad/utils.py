import functools as ft

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

from chex import Array
import optax
import equinox as eqx


def symlog(x: Array) -> Array:
    return jnp.sign(x)*jnp.log(jnp.abs(x)+1)


def symexp(x: Array) -> Array:
    return jnp.sign(x)*jnp.exp(jnp.abs(x)-1)  


@ft.partial(jax.vmap, in_axes=(None, 0, 0, 0, None, None, None, 0))
def _A0_loss(network, 
			policy_target,
			value_target,
			state,
			policy_weight,
			L2_weight,
			entropy_weight,
			key):
	output = network(state, key=key)
	policy_logits = output[1:]
	value = output[0:1]

	policy_log_probs = jnn.log_softmax(policy_logits, axis=-1)
	policy_loss = optax.kl_divergence(policy_log_probs, policy_target)
	entropy_reg = optax.kl_divergence(policy_log_probs, jnp.exp(policy_log_probs))
	value_loss = optax.l2_loss(value, symlog(value_target))[0] # added symlog for reward scaling
 
	params = eqx.filter(network, eqx.is_array)
	squared_sums = jtu.tree_map(lambda x: jnp.sum(jnp.square(x)), params)
	L2_reg = jtu.tree_reduce(lambda x, y: x+y, squared_sums)
 
	loss = policy_weight*policy_loss + value_loss + L2_weight*L2_reg + entropy_weight*entropy_reg
	return loss, jnp.stack((policy_weight*policy_loss, value_loss, L2_weight*L2_reg, entropy_reg*entropy_weight))


def A0_loss(network, 
            policy_target, 
            value_target,
            state, 
            policy_weight,
            L2_weight,
            entropy_weight,
            keys):
	loss, aux =  _A0_loss(network, 
						policy_target, 
						value_target, 
						state,
						policy_weight,
						L2_weight,
						entropy_weight,
						keys)
	return loss.mean(), aux.mean(axis=0)


def get_masked_logits(logits, state):
	# Create action mask
	mask = state.at[1, 0, :].get()
	return jnp.where(mask == 0, logits, jnp.finfo(logits.dtype).min)


@ft.partial(jax.vmap, in_axes=(0,))
def postprocess_data(data: Array) -> Array:
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


def make_init_state(edges, key):
	"""
	TODO add docstring
	"""
	batchsize = edges.shape[0]
	return (edges, jnp.zeros(batchsize), key)  


def make_batch(edges, num_devices: int = 1) -> Array:
	"""
	TODO add docstring
	"""
	batchsize = edges.shape[0]
	edges = edges.numpy().astype(jnp.int32)
	edges = jnp.array(edges)
 
	if num_devices > 1:
		edges_shape = edges.shape[1:]
		per_device_batchsize = batchsize // num_devices
		edges = edges.reshape(num_devices, per_device_batchsize, *edges_shape)

	return edges


def update_best_scores(single_best_performance, names, rews, orders):
	for name, rew, order in zip(names, rews, orders):
		if rew > single_best_performance[name][0]:
			single_best_performance[name] = (rew, order)
	print(single_best_performance)

