import functools as ft

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

from chex import Array, PRNGKey
import optax
import equinox as eqx


def symlog(x: Array) -> Array:
    return jnp.sign(x)*jnp.log(jnp.abs(x)+1)


def symexp(x: Array) -> Array:
    return jnp.sign(x)*jnp.exp(jnp.abs(x)-1)  


# Definition of some RL metrics for diagnostics
def explained_variance(value, empirical_return):
    return 1. - jnp.var(value)/jnp.var(empirical_return)


# Function to calculate the entropy of a probability distribution
def entropy(prob_dist):
    return -jnp.sum(prob_dist*jnp.log(prob_dist + 1e-7), axis=-1)


@ft.partial(jax.vmap, in_axes=(None, 0, 0, 0, None, None, None, 0))
def A0_loss_fn(network, 
				policy_target,
				value_target,
				state,
				value_weight,
				L2_weight,
				entropy_weight,
				key: PRNGKey):
	"""Loss function as defined in AlphaZero paper with additional entropy
	regularization to promote exploration


	Args:
		network (_type_): _description_
		policy_target (_type_): _description_
		value_target (_type_): _description_
		state (_type_): _description_
		policy_weight (_type_): _description_
		L2_weight (_type_): _description_
		entropy_weight (_type_): _description_
		key (PRNGKey): _description_

	Returns:
		_type_: _description_
	"""
	output = network(state, key=key)
	policy_logits = output[1:]
	value = output[0]

	policy_probs = jnn.softmax(policy_logits, axis=-1) # action_weights are a probability distribution!
	policy_log_probs = jnn.log_softmax(policy_logits, axis=-1)
	# policy_loss = -jnp.dot(policy_log_probs, policy_target)
	# entropy_reg = -jnp.sum(policy_log_probs*jnp.exp(policy_log_probs))
	policy_loss = optax.softmax_cross_entropy(policy_logits, policy_target).sum() # -jnp.sum(policy_probs*(jnp.log(policy_target+1e-7) - policy_log_probs))
	entropy = -jnp.sum(policy_probs*policy_log_probs)
 
	# Added symlog for reward scaling
	value_loss = optax.l2_loss(value, symlog(value_target[0]))
 
	# Computing the L2 regularization
	params = eqx.filter(network, eqx.is_array)
	squared_sums = jtu.tree_map(lambda x: jnp.sum(jnp.square(x)), params)
	L2_loss = jtu.tree_reduce(lambda x, y: x+y, squared_sums)
 
	# Computing the explained variance
	explained_var = explained_variance(value, value_target)
 
	loss = policy_loss 
	loss += value_weight*value_loss 
	loss += L2_weight*L2_loss
	# loss += entropy_weight*entropy_reg
	aux = jnp.stack((policy_loss, 
                	value_weight*value_loss, 
                 	L2_weight*L2_loss, 
					entropy,
                   	explained_var))
	return loss, aux



def A0_loss(network, 
            policy_target, 
            value_target,
            state, 
            value_weight,
            L2_weight,
            entropy_weight,
            keys):
	loss, aux =  A0_loss_fn(network, 
							policy_target, 
							value_target, 
							state,
							value_weight,
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
	"""Function that computes the returns for certain steps appropriately.

	Args:
		data (_type_): _description_ 

	Returns:
		_type_: _description_
	"""
	values = data[::-1, -2]
	return data.at[:, -2].set(values)


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

