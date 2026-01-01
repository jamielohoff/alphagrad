from typing import Any, Sequence, Tuple
from functools import partial
from importlib import import_module

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu

import optax
import equinox as eqx

Array = jax.Array
PRNGKey = jax.Array

# Taken from https://openreview.net/forum?id=r1lyTjAqYX
def symlog(x: float) -> float:
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def symexp(x: float) -> float:
    return jnp.sign(x) * jnp.exp(jnp.abs(x) - 1)  


# Taken from https://arxiv.org/pdf/1805.11593
def default_value_transform(x: Array, eps: float = 0.001) -> Array:
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x


def default_inverse_value_transform(x: Array, eps: float = 0.001) -> Array:
    return jnp.sign(x)*((( jnp.sqrt( 1+4*eps*(jnp.abs(x)+1+eps) ) - 1 )/(2*eps))**2 - 1)


# Definition of some RL metrics for diagnostics
def explained_variance(value, empirical_return) -> float:
    return 1. - jnp.var(value) / jnp.var(empirical_return)


# Function to calculate the entropy of a probability distribution
def entropy(prob_dist: Array) -> float:
    return -jnp.sum(prob_dist * jnp.log(prob_dist + 1e-7), axis=-1)


select_first = lambda x: x[0] if isinstance(x, jax.Array) else x
parallel_mean = lambda x: jnp.mean(x, axis=0)


id = lambda x: x

def get_value_tf(name: str):
	if name == "log":
		# Stolen from DreamerV3
		return symlog, symexp
	elif name == "default":
		# Stolen from R2D2
		return default_value_transform, default_inverse_value_transform
	else:
		return id, id


@partial(jax.vmap, in_axes=(None, None, 0, 0, 0, None, 0))
def A0_loss_fn(
    value_transform,
    network, 
	policy_target,
    value_target,
	state,
	value_weight,
	key: PRNGKey
) -> Tuple[float, Any]:
	"""Loss function as defined in AlphaZero paper.

	Args:
		value_transform (Callable): Value transform function.
		network (Array): Transformer policy and value model.
		policy_target (Array): Policy targets computed in the tree search.
		value_target (Array): Value targets computed in the tree search.
		state (Array): States from the environment.
		value_weight (Array): Weight of the value function loss.
		key (PRNGKey): Random key.
	Returns:
		Tuple: Returns loss function and auxiliary metrics.
	"""
	output = network(state, key=key)
	value, policy_logits = output[0], output[1:]

	policy_probs = jnn.softmax(policy_logits, axis=-1)

 	# Action_weights, i.e. policy_target used as prob. dist. targets!
	policy_loss = optax.softmax_cross_entropy(policy_logits, policy_target)
	entropy = optax.softmax_cross_entropy(policy_logits, policy_probs)
 
	# Value targets are transformed with `value_transform`
	value_loss = optax.l2_loss(value, value_transform(value_target[0]))
 
	# Computing the L2 regularization
	params = eqx.filter(network, eqx.is_array)
	squared_sums = jtu.tree_map(lambda x: jnp.sum(jnp.square(x)), params)
	l2_loss = jtu.tree_reduce(lambda x, y: x+y, squared_sums)
 
	# Computing the explained variance
	explained_var = explained_variance(value, value_target)
 
	loss = policy_loss 
	loss += value_weight*value_loss 
	aux = jnp.stack((
    	policy_loss, 
		value_weight*value_loss, 
		l2_loss, 
		entropy,
		explained_var
	))
	return loss, aux


def A0_loss(
    value_transform,
    network, 
    policy_target, 
    value_target,
    state, 
    value_weight,
    keys
):
	loss, aux = A0_loss_fn(
    	value_transform,
		network, 
		policy_target, 
		value_target, 
		state,
		value_weight,
		keys
	)
	return loss.mean(), aux.mean(axis=0)


def get_masked_logits(logits, state):
	# Create action mask
	mask = state.at[1, 0, :].get()
	return jnp.where(mask == 0, logits, jnp.finfo(logits.dtype).min)


@partial(jax.vmap, in_axes=(0,))
def postprocess_data(data: Array) -> Array:
	"""Reverses the partial cumulative reward.

	Args:
		data (Array): Tree search output.
	Returns:
		data: Reversed cumulative reward.
	"""
	values = data[::-1, -3]
	return data.at[:, -3].set(values)


def make_init_state(graph: Array, key: PRNGKey) -> Tuple:
	"""
	Function that creates the initial state for the tree search.
 
	Args:
		graph (Array): Tree search input graph.
		key (PRNGKey): Random key.
	Returns:
		Tuple: Initial state for the tree search.
	"""
	batchsize = graph.shape[0]
	return (graph, jnp.zeros(batchsize), key)


def task_builder(task: str):
    """Importing the `make_XXX` function of the task and creating the graph
    """
    package = "alphagrad.experiments"
    module = import_module(package)
    
    task_fn_name = "make_" + task
    make_fn = getattr(module, task_fn_name)
    graph, graph_shape, task_fn = make_fn()
    return graph, graph_shape, task_fn


def joint_task_builder(task_names: Sequence[str]):
    package = "alphagrad.experiments"
    module = import_module(package)
    
    graphs, graph_shapes, task_fns = [], [], []
  
    for task_fn_name in task_names:
        print(f"task: {task_fn_name}")
        make_fn = getattr(module, "make_" + task_fn_name)
        graph, graph_shape, task_fn = make_fn()
        graphs.append(graph)
        graph_shapes.append(graph_shape)
        task_fns.append(task_fn)
    return graphs, graph_shapes, task_fns

