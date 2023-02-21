import functools as ft
import random
import numpy as np
from collections import deque

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu

import optax
import equinox as eqx


# find good value for regularizer
@ft.partial(jax.vmap, in_axes=(None, 0, 0, 0, None, 0))
def _A0_loss(network, 
			policy_target,
			value_target,
			obs,
			L2_weight,
			key):
	output = network(obs)
	policy_logits = output[1:]
	value = output[0]

	policy_loss = optax.softmax_cross_entropy(policy_logits, policy_target) # this is a crossentropy
	value_loss = optax.l2_loss(value, value_target)
 
	params = eqx.filter(network, eqx.is_array)
	squared_sums = jtu.tree_map(lambda x: jnp.sum(jnp.square(x)), params)
	L2_reg = jtu.tree_reduce(lambda x, y: x+y, squared_sums)

	return value_loss + policy_loss + L2_weight*L2_reg 


def A0_loss(network, 
            policy_target, 
            value_target, 
            obs, 
            L2_weight,
            keys):
	return _A0_loss(network, 
					policy_target, 
					value_target, 
					obs,
					L2_weight,
					keys).mean()


def get_masked_logits(logits, state, num_intermediates):
	one_hot_state = jnn.one_hot(state.state-1, num_intermediates)
	action_mask = one_hot_state.sum(axis=0)
	return jnp.where(action_mask == 0, logits, -100000.)

