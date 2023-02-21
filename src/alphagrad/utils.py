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


# find good value for regularizer
@ft.partial(jax.vmap, in_axes=(None, 0, 0, 0, None, 0))
def _A0_loss(network, 
			policy_target,
			value_target,
			obs,
			L2_weight,
			key):
	output = network(obs, key)
	policy_logits = output[:, 1:]
	value = output[:, 0:1]

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
	one_hot_state = jnn.one_hot(state.vertices-1, num_intermediates)
	action_mask = one_hot_state.sum(axis=0)
	return jnp.where(action_mask == 0, logits, -100000.)


@ft.partial(jax.vmap, in_axes=(0, None))
def preprocess_data(data: chex.Array, idx: int = 0) -> chex.Array:
    """TODO add documentation

    Args:
        data (_type_): _description_
        idx (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    final_rew = data.at[-1, idx].get()
    
    rew = jnp.roll(data[:, idx], 1, axis=0)
    rew = rew.at[0].set(0.)
    
    val = final_rew - rew
    return data.at[:, idx].set(val)


def random_sample_mask(key: chex.PRNGKey, 
		       			obs: chex.Array,
						search_policy: chex.Array,
						search_value: chex.Array,
						terminated: chex.Array) -> chex.Array:
	t = jrand.randint(key, shape=(1,), minval=1, maxval=obs.shape[1]+1)[0]

	batchsize = obs.shape[0]
	dt = obs.shape[1]-t
	zeros = jnp.zeros((batchsize, dt, obs.shape[2], obs.shape[3]))
	obs = lax.dynamic_update_slice_in_dim(obs, zeros, t, axis=1) # obs.at[t:, :, :].set(0)

	zeros = jnp.zeros((batchsize, dt, search_policy.shape[-1]))
	search_policy = lax.dynamic_update_slice_in_dim(search_policy, zeros, t, axis=1) # search_policy.at[t:, :].set(0)

	zeros = jnp.zeros((batchsize, dt, search_value.shape[-1]))
	search_value = lax.dynamic_update_slice_in_dim(search_value, zeros, t, axis=1) # search_value.at[t:, :].set(0)
	return obs, search_policy, search_value, terminated
	
