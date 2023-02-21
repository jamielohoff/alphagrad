from typing import Callable
import functools as ft

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import chex
import mctx
import equinox as eqx


# TODO needs adjustment to new auto-regressive models!
# TODO add documentation
def make_recurrent_fn(nn_model: chex.PyTreeDef,
                    num_intermediates: int, 
                    batched_step: Callable,
                    batched_get_masked_logits: Callable) -> Callable:
    """TODO write docstring

    Args:
        nn_model (chex.PyTreeDef): _description_
        num_intermediates (int): _description_
        batched_step (Callable): _description_
        batched_get_masked_logits (Callable): _description_

    Returns:
        Callable: _description_
    """
    def recurrent_fn(params, rng_key, actions, state):
        del rng_key
        batchsize, nn_params = params
        next_state, reward, _ = batched_step(state, actions) # dynamics function
        next_obs = next_state.edges[:, jnp.newaxis, :, :]

        # prediction function
        network = jtu.tree_map(lambda x, y: y if eqx.is_inexact_array(x) else x, nn_model, nn_params)
        batch_network = jax.vmap(network)

        output = batch_network(next_obs)
        policy_logits = output[:, 1:]
        masked_logits = batched_get_masked_logits(policy_logits, state, num_intermediates)
        value = output[:, 0]

        # On a single-player environment, use discount from [0, 1].
        discount = jnp.ones(batchsize)
        recurrent_fn_output = mctx.RecurrentFnOutput(reward=reward,
                                                    discount=discount,
                                                    prior_logits=masked_logits,
                                                    value=value)
        return recurrent_fn_output, next_state
    
    return recurrent_fn


def make_environment_interaction(num_intermediates: int,
                                num_simulations: int,
                                recurrent_fn: Callable,
                                batched_step: Callable,
                                batched_one_hot: Callable, **kwargs):
    """TODO write docstring
    """
    def environment_interaction(network: chex.PyTreeDef, batchsize: int, init_carry):
        batched_network = jax.vmap(network)
        nn_params = eqx.filter(network, eqx.is_inexact_array)
        
        def loop_fn(carry, _):
            state, rews, key = carry
            obs = state.edges[:, jnp.newaxis, :, :]
    
            # create action mask
            one_hot_state = batched_one_hot(state.state-1, num_intermediates)
            mask = one_hot_state.sum(axis=1)

            output = batched_network(obs)
            policy_logits = output[:, 1:]
            values = output[:, 0]

            root = mctx.RootFnOutput(prior_logits=policy_logits,
                                    value=values,
                                    embedding=state)

            key, subkey = jrand.split(key, 2)

            params = (batchsize, nn_params)
            policy_output = mctx.muzero_policy(params,
                                                subkey,
                                                root,
                                                recurrent_fn,
                                                num_simulations,
                                                invalid_actions=mask,
                                                **kwargs)

            # tree search derived targets for policy and value function
            search_policy = policy_output.action_weights

            # always take action recommended by tree search
            action = policy_output.action

            # step the environment
            next_state, rewards, done = batched_step(state, action)
            rews += rewards	
    
            flattened_obs = obs.reshape(batchsize, -1)
            return (next_state, rews, key), jnp.concatenate([flattened_obs, search_policy, rews[:, jnp.newaxis], done[:, jnp.newaxis]], axis=1)

        _, output = lax.scan(loop_fn, init_carry, None, length=num_intermediates)

        return jnp.stack(output).transpose(1, 0, 2)
    
    return environment_interaction


@ft.partial(jax.vmap, in_axes=(0, None))
def preprocess_data(data, idx=0):
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

