from typing import Any, Callable
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import mctx
import equinox as eqx

PyTreeDef = Any

# Preconfigured functions for tree search
select_params = lambda x, y: y if eqx.is_inexact_array(x) else x   


def make_recurrent_fn(
    value_transform: Callable,
    inverse_value_transform: Callable,
    network: PyTreeDef,
    step: Callable,
    get_masked_logits: Callable
) -> Callable:
    """Implementation of the recurrent function for tree searchas required by
    the MuZero algorithm. The recurrent function is used to expand the tree
    at the leaf node with a new node. The initial action probabilities will be
    biased with the prediction from the neural network.

    Args:
        value_transform (Callable): Function to transform value for training
        inverse_value_transform (Callable): Function to inverse transform value for prediction
        network (chex.PyTreeDef): Neural network model
        step (Callable): Environment dynamics function
        get_masked_logits (Callable): Function that masks logits for invalid actions

    Returns:
        Callable: Recurrent function for tree search
    """
    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def recurrent_fn(params, rng_key, actions, states):
        state = states[-5:]
        next_state, reward, _ = step(state, actions) # Env dynamics function
        states = jnp.roll(states, shift=-5, axis=1)
        next_states = states.at[-5:].set(next_state)
        
        # Map parameters to prediction function, i.e. neural network
        model = jtu.tree_map(select_params, network, params)

        # Compute policy and value for leaf node with neural network model
        # Symexp for reward scaling since value head is trained on symlog(value)
        output = model(next_states, rng_key)
        policy_logits = output[1:]
        value = inverse_value_transform(output[0])
        
        # Create mask for invalid actions, i.e. already eliminated vertices
        masked_logits = get_masked_logits(policy_logits, state)

        # Expand the tree at the leaf with a new node
        # The initial action probabilities will be biased with the prediction
        # from the neural network
        # On a single-player environment, use discount from [0, 1].
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward, discount=1., prior_logits=masked_logits, value=value
        )
        return recurrent_fn_output, next_states
    
    return recurrent_fn


def make_environment_interaction(
    value_transform: Callable,
    inverse_value_transform: Callable,
    num_actions: int,
    num_considered_actions: int,
    gumbel_scale: int,
    num_simulations: int,
    recurrent_fn: Callable,
    step: Callable,
    **kwargs
) -> Callable:
    """Implementation of the environment interaction function for the MuZero
    algorithm. The environment interaction function is used to simulate the
    environment and the agent's interaction with the environment. The agent
    will interact with the environment by selecting actions based on the
    policy derived from the tree search.
    
    Args:
        value_transform (Callable): Function to transform value for training
        inverse_value_transform (Callable): Function to inverse transform value for prediction
        num_actions (int): Number of actions
        num_considered_actions (int): Number of considered actions
        gumbel_scale (int): Gumbel scale parameter
        num_simulations (int): Number of simulations
        recurrent_fn (Callable): Recurrent function for tree search
        step (Callable): Environment dynamics function
        **kwargs: Additional keyword arguments
    
    Returns:
        Callable: Environment interaction function for the MuZero algorithm
    """
    qtransform = partial(mctx.qtransform_completed_by_mix_value, **kwargs)
    
    def environment_interaction(network, init_carry):
        batchsize = init_carry[1].shape[0]
        batched_network = eqx.filter_vmap(network)
        params = eqx.filter(network, eqx.is_inexact_array)
        
        def loop_fn(carry, _):
            states, num_muls, key = carry
            key, subkey = jrand.split(key, 2)
            state = states[:, -5:] 
            
            # Create action mask
            mask = state.at[:, 1, 0, :].get()

            # Getting policy and value estimates
            keys = jrand.split(key, batchsize)
            output = batched_network(states, keys)
            policy_logits = output[:, 1:]
            value = inverse_value_transform(output[:, 0])

            # Configuring root node of the tree search
            treesearch_root = mctx.RootFnOutput(
                prior_logits=policy_logits, value=value, embedding=states
            )
                                    
            # Gumbel MuZero is so much better!
            # Smaller Gumbel noise helps improve performance, but too small kills learning
            policy_output = mctx.gumbel_muzero_policy(
                params,
                subkey,
                treesearch_root,
                recurrent_fn,
                num_simulations,
                invalid_actions=mask,
                qtransform=qtransform,
                gumbel_scale=gumbel_scale,
                max_num_considered_actions=num_considered_actions
            )

            # Tree search derived targets for policy and value function
            search_policy = policy_output.action_weights
            
            # Always take action recommended by tree search because for this
            # action, the gumbel method guarantees a policy improvement
            action = policy_output.action

            # Step the environment
            next_state, rewards, done = jax.vmap(step)(state, action)
            states = jnp.roll(states, shift=-5, axis=1)
            next_states = states.at[:, -5:].set(next_state)
            
            # Compute the number of multiplications
            num_muls += rewards
            
            # Package up everything for further processing
            state_flattened = state.reshape(batchsize, -1)
            
            aux = jnp.concatenate([
                state_flattened,
                search_policy, 
                rewards[:, jnp.newaxis], 
                value[:, jnp.newaxis],
                # search_value[:, jnp.newaxis],
                done[:, jnp.newaxis]], 
                axis=1
            )
            
            return (next_states, num_muls, key), aux

        perf, output = lax.scan(loop_fn, init_carry, None, length=num_actions)
        final_state = perf[0][:, -5:]
        num_muls = perf[1]
        return final_state, num_muls, output.transpose(1, 0, 2)
    
    return environment_interaction

