from typing import Any, Callable, Tuple
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import mctx
import equinox as eqx

import alphagrad.utils as u


Array = jax.Array
PyTreeDef = Any
EnvStepFn = Callable[[Array, Array], Tuple[Array, float, bool]]
ValueTransform = Callable[[float], float]

# Preconfigured functions for tree search
select_params = lambda x, y: y if eqx.is_inexact_array(x) else x   


def make_recurrent_fn(
    model: PyTreeDef,
    step: EnvStepFn,
    inverse_value_transform: ValueTransform,
) -> Callable:
    """Creates a recurrent function for tree search as required by the MuZero 
    algorithm. This function is used to expand the tree at the leaf node with a 
    new node. The initial action probabilities are biased with the prediction 
    from the neural network.
    
    Args:
        model: Neural network model of the agent.
        step: Environment dynamics function that takes state and action, 
        returns next state, reward, and done flag.
        inverse_value_transform: Function to inverse transform the value 
        for prediction.
    Returns:
        A callable recurrent function for tree search node expansion.
    """
    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def recurrent_fn(params, rng_key, actions, states):
        state = states[-5:]
        next_state, reward, _ = step(state, actions) # Env dynamics function
        states = jnp.roll(states, shift=-5, axis=1)
        next_states = states.at[-5:].set(next_state)
        
        # Map parameters to prediction function, i.e. neural network
        _model = jtu.tree_map(select_params, model, params)

        # Compute policy and value for leaf node with neural network model
        # Symexp for reward scaling since value head is trained on symlog(value)
        output = _model(next_states, rng_key)
        policy_logits = output[1:]
        value = inverse_value_transform(output[0])
        
        # Create mask for invalid actions, i.e. already eliminated vertices
        masked_logits = u.get_masked_logits(policy_logits, state)

        # Expand the tree at the leaf with a new node
        # The initial action probabilities will be biased with the prediction
        # from the neural network
        # On a single-player environment, use discount from [0, 1].
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward, discount=1., prior_logits=masked_logits, value=value
        )
        return recurrent_fn_output, next_states
    
    return recurrent_fn


def make_tree_search(
    model: PyTreeDef,
    step: EnvStepFn,
    num_actions: int,
    inverse_value_transform: ValueTransform,
    num_considered_actions: int = 5,
    gumbel_scale: float = 1.0,
    num_simulations: int = 25,
    **qtransformkwargs
) -> Tuple[Array, int, Array]:
    """Implementation of the environment interaction function for the MuZero
    algorithm. The environment interaction function is used to simulate the
    environment and the agent's interaction with the environment. The agent
    will interact with the environment by selecting actions based on the
    policy derived from the tree search.
    
    Args:
        model (PyTreeDef): Neural network model of the agent. 
        step (Callable): Environment dynamics function.
        num_actions (int): Number of actions. Should be same as rollout length,
        i.e. number of intermediate vertices.
        inverse_value_transform (Callable): Function to inverse transform value 
        for prediction.
        num_considered_actions (int): Number of considered actions.
        gumbel_scale (float): Gumbel scale parameter.
        num_simulations (int): Number of simulations.
        **qtransformkwargs: Additional keyword arguments for Q-transform.
    
    Returns:
        A callable to create model-based RL samples for MCTS.
    """
    qtransform = partial(
        mctx.qtransform_completed_by_mix_value, **qtransformkwargs
    )
    
    recurrent_fn = make_recurrent_fn(model, step, inverse_value_transform)
    
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
                done[:, jnp.newaxis]], 
                axis=1
            )
            
            return (next_states, num_muls, key), aux

        perf, output = lax.scan(loop_fn, init_carry, None, length=num_actions)
        final_state = perf[0][:, -5:]
        num_muls = perf[1]
        return final_state, num_muls, output.transpose(1, 0, 2)
    
    return environment_interaction

