
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
A demonstration of the policy improvement by planning with Gumbel.
"""

import os
import functools as ft 

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import mctx
import optax
import equinox as eqx
import chex

from gridworld import GridworldGame2D

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

SEED = 42
BATCHSIZE = 32
NUM_SIMULATIONS = 50
NUM_ACTIONS = 5
NUM_RUNS = 100
ITERATIONS = 500
UPDATE_FREQ = 25

key = jrand.PRNGKey(SEED)
goal = jnp.array([4, 0])
walls = jnp.array( [[0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0]])  

env = GridworldGame2D(walls, goal)

pi_key, v_key, key = jrand.split(key, 3)
class NeuralNetwork(eqx.Module):
    """
    Simple neural network.
    """
    embedding: eqx.nn.Embedding
    mlp: eqx.nn.MLP

    def __init__(self, in_size, embedding_size, out_size, width_size, depth_size, key):
        mlp_key, embed_key = jrand.split(key, 2)
        self.embedding = eqx.nn.Embedding(in_size, embedding_size, key=embed_key)
        self.mlp = eqx.nn.MLP(embedding_size, out_size, width_size, depth_size, key=mlp_key)

    def __call__(self, x: chex.Array):
        idx = jnp.nonzero(x.flatten(), size=1, fill_value=0)
        embedding = self.embedding(idx)[0]
        return self.mlp(embedding)


policy_network = eqx.nn.MLP(4*25, 5, 128, 3, key=pi_key)
value_network = eqx.nn.MLP(4*25, 1, 128, 3, key=v_key)

policy_params = eqx.filter(policy_network, eqx.is_inexact_array)
value_params = eqx.filter(value_network, eqx.is_inexact_array)

batch_step = jax.vmap(env.step, in_axes=(0, 0))
batch_reset = jax.vmap(env.reset, in_axes=(0,))
batch_get_obs = jax.vmap(env.get_observation, in_axes=(0,))


def recurrent_fn(params, rng_key, actions, states):
	del rng_key
	policy_params, value_params = params

	next_states, obs, reward, done = batch_step(states, actions)

	pi_network = jtu.tree_map(lambda x, y: y if eqx.is_inexact_array(x) else x, policy_network, policy_params)
	v_network = jtu.tree_map(lambda x, y: y if eqx.is_inexact_array(x) else x, value_network, value_params)
	policy_logits = jax.vmap(pi_network)(obs)
	values = jnp.ravel(jax.vmap(v_network)(obs))

	# On a single-player environment, use discount from [0, 1].
	discount = jnp.ones(BATCHSIZE)
	recurrent_fn_output = mctx.RecurrentFnOutput(reward=reward,
												discount=discount,
												prior_logits=policy_logits,
												value=values)
	return recurrent_fn_output, next_states


@ft.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0))
def actor_critic_loss(policy_params, 
					value_params,
					policy_target,
					value_target,
					obs,
					key):
	pi_network = jtu.tree_map(lambda x, y: y if eqx.is_inexact_array(x) else x, policy_network, policy_params)
	v_network = jtu.tree_map(lambda x, y: y if eqx.is_inexact_array(x) else x, value_network, value_params)
	policy_logits = pi_network(obs)
	values = v_network(obs)[0]

	policy_loss = jnp.sum(policy_target*(jnp.log(policy_target) - policy_logits)) # this is a KL divergence
	value_loss = optax.l2_loss(value_target - values)

	return policy_loss + value_loss


def batch_loss(params, 
				policy_target, 
				value_target, 
				states, 
				keys):
	policy_params, value_params = params
	return actor_critic_loss(policy_params, 
							value_params, 
							policy_target, 
							value_target, 
							states, 
							keys).mean()


avg_return_smoothing=.9
value_target = "maxq"
def scan(f, init, xs, length=None):
	if xs is None:
		xs = [None] * length
	carry = init
	ys = []
	for x in xs:
		carry, y = f(carry, x)
		ys.append(y)
	return carry, None


def agent_environment_interaction_loop_fn(policy_net, value_net, init_carry):
	def loop_function(carry, _):
		policy_params, value_params, target_params, policy_opt_state, value_opt_state, states, episode_return, avg_return, t, key = carry

		pi_network = jtu.tree_map(lambda x, y: y if eqx.is_inexact_array(x) else x, policy_net, policy_params)
		v_network = jtu.tree_map(lambda x, y: y if eqx.is_inexact_array(x) else x, value_net, target_params)
		obs = batch_get_obs(states)
		policy_logits = jax.vmap(pi_network)(obs)
		values = jnp.ravel(jax.vmap(v_network)(obs))

		root = mctx.RootFnOutput(prior_logits=policy_logits,
								value=values,
								embedding=states)
		key, subkey = jrand.split(key, 2)
		params = (policy_params, target_params)
		qtransform = ft.partial(mctx.qtransform_completed_by_mix_value,
								use_mixed_value=True,
								rescale_values=False,
        						maxvisit_init=50,
								value_scale=.1)

		policy_output = mctx.gumbel_muzero_policy(params=params,
												rng_key=subkey,
												root=root,
												recurrent_fn=recurrent_fn,
												num_simulations=NUM_SIMULATIONS,
												max_num_considered_actions=NUM_ACTIONS,
												qtransform=qtransform)

		# tree search derived targets for policy and value function
		search_policy = policy_output.action_weights
		if(value_target == "maxq"):
			search_value = policy_output.search_tree.qvalues(jnp.full(BATCHSIZE, policy_output.search_tree.ROOT_INDEX))[jnp.arange(BATCHSIZE), policy_output.action]
		elif(value_target == "nodev"):
			search_value = policy_output.search_tree.node_values[:, policy_output.search_tree.ROOT_INDEX]
		else:
			raise ValueError("Unknown value target.")

		# compute loss gradient compared to tree search targets and update parameters
		key, subkey = jrand.split(key, 2)
		subkeys = jrand.split(subkey, BATCHSIZE)
		params = (policy_params, value_params)
		loss, grads = eqx.filter_value_and_grad(batch_loss)(params, search_policy, search_value, obs, subkeys)
		policy_grads, value_grads = grads

		policy_updates, new_policy_opt_state = optim.update(policy_grads, policy_opt_state, params=policy_params)
		value_updates, new_value_opt_state = optim.update(value_grads, value_opt_state, params=value_params)

		new_policy_network = eqx.apply_updates(policy_network, policy_updates)
		new_value_network = eqx.apply_updates(value_network, value_updates)

		new_policy_params = eqx.filter(new_policy_network, eqx.is_inexact_array)
		new_value_params = eqx.filter(new_value_network, eqx.is_inexact_array)

		# Update target params after a particular number of parameter updates
		t += 1
		new_target_params = jtu.tree_map(lambda x, y: jnp.where(t % UPDATE_FREQ == 0, x, y), new_value_params, target_params)

		# always take action recommended by tree search
		actions = policy_output.action
		print(actions)

		# step the environment
		next_states, _, reward, done = batch_step(states, actions)

		# update statistics for computing average return
		episode_return += reward
		avg_return = jnp.where(done, avg_return*avg_return_smoothing + episode_return*(1-avg_return_smoothing), avg_return)
		next_carry = (new_policy_params, new_value_params, new_target_params, new_policy_opt_state, new_value_opt_state, next_states, episode_return, avg_return, t, key)
		return next_carry, None

	output, _ = lax.scan(loop_function, init_carry, None, length=ITERATIONS)

	return output


optim = optax.adamw(1e-3)
policy_opt_state = optim.init(eqx.filter(policy_network, eqx.is_inexact_array))
value_opt_state = optim.init(eqx.filter(value_network, eqx.is_inexact_array))

keys = jrand.split(key, BATCHSIZE)
states = batch_reset(keys)
carry = (policy_params, value_params, value_params, policy_opt_state, value_opt_state, states, jnp.zeros(BATCHSIZE), jnp.zeros(BATCHSIZE), 0, key)
for _ in range(NUM_RUNS):
	output = eqx.filter_jit(agent_environment_interaction_loop_fn)(policy_network, value_network, carry) # agent_environment_interaction_loop_fn(policy_network, value_network, carry) # 
	policy_params, value_params, target_params, policy_opt_state, value_opt_state, _, episode_return, avg_return, t, key = output

	print(episode_return)
	print("Average return", episode_return.mean())

	states = batch_reset(keys)
	carry = (policy_params, value_params, value_params, policy_opt_state, value_opt_state, states, jnp.zeros(BATCHSIZE), jnp.zeros(BATCHSIZE), 0, key)


	p_network = jtu.tree_map(lambda x, y: y if eqx.is_inexact_array(x) else x, policy_network, policy_params)
	state = jnp.array([0, 1])
	obs = env.get_observation(state)
	print(p_network(obs))
	action = jnp.argmax(policy_network(obs)).astype(jnp.int32)
	
	for i in range(50):
		state, obs, reward, done = env.step(state, action)
		print(p_network(obs))
		action = jnp.argmax(p_network(obs)).astype(jnp.int32)
		print(state, action)

