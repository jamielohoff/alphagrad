import os
import copy
import argparse
import wandb
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

from tqdm import tqdm

from graphax.game import VertexGame
from graphax.elimination import forward, reverse
from graphax.examples.random import construct_random_graph
from graphax.examples.helmholtz import construct_Helmholtz

from alphazero import A0_loss, get_masked_logits, preprocess_data, MCTSReplayMemory


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="Vertex_A0_test", help="Name of the experiment.")

parser.add_argument("--gpu", type=str, 
                    default="0", help="GPU identifier.")

parser.add_argument("--seed", type=int,
                    default=42, help="Random seed.")

parser.add_argument("--episodes", type=int, 
                    default=500, help="Number of runs on random data.")

parser.add_argument("--replay_size", type=int, 
                    default=2048, help="Size of the replay buffer.")

parser.add_argument("--num_actions", type=int, 
                    default=11, help="Number of actions.")

parser.add_argument("--num_simulations", type=int, 
                    default=15, help="Number of simulations.")

parser.add_argument("--iterations", type=int, 
                    default=11, help="Number of iterations for each MCTS.")

parser.add_argument("--rollout_length", type=int, default=11, 
                    help="Duration of the rollout phase of the MCTS algorithm.")

parser.add_argument("--batchsize", type=int, 
                    default=256, help="Learning batchsize.")

parser.add_argument("--rollout_batchsize", type=int,
                    default=32, 
                    help="Batchsize for environment interaction.")

parser.add_argument("--value_mixing", type=float, default=.5, 
                   help="Mixing between predicted value and MCTS result.")

parser.add_argument("--regularization", type=float, 
                    default=0., help="Contribution of L2_regularization.")

parser.add_argument("--lr", type=float, 
                    default=3e-3, help="Learning rate.")


parser.add_argument("--num_inputs", type=int, 
                    default=4, help="Number input variables.")

parser.add_argument("--num_intermediates", type=int, 
                    default=11, help="Number of intermediate variables.")

parser.add_argument("--num_outputs", type=int, 
                    default=4, help="Number of output variables.")

args = parser.parse_args()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

wandb.init("Vertex_AlphaZero")
wandb.run.name = args.name
wandb.config = vars(args)

NUM_INPUTS = args.num_inputs
NUM_INTERMEDIATES = args.num_intermediates
NUM_OUTPUTS = args.num_outputs

key = jrand.PRNGKey(args.seed)
GS = construct_Helmholtz()


forward_gs = copy.deepcopy(GS)
_, ops = forward(forward_gs, GS.get_info())
print("forward-mode:", ops)


reverse_gs = copy.deepcopy(GS)
_, ops = reverse(reverse_gs, GS.get_info())
print("reverse-mode:", ops)


replay_buffer = MCTSReplayMemory(args.replay_size)
env = VertexGame(GS)


nn_key, key = jrand.split(key, 2)
subkeys = jrand.split(nn_key, 4)
# NN = eqx.nn.MLP(15*15, 12, 128, 2, key=nn_key)
NN = eqx.nn.Sequential([eqx.nn.Conv2d(1, 8, 7, key=subkeys[0]),
                        eqx.nn.Lambda(jnn.relu),
                        
                        eqx.nn.Conv2d(8, 16, 7, key=subkeys[1]),
                        eqx.nn.Lambda(jnn.relu),
                        
                        eqx.nn.Lambda(jnp.ravel),
                        eqx.nn.Linear(144, 32, key=subkeys[2]),
                        eqx.nn.Lambda(jnn.relu),
                        eqx.nn.Linear(32, 12, key=subkeys[3])])

batched_step = jax.vmap(env.step)
batched_reset = jax.vmap(env.reset)
batched_one_hot = jax.vmap(jnn.one_hot, in_axes=(0, None))
batched_get_masked_logits = jax.vmap(get_masked_logits, in_axes=(0, 0, None))


def rollout(policy_network, init_states, batchsize, key):
	def loop_fn(carry, _):
		state, key = carry
		subkey, key = jrand.split(key, 2)
  
		obs = state.edges[:, jnp.newaxis, :, :]
		output = policy_network(obs)
  
		policy_logits = output[:, 1:]
		masked_logits = batched_get_masked_logits(policy_logits, state, NUM_INTERMEDIATES)
  
		actions = jrand.categorical(subkey, masked_logits)
		next_state, reward, _ = batched_step(state, actions)

		return (next_state, key), reward

	init_carry = (init_states, key)
	_, rewards = lax.scan(loop_fn, init_carry, None, length=args.rollout_length)
	return rewards.sum(axis=0)


def recurrent_fn(params, rng_key, actions, state):
	batchsize, nn_params = params
	next_state, reward, _ = batched_step(state, actions) # dynamics function
	next_obs = next_state.edges[:, jnp.newaxis, :, :]
 
	# prediction function
	network = jtu.tree_map(lambda x, y: y if eqx.is_inexact_array(x) else x, NN, nn_params)
	batch_network = jax.vmap(network)
 
	output = batch_network(next_obs)
	policy_logits = output[:, 1:]
	masked_logits = batched_get_masked_logits(policy_logits, state, NUM_INTERMEDIATES)
	value = output[:, 0] 
 
	# On a single-player environment, use discount from [0, 1].
	discount = jnp.ones(batchsize)
	recurrent_fn_output = mctx.RecurrentFnOutput(reward=reward,
												discount=discount,
												prior_logits=masked_logits,
												value=value)
	return recurrent_fn_output, next_state


def environment_interaction(network, batchsize, init_carry):
	batched_network = jax.vmap(network)
	nn_params = eqx.filter(network, eqx.is_inexact_array)
	def loop_fn(carry, _):
		state, rews, key = carry
		obs = state.edges[:, jnp.newaxis, :, :]
  
		# create action mask
		one_hot_state = batched_one_hot(state.state-1, NUM_INTERMEDIATES)
		mask = one_hot_state.sum(axis=1)

		output = batched_network(obs)
		policy_logits = output[:, 1:]
		values = output[:, 0]

		root = mctx.RootFnOutput(prior_logits=policy_logits,
								value=values,
								embedding=state)

		key, subkey = jrand.split(key, 2)

		params = (batchsize, nn_params)
		policy_output = mctx.muzero_policy(params=params,
											rng_key=subkey,
											root=root,
											invalid_actions=mask,
											recurrent_fn=recurrent_fn,
											dirichlet_fraction=0.25,
											num_simulations=args.num_simulations)

		# tree search derived targets for policy and value function
		search_policy = policy_output.action_weights


		# always take action recommended by tree search
		action = policy_output.action

		# step the environment
		next_state, rewards, done = batched_step(state, action)
		rews += rewards	
  
		flattened_obs = obs.reshape(batchsize, -1)
		return (next_state, rews, key), jnp.concatenate([flattened_obs, search_policy, rews[:, jnp.newaxis], done[:, jnp.newaxis]], axis=1)

	_, output = lax.scan(loop_fn, init_carry, None, length=args.iterations)

	return jnp.stack(output).transpose(1, 0, 2)


@ft.partial(jax.vmap, in_axes=(0, None))
def preprocess_data(data, idx=0):
    final_rew = data.at[-1, idx].get()
    
    rew = jnp.roll(data[:, idx], 1, axis=0)
    rew = rew.at[0].set(0.)
    
    val = final_rew - rew
    return data.at[:, idx].set(val)


def initialize_replay_buffer(buf, model, size, key):
	print("Initializing replay buffer...")
	batchsize = size // args.iterations + 1
	keys = jrand.split(key, batchsize)
	init_carry = (batched_reset(keys), jnp.full(batchsize, 0.), key)

	transitions = eqx.filter_jit(environment_interaction)(model, batchsize, init_carry) # 
	transitions = preprocess_data(transitions, -2)
	transitions = transitions.reshape(-1, transitions.shape[-1])

	for i in range(size):
		buf.push(transitions[i])

	return buf


optim = optax.adam(args.lr)
opt_state = optim.init(eqx.filter(NN, eqx.is_inexact_array))


def train_agent(samples, network, opt_state, key):
	# compute loss gradient compared to tree search targets and update parameters
	key, subkey = jrand.split(key, 2)
	subkeys = jrand.split(subkey, samples.shape[0])
	obs = samples[:, 0:15*15].reshape(-1, 1, 15, 15)
	search_policy = samples[:, 15*15:15*15+args.num_actions]
	search_value = samples[:, -2]

	loss, grads = eqx.filter_value_and_grad(A0_loss)(network, 
                                                	search_policy, 
                                                	search_value, 
                                                	obs,
													args.regularization,
                                                	subkeys)

	updates, opt_state = optim.update(grads, opt_state)

	network = eqx.apply_updates(network, updates)

	return loss, network, opt_state


def differentiate(network):
	state = env.reset(key)

	rew = 0
	solved = False
	while not solved:
		obs = state.edges[jnp.newaxis, :, :]
		value = network(obs)[0]
		logits = network(obs)[1:]
		
		masked_logits = get_masked_logits(logits, state, NUM_INTERMEDIATES)
  
		action = jnp.argmax(masked_logits).astype(jnp.int32)
		print(state.state, action+1, value, jnn.softmax(masked_logits))
		state, reward, terminated = env.step(state, action)
		rew += reward
		solved = terminated
	return rew

replay_buffer = initialize_replay_buffer(replay_buffer, NN, args.replay_size, key)

pbar = tqdm(range(args.episodes))
for e in pbar:
	# Data wrangling
	keys = jrand.split(key, args.rollout_batchsize)
	init_carry = (batched_reset(keys), jnp.full(args.rollout_batchsize, 0.), key)
 
	data = eqx.filter_jit(environment_interaction)(NN, args.rollout_batchsize, init_carry)
	data = preprocess_data(data, -2)
	data = data.reshape(-1, data.shape[-1])
 
	for i in range(data.shape[0]):
		replay_buffer.push(data[i])

	samples = replay_buffer.sample(args.batchsize)

	train_key, key = jrand.split(key)
	loss, NN, opt_state = eqx.filter_jit(train_agent)(samples, NN, opt_state, train_key)
	rew = differentiate(NN)
	wandb.log({"loss": loss, "# computations": -rew})
	pbar.set_description(f"episode: {e}, loss: {loss}, return: {rew}")

