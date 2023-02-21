import os
import wandb
import argparse
from tqdm import tqdm

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import mctx
import optax
import equinox as eqx


from gridworld import GridworldGame2D, GridworldState
from replay_memory import MCTSReplayMemory

from alphazero import preprocess_data, A0_loss


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, 
                    default="Maze_A0_test", help="Name of the experiment.")

parser.add_argument("--gpu", type=str, 
                    default="0", help="GPU identifier.")

parser.add_argument("--seed", type=int,
                    default=123, help="Random seed.")

parser.add_argument("--episodes", type=int, 
                    default=500, help="Number of runs on random data.")

parser.add_argument("--replay_size", type=int, 
                    default=2048, help="Size of the replay buffer.")

parser.add_argument("--num_actions", type=int, 
                    default=4, help="Number of actions.")

parser.add_argument("--num_simulations", type=int, 
                    default=50, help="Number of simulations.")

parser.add_argument("--iterations", type=int, 
                    default=50, help="Number of iterations for each MCTS.")

parser.add_argument("--rollout_length", type=int, default=50, 
                    help="Duration of the rollout phase of the MCTS algorithm.")

parser.add_argument("--batchsize", type=int, 
                    default=256, help="Learning batchsize.")

parser.add_argument("--rollout_batchsize", type=int,
                    default=16, 
                    help="Batchsize for environment interaction.")

parser.add_argument("--value_mixing", type=float, default=1., 
                   help="Mixing between predicted value and MCTS result.")

parser.add_argument("--regularization", type=float, 
                    default=0., help="Contribution of L2_regularization.")

parser.add_argument("--lr", type=float, 
                    default=1e-2, help="Learning rate.")

args = parser.parse_args()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

wandb.init("Maze_AlphaZero")
wandb.run.name = args.name
wandb.config = vars(args)

key = jrand.PRNGKey(args.seed)
# goal = jnp.array([0, 4])
# walls = jnp.array( [[0, 1, 0, 1, 0],
#                     [0, 1, 0, 1, 0],
#                     [0, 0, 0, 1, 0],
#                     [0, 1, 0, 1, 0],
#                     [0, 1, 0, 0, 0]])


goal = jnp.array([9, 9])
walls = 1. - jnp.array([[ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
						[ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
						[ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
						[ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.],
						[ 1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.],
						[ 1.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
						[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
						[ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
						[ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
						[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.]])



replay_buffer = MCTSReplayMemory(args.replay_size)
env = GridworldGame2D(walls, goal, max_steps=args.iterations+1)

nn_key, key = jrand.split(key, 2)
NN = eqx.nn.MLP(100, 5, 128, 3, key=nn_key)

batched_step = jax.vmap(env.step)
batched_reset = jax.vmap(env.reset)


def rollout(policy_network, init_states, init_obs, key):
	def loop_fn(carry, _):
		state, obs, key = carry
		subkey, key = jrand.split(key, 2)
		output = policy_network(obs)
		policy_logits = jnp.ones(4) # output[:, 1:]
		actions = jrand.categorical(subkey, policy_logits)
		next_state, next_obs, reward, _ = batched_step(state, actions)

		return (next_state, next_obs, key), reward

	init_carry = (init_states, init_obs, key)
	_, rewards = lax.scan(loop_fn, init_carry, None, length=args.rollout_length)
	return rewards.sum(axis=0)


def recurrent_fn(params, rng_key, actions, state):
	bs, nn_params = params
	next_state, next_obs, reward, _ = batched_step(state, actions) # dynamics function
 
	# prediction function
	network = jtu.tree_map(lambda x, y: y if eqx.is_inexact_array(x) else x, NN, nn_params)
	batch_network = jax.vmap(network)
 
	output = batch_network(next_obs)
	policy_logits = output[:, 1:]
	nn_value = output[:, 0] 
 
	rollout_value = rollout(batch_network, next_state, next_obs, rng_key)
	mixing = args.value_mixing
	value = mixing*rollout_value + (1. - args.value_mixing)*nn_value

	# On a single-player environment, use discount from [0, 1].
	discount = jnp.ones(bs)
	recurrent_fn_output = mctx.RecurrentFnOutput(reward=reward,
												discount=discount,
												prior_logits=policy_logits,
												value=value)
	return recurrent_fn_output, next_state


def scan(f, init, xs, length=None):
	if xs is None:
		xs = [None] * length
	carry = init
	ys = []
	for x in xs:
		carry, y = f(carry, x)
		ys.append(y)
	return carry, ys


def environment_interaction(network, batchsize, init_carry):
	batched_network = jax.vmap(network)
	nn_params = eqx.filter(network, eqx.is_inexact_array)
	params = (batchsize, nn_params)
	def loop_fn(carry, _):
		state, rews, key = carry
		obs = state.moves.reshape(batchsize, -1)
		output = batched_network(obs)
		policy_logits = output[:, 1:]
		values = output[:, 0]

		root = mctx.RootFnOutput(prior_logits=policy_logits,
								value=values,
								embedding=state)

		key, subkey = jrand.split(key, 2)

		policy_output = mctx.muzero_policy(params=params,
											rng_key=subkey,
											root=root,
											recurrent_fn=recurrent_fn,
											dirichlet_fraction=0.25,
											num_simulations=args.num_simulations)

		# tree search derived targets for policy and value function
		search_policies = policy_output.action_weights

		# always take action recommended by tree search
		actions = policy_output.action

		# step the environment
		next_state, _, rewards, done = batched_step(state, actions)
		rews += rewards

		return (next_state, rews, key), jnp.concatenate([rews[:, jnp.newaxis], search_policies, done[:, jnp.newaxis], obs], axis=1)

	_, output = lax.scan(loop_fn, init_carry, None, length=args.iterations)

	return jnp.stack(output).transpose(1, 0, 2)


def initialize_replay_buffer(buf, model, size, key):
	print("Initializing replay buffer...")
	bs = size // args.iterations + 1
	keys = jrand.split(key, bs)
	init_carry = (batched_reset(keys), jnp.full(bs, 0.), key)
	transitions = eqx.filter_jit(environment_interaction)(model, bs, init_carry)
	transitions = preprocess_data(transitions, 0)
	transitions = transitions.reshape(-1, transitions.shape[-1])
	transitions = transitions[transitions[:, 5] == 0]

	for i in range(size):
		buf.push(transitions[i])
	return buf
   

optim = optax.adam(args.lr)
opt_state = optim.init(eqx.filter(NN, eqx.is_inexact_array))


def train_agent(samples, network, opt_state, key):
	# compute loss gradient compared to tree search targets and update parameters
	key, subkey = jrand.split(key, 2)
	subkeys = jrand.split(subkey, samples.shape[0])

	search_value = samples[:, 0]
	search_policy = samples[:, 1:5]
	obs = samples[:, 6:]

	loss, grads = eqx.filter_value_and_grad(A0_loss)(network, 
                                                	search_policy, 
                                                	search_value, 
                                                	obs, 
                                                	args.regularization, 
                                                	subkeys)

	updates, opt_state = optim.update(grads, opt_state)

	network = eqx.apply_updates(network, updates)

	return loss, network, opt_state


def solve_maze(network, starting_point):
	moves = jnp.zeros_like(walls)
	moves = moves.at[starting_point[0], starting_point[1]].set(1.)
	state = GridworldState(t=0, position=starting_point, moves=moves)
	obs = state.moves.reshape(-1)

	rew = 0
	counter = 0
	for _ in range(50):
		value = network(obs)[0]
		logits = network(obs)[1:]
		action = jnp.argmax(jnn.softmax(logits)).astype(jnp.int32)
		print(state.position, action, value, jnn.softmax(logits))
		state, obs, reward, done = env.step(state, action)
		rew += reward
		if done:
			break
		counter += 1
	return rew, counter

replay_buffer = initialize_replay_buffer(replay_buffer, NN, args.replay_size, key)

print("Starting Training...")
pbar = tqdm(range(args.episodes))
for e in pbar:
	# Data wrangling
	keys = jrand.split(key, args.rollout_batchsize)
	init_carry = (batched_reset(keys), jnp.full(args.rollout_batchsize, 0.), key)

	data = eqx.filter_jit(environment_interaction)(NN, args.rollout_batchsize, init_carry) # generate_data(NN, ROLLOUT_BATCHSIZE, init_carry) #
	data = preprocess_data(data, 0)
	data = data.reshape(-1, data.shape[-1])
	data = data[data[:, 5] == 0]

	for i in range(data.shape[0]):
		replay_buffer.push(data[i])

	samples = replay_buffer.sample(args.batchsize)

	train_key, key = jrand.split(key)
	loss, NN, opt_state = eqx.filter_jit(train_agent)(samples, NN, opt_state, train_key) # train_agent(samples, NN, opt_state, train_key) # 
	
	rew, counter = solve_maze(NN, jnp.array([0, 0]))
	wandb.log({"loss": loss, "return": rew, "counter": counter})
	pbar.set_description(f"episode: {e}, loss: {loss}, return: {rew}")

