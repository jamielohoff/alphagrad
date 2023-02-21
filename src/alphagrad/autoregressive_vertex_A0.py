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

from graphax import GraphInfo, VertexGameState, VertexGame, make_vertex_game_state
from graphax.core import forward, reverse
from graphax.examples import construct_random, \
							construct_Helmholtz, \
							construct_LIF

from alphagrad.utils import A0_loss, get_masked_logits, preprocess_data
from alphagrad.data import VertexGameGenerator, \
							AlphaGradReplayBuffer, \
							make_recurrent_fn, \
							make_environment_interaction


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="Vertex_A0_test", help="Name of the experiment.")

parser.add_argument("--gpu", type=str, 
                    default="0", help="GPU identifier.")

parser.add_argument("--seed", type=int,
                    default=42, help="Random seed.")

parser.add_argument("--episodes", type=int, 
                    default=2500, help="Number of runs on random data.")

parser.add_argument("--replay_size", type=int, 
                    default=4096, help="Size of the replay buffer.")

parser.add_argument("--num_actions", type=int, 
                    default=11, help="Number of actions.")

parser.add_argument("--num_simulations", type=int, 
                    default=25, help="Number of simulations.")

parser.add_argument("--batchsize", type=int, 
                    default=256, help="Learning batchsize.")

parser.add_argument("--rollout_batchsize", type=int, default=16, 
                    help="Batchsize for environment interaction.")

parser.add_argument("--regularization", type=float, 
                    default=0., help="Contribution of L2 regularization.")

parser.add_argument("--lr", type=float, 
                    default=3e-3, help="Learning rate.")

parser.add_argument("--num_inputs", type=int, 
                    default=4, help="Number input variables.")

parser.add_argument("--num_outputs", type=int, 
                    default=4, help="Number of output variables.")

args = parser.parse_args()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# wandb.init("Vertex_AlphaZero")
# wandb.run.name = args.name
# wandb.config = vars(args)

NUM_INPUTS = args.num_inputs
NUM_INTERMEDIATES = args.num_actions
NUM_OUTPUTS = args.num_outputs
NUM_GAMES = 64
SHAPE = (NUM_INPUTS+NUM_INTERMEDIATES, NUM_INTERMEDIATES+NUM_OUTPUTS)

key = jrand.PRNGKey(args.seed)
edges, INFO = construct_Helmholtz() # construct_LIF() # 
state = make_vertex_game_state(INFO, edges)
env = VertexGame(state)
buf = AlphaGradReplayBuffer(args.replay_size, INFO)

forward_edges = copy.deepcopy(edges)
_, ops = forward(forward_edges, INFO)
print("forward-mode:", ops)


reverse_edges = copy.deepcopy(edges)
_, ops = reverse(reverse_edges, INFO)
print("reverse-mode:", ops)


nn_key, key = jrand.split(key, 2)
subkeys = jrand.split(nn_key, 4)


MODEL = AlphaGradModel((4, 11, 4), 128, 16, 9, 3, 6, 512, key=key)


batched_step = jax.vmap(env.step)
batched_reset = jax.vmap(env.reset)
batched_one_hot = jax.vmap(jnn.one_hot, in_axes=(0, None))
batched_get_masked_logits = jax.vmap(get_masked_logits, in_axes=(0, 0, None))


recurrent_fn = make_recurrent_fn(MODEL, 
                                NUM_INTERMEDIATES, 
                                batched_step, 
                                batched_get_masked_logits)


env_interaction = make_environment_interaction(NUM_INTERMEDIATES, 
                                            	args.num_simulations,
                                                recurrent_fn,
                                                batched_step,
                                                batched_one_hot)

game_generator = VertexGameGenerator(NUM_GAMES, INFO, key)

def initialize_replay_buffer(buf, model, size, key):
	print("Initializing replay buffer...")
	batchsize = size // NUM_INTERMEDIATES + 1
	batch_games = game_generator(batchsize, key)
	init_carry = (batch_games, jnp.full(batchsize, 0.), key) # needs editing

	transitions = eqx.filter_jit(env_interaction)(model, batchsize, init_carry)
	transitions = preprocess_data(transitions, -2)
	transitions = transitions.reshape(-1, transitions.shape[-1])

	for i in range(size):
		buf.push(transitions[i])

	return buf

buf = initialize_replay_buffer(buf, MODEL, args.replay_size, key)

optim = optax.adam(args.lr)
opt_state = optim.init(eqx.filter(NN, eqx.is_inexact_array))


# def train_agent(samples, network, opt_state, key):
# 	# compute loss gradient compared to tree search targets and update parameters
# 	key, subkey = jrand.split(key, 2)
# 	subkeys = jrand.split(subkey, samples.shape[0])
# 	end_obs = SHAPE[0]*SHAPE[1]
# 	obs = samples[:, 0:end_obs].reshape(-1, 1, SHAPE[0], SHAPE[1])
# 	search_policy = samples[:, end_obs:end_obs+args.num_actions]
# 	search_value = samples[:, -2]

# 	loss, grads = eqx.filter_value_and_grad(A0_loss)(network, 
#                                                 	search_policy, 
#                                                 	search_value, 
#                                                 	obs,
# 													args.regularization,
#                                                 	subkeys)

# 	updates, opt_state = optim.update(grads, opt_state)

# 	network = eqx.apply_updates(network, updates)

# 	return loss, network, opt_state


# def differentiate(network):
# 	state = env.reset(key)

# 	rew = 0
# 	solved = False
# 	while not solved:
# 		obs = state.edges[jnp.newaxis, :, :]
# 		value = network(obs)[0]
# 		logits = network(obs)[1:]
		
# 		masked_logits = get_masked_logits(logits, state, NUM_INTERMEDIATES)
  
# 		action = jnp.argmax(masked_logits).astype(jnp.int32)
# 		print(state.state, action+1, value, jnn.softmax(masked_logits))
# 		state, reward, terminated = env.step(state, action)
# 		rew += reward
# 		solved = terminated
# 	return rew

# replay_buffer = initialize_replay_buffer(replay_buffer, NN, args.replay_size, key)

# pbar = tqdm(range(args.episodes))
# rewards = []
# for e in pbar:
# 	# Data wrangling
# 	batch_gs = get_batch(args.rollout_batchsize, GAMES, key)

# 	init_carry = (batch_gs, jnp.full(args.rollout_batchsize, 0.), key)

# 	data = eqx.filter_jit(environment_interaction)(NN, args.rollout_batchsize, init_carry)
# 	data = preprocess_data(data, -2)
# 	data = data.reshape(-1, data.shape[-1])

# 	for i in range(data.shape[0]):
# 		replay_buffer.push(data[i])

# 	samples = replay_buffer.sample(args.batchsize)

# 	train_key, key = jrand.split(key)
# 	loss, NN, opt_state = eqx.filter_jit(train_agent)(samples, NN, opt_state, train_key)
# 	rew = differentiate(NN)
# 	print(rew, loss)
# 	# wandb.log({"loss": loss.tolist(), "# computations": -rew.tolist()})
# 	rewards.append(-rew.tolist())
# 	pbar.set_description(f"episode: {e}, loss: {loss}, return: {rew}")

