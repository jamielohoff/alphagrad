import os
import copy
import argparse
import wandb

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import optax
import equinox as eqx

from tqdm import tqdm

from graphax import VertexGame, make_vertex_game_state
from graphax.core import forward, reverse
from graphax.examples import construct_random, \
							construct_Helmholtz, \
							construct_LIF

from alphagrad.utils import A0_loss, get_masked_logits, preprocess_data
from alphagrad.data import VertexGameGenerator, \
							make_recurrent_fn, \
							make_environment_interaction
from alphagrad.modelzoo import TransformerModel, CNNModel
from alphagrad.differentiate import differentiate


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="Vertex_A0_test", help="Name of the experiment.")

parser.add_argument("--gpu", type=str, 
                    default="1", help="GPU identifier.")

parser.add_argument("--seed", type=int,
                    default=1337, help="Random seed.")

parser.add_argument("--episodes", type=int, 
                    default=2500, help="Number of runs on random data.")

parser.add_argument("--num_actions", type=int, 
                    default=11, help="Number of actions.")

parser.add_argument("--num_simulations", type=int, 
                    default=25, help="Number of simulations.")

parser.add_argument("--batchsize", type=int, 
                    default=32, help="Learning batchsize.")

parser.add_argument("--regularization", type=float, 
                    default=0., help="Contribution of L2 regularization.")

parser.add_argument("--lr", type=float, 
                    default=2e-4, help="Learning rate.")

parser.add_argument("--num_inputs", type=int, 
                    default=4, help="Number input variables.")

parser.add_argument("--num_outputs", type=int, 
                    default=4, help="Number of output variables.")

args = parser.parse_args()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
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


nn_key, key = jrand.split(key, 2)
subkeys = jrand.split(nn_key, 4)
MODEL = TransformerModel(INFO, 64, 32, 7, 1, 6, key=key) # CNNModel(INFO, 7, key=key) # 


batched_step = jax.vmap(env.step)
batched_reset = jax.vmap(env.reset)
batched_one_hot = jax.vmap(jnn.one_hot, in_axes=(0, None))
batched_get_masked_logits = jax.vmap(get_masked_logits, in_axes=(0, 0, None))


recurrent_fn = make_recurrent_fn(MODEL, 
                                INFO, 
                                batched_step, 
                                batched_get_masked_logits)


env_interaction = make_environment_interaction(INFO, 
                                            	args.num_simulations,
                                                recurrent_fn,
                                                batched_step,
                                                batched_one_hot,
												temperature=0)


game_generator = VertexGameGenerator(NUM_GAMES, INFO, key)
optim = optax.adam(args.lr)
opt_state = optim.init(eqx.filter(MODEL, eqx.is_inexact_array))

### needed to reassemble data
num_i = INFO.num_inputs
num_v = INFO.num_intermediates
num_o = INFO.num_outputs
edges_shape = (num_i+num_v, num_v+num_o)
obs_idx = jnp.prod(jnp.array(edges_shape))
policy_idx = obs_idx + num_v
reward_idx = policy_idx + 1
split_idxs = (obs_idx, policy_idx, reward_idx)


def train_agent(batch_games, network, opt_state, key):
	# compute loss gradient compared to tree search targets and update parameters
	init_carry = (batch_games, jnp.zeros(args.batchsize), key)
	data = env_interaction(network, args.batchsize, init_carry)
	data = preprocess_data(data, -2)

	obs, search_policy, search_value, _ = jnp.split(data, split_idxs, axis=-1)
	batchsize = args.batchsize*num_v
	search_policy = search_policy.reshape(batchsize, num_v)
	search_value = search_value.reshape(batchsize, 1)
	obs = obs.reshape(batchsize, *edges_shape)

	key, subkey = jrand.split(key, 2)
	subkeys = jrand.split(subkey, obs.shape[0])
	loss, grads = eqx.filter_value_and_grad(A0_loss)(network, 
                                                	search_policy, 
                                                	search_value, 
                                                	obs,
													args.regularization,
                                                	subkeys)
	updates, opt_state = optim.update(grads, opt_state)
	network = eqx.apply_updates(network, updates)
	return loss, network, opt_state


edges, info = construct_Helmholtz()
helmholtz_game = make_vertex_game_state(info, edges)

gkey, key = jrand.split(key)
edges, info = construct_random(gkey, INFO, fraction=0.5)
random_game = make_vertex_game_state(info, edges)


forward_edges = copy.deepcopy(edges)
_, ops = forward(forward_edges, info)
print("forward-mode:", ops)


reverse_edges = copy.deepcopy(edges)
_, ops = reverse(reverse_edges, info)
print("reverse-mode:", ops)

import time
pbar = tqdm(range(args.episodes))
rewards = []
for e in pbar:
	# Data wrangling
	batch_games = game_generator(args.batchsize, key)

	train_key, key = jrand.split(key)
	start_time = time.time()
	loss, MODEL, opt_state = eqx.filter_jit(train_agent)(batch_games, MODEL, opt_state, train_key)
	print(time.time() - start_time)
 
	start_time = time.time()
	rew = differentiate(MODEL, env_interaction, key, random_game, helmholtz_game)
	print(time.time() - start_time)
	# wandb.log({"loss": loss.tolist(), "# computations": rew[0].tolist()})
	pbar.set_description(f"loss: {loss}, return: {rew}")

