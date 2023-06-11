import os
import argparse
import wandb
import time

from torch.utils.data import DataLoader

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import optax
import equinox as eqx

from tqdm import tqdm
from functools import partial

import sys
jnp.set_printoptions(threshold=sys.maxsize)

from graphax import VertexGame, make_graph_info
from graphax.dataset import GraphDataset

from alphagrad.utils import (A0_loss,
    						get_masked_logits,
              				make_init_state,
                      		postprocess_data,
                        	make_batched_vertex_game)
from alphagrad.data import (make_recurrent_fn,
							make_environment_interaction)

from alphagrad.sequential_transformer import SequentialTransformerModel
from alphagrad.differentiate import differentiate

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="Vertex_A0_test", help="Name of the experiment.")

parser.add_argument("--gpus", type=str, 
                    default="0,1,2,3", help="GPU identifier.")

parser.add_argument("--seed", type=int,
                    default=1337, help="Random seed.")

parser.add_argument("--num_episodes", type=int, 
                    default=1000, help="Number of runs on random data.")

parser.add_argument("--num_simulations", type=int, 
                    default=100, help="Number of simulations.")

parser.add_argument("--batchsize", type=int, 
                    default=768, help="Learning batchsize.")

parser.add_argument("--policy_weight", type=float, 
                    default=.1, help="Contribution of policy.")

parser.add_argument("--L2_weight", type=float, 
                    default=1e-5, help="Contribution of L2 regularization.")

parser.add_argument("--lr", type=float, 
                    default=1e-4, help="Learning rate.")

parser.add_argument("--num_inputs", type=int, 
                    default=10, help="Number input variables.")

parser.add_argument("--num_actions", type=int, 
                    default=30, help="Number of actions.")

parser.add_argument("--num_outputs", type=int, 
                    default=5, help="Number of output variables.")

parser.add_argument("--disable_wandb", action="store_const", const=True,
                    default=False, help="Use wandb for logging or not.")

parser.add_argument("--test_frequency", type=int,
                    default=5, help="The frequency with which we test the current policy.")

parser.add_argument("--file_path", type=str,
                    default="./data/large_random_samples", help="Path to the dataset.")

parser.add_argument("--test_file_path", type=str,
                    default="./data/test_samples", help="Path to the test dataset.")

parser.add_argument("--num_layers", type=int,
                    default=12, help="Number of transformer blocks.")

parser.add_argument("--num_heads", type=int,
                    default=2, help="Number of attention heads.")

parser.add_argument("--embedding_dim", type=int,
                    default=34, help="Dimension of the token embeddings.")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpu_devices = jax.devices("gpu")
print("gpu", gpu_devices)

if not args.disable_wandb:
	wandb.init("Vertex_AlphaZero")
	wandb.run.name = args.name
	wandb.config = vars(args)

BS = args.batchsize*args.num_actions // len(gpu_devices)
NUM_INPUTS = args.num_inputs
NUM_INTERMEDIATES = args.num_actions
NUM_OUTPUTS = args.num_outputs
SHAPE = (NUM_INPUTS+NUM_INTERMEDIATES, NUM_INTERMEDIATES+NUM_OUTPUTS)

key = jrand.PRNGKey(args.seed)
INFO = make_graph_info([NUM_INPUTS, NUM_INTERMEDIATES, NUM_OUTPUTS])
env = VertexGame(INFO)

nn_key, key = jrand.split(key, 2)
MODEL = SequentialTransformerModel(INFO, args.num_layers, args.num_heads, kernel_size=7, stride=1, key=nn_key)

batched_step = jax.vmap(env.step)
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

diff_env_interaction = make_environment_interaction(INFO, 
													args.num_simulations,
													recurrent_fn,
													batched_step,
													batched_one_hot,
													temperature=0)


graph_dataset = GraphDataset(args.file_path)
test_graph_dataset = GraphDataset(args.test_file_path, include_code=True)
dataloader = DataLoader(graph_dataset, batch_size=args.batchsize, shuffle=True, num_workers=1, drop_last=True)
benchmark_dataloader = DataLoader(test_graph_dataset, batch_size=8, shuffle=False, num_workers=1)

optim = optax.adamw(learning_rate=args.lr, weight_decay=1e-4) # optax.clip_by_global_norm(1.))
opt_state = optim.init(eqx.filter(MODEL, eqx.is_inexact_array))


### needed to reassemble data
edges_shape = (NUM_INPUTS+NUM_INTERMEDIATES, NUM_INTERMEDIATES)
obs_idx = jnp.prod(jnp.array(edges_shape))
mask_idx = obs_idx + NUM_INTERMEDIATES**2
policy_idx = mask_idx + NUM_INTERMEDIATES
reward_idx = policy_idx + 1
split_idxs = (obs_idx, mask_idx, policy_idx, reward_idx)


def tree_search(games, network, key):
	init_carry = make_init_state(games, NUM_INTERMEDIATES, key)
	data = env_interaction(network, init_carry)
	return postprocess_data(data)

select_first = lambda x: x[0] if isinstance(x, jax.Array) else x
parallel_mean = lambda x: lax.pmean(x, "num_devices")

# TODO pretrain the transformer model!!!!!!!!! 
@partial(eqx.filter_pmap, in_axes=(0, None, None, None), axis_name="num_devices")
def train_agent(games, network, opt_state, key):
	data = tree_search(games, network, key)

	obs, attn_masks, search_policy, search_value, _ = jnp.split(data, split_idxs, axis=-1)
	attn_masks = attn_masks.reshape(BS, NUM_INTERMEDIATES, NUM_INTERMEDIATES)
	search_policy = search_policy.reshape(BS, NUM_INTERMEDIATES)
	search_value = search_value.reshape(BS, 1)
	obs = obs.reshape(BS, *edges_shape)
 
	subkeys = jrand.split(key, obs.shape[0])
	val, grads = eqx.filter_value_and_grad(A0_loss, has_aux=True)(network, 
																	search_policy, 
																	search_value, 
																	obs,
																	attn_masks,
																	args.policy_weight,
																	args.L2_weight,
																	subkeys)
	loss, aux = val
	loss = lax.pmean(loss, axis_name="num_devices")
	aux = lax.pmean(aux, axis_name="num_devices")
	grads = jtu.tree_map(parallel_mean, grads)

	updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(network, eqx.is_inexact_array))
	network = eqx.apply_updates(network, updates)
	return loss, aux, network, opt_state, search_value, obs

pbar = tqdm(range(args.num_episodes))
rewards = []
counter = 0
for e in pbar:
	for edges, info, vertices, attn_masks in tqdm(dataloader):
		data_key, env_key, train_key, key = jrand.split(key, 4)
		games = make_batched_vertex_game(edges, vertices, attn_masks, num_devices=len(gpu_devices))

		start_time = time.time()
		losses, aux, models, opt_states, search_value, obs = train_agent(games, MODEL, opt_state, train_key)	
		print("train", time.time() - start_time)

		loss = losses[0]
		aux = aux[0]
		MODEL = jtu.tree_map(select_first, models, is_leaf=eqx.is_inexact_array)
		opt_state = jtu.tree_map(select_first, opt_states, is_leaf=eqx.is_inexact_array)

		if not args.disable_wandb:
			wandb.log({"loss": loss.tolist(),
						"policy_loss": aux[0].tolist(),
						"value_loss": aux[1].tolist(),
						"L2_reg": aux[2].tolist()})

		if counter % args.test_frequency == 0:
			start_time = time.time()
			for names, edges, info, vertices, attn_masks in test_graph_dataset:
				benchmark_games = make_batched_vertex_game(edges, vertices, attn_masks, num_devices=1)
				# possibly pmap this?
				rews = eqx.filter_jit(differentiate)(MODEL, NUM_INTERMEDIATES, diff_env_interaction, key, benchmark_games)
				print("diff", time.time() - start_time)

				if not args.disable_wandb:
					num_computations = {name:-nops for name, nops in zip(names, rews.tolist())}
					wandb.log(num_computations)
		counter += 1
		pbar.set_description(f"loss: {loss}, return: {rews}")

