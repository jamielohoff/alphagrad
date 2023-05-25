import os
import argparse
import wandb
import time

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


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="Vertex_A0_test", help="Name of the experiment.")

parser.add_argument("--num_games", type=int, 
                    default=25000, help="Number of example games to be played.")

parser.add_argument("--gpus", type=str, 
                    default="0,1,2,3", help="GPU identifier.")

parser.add_argument("--seed", type=int,
                    default=1337, help="Random seed.")

parser.add_argument("--num_episodes", type=int, 
                    default=100000, help="Number of runs on random data.")

parser.add_argument("--num_simulations", type=int, 
                    default=100, help="Number of simulations.")

parser.add_argument("--batchsize", type=int, 
                    default=512, help="Learning batchsize.")

parser.add_argument("--policy_weight", type=float, 
                    default=3.5, help="Contribution of policy.")

parser.add_argument("--L2_weight", type=float, 
                    default=1e-5, help="Contribution of L2 regularization.")

parser.add_argument("--lr", type=float, 
                    default=3e-4, help="Learning rate.")

parser.add_argument("--num_inputs", type=int, 
                    default=10, help="Number input variables.")

parser.add_argument("--num_actions", type=int, 
                    default=30, help="Number of actions.")

parser.add_argument("--num_outputs", type=int, 
                    default=5, help="Number of output variables.")

parser.add_argument("--test_frequency", type=int,
                    default=5, help="The frequency with which we test the current policy.")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpu_devices = jax.devices("gpu")
print("gpu", gpu_devices)

from graphax import VertexGame, make_vertex_game_state, make_graph_info, embed
from graphax.examples import make_Helmholtz

from alphagrad.utils import (A0_loss,
    						get_masked_logits,
              				preprocess_data,
                      		postprocess_data)
from alphagrad.data import (VertexGameGenerator,
							make_recurrent_fn,
							make_environment_interaction,
           					make_benchmark_games)

from alphagrad.axial_transformer import AxialTransformerModel
from alphagrad.resnet import ResNet34
from alphagrad.sequential_transformer import SequentialTransformerModel
from alphagrad.differentiate import differentiate

wandb.init("Vertex_AlphaZero")
wandb.run.name = args.name
wandb.config = vars(args)

BS = args.batchsize*args.num_actions // len(gpu_devices)
NUM_INPUTS = args.num_inputs
NUM_INTERMEDIATES = args.num_actions
NUM_OUTPUTS = args.num_outputs
NUM_GAMES = args.num_games
SHAPE = (NUM_INPUTS+NUM_INTERMEDIATES, NUM_INTERMEDIATES+NUM_OUTPUTS)

key = jrand.PRNGKey(args.seed)
INFO = make_graph_info([NUM_INPUTS, NUM_INTERMEDIATES, NUM_OUTPUTS])
edges, info = make_Helmholtz()
edges, _ = embed(key, edges, info, INFO)
state = make_vertex_game_state(edges, INFO)

env = VertexGame(state)
benchmark_games, names = make_benchmark_games(key, INFO)

# edges = benchmark_games[-2].edges
# edges, nops = forward_gpu(edges, INFO)
# print("random", nops)

# edges = benchmark_games[-2].edges
# edges, nops = reverse_gpu(edges, INFO)
# print("random", nops)

# edges = benchmark_games[-1].edges
# edges, nops = forward_gpu(edges, INFO)
# print("random funnel", nops)

# edges = benchmark_games[-1].edges
# edges, nops = reverse_gpu(edges, INFO)
# print("random funnel", nops)

nn_key, key = jrand.split(key, 2)
MODEL = SequentialTransformerModel(INFO, 3, 2, key=nn_key)

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

diff_env_interaction = make_environment_interaction(INFO, 
													25,
													recurrent_fn,
													batched_step,
													batched_one_hot,
													temperature=0)


game_generator = VertexGameGenerator(NUM_GAMES, INFO, key)
optim = optax.adamw(learning_rate=args.lr, weight_decay=1e-4)
opt_state = optim.init(eqx.filter(MODEL, eqx.is_inexact_array))

### needed to reassemble data
edges_shape = (NUM_INPUTS+NUM_INTERMEDIATES, NUM_INTERMEDIATES+NUM_OUTPUTS)
obs_idx = jnp.prod(jnp.array(edges_shape))
policy_idx = obs_idx + NUM_INTERMEDIATES
reward_idx = policy_idx + 1
split_idxs = (obs_idx, policy_idx, reward_idx)


def tree_search(games, attn_masks, network, key):
	init_carry = preprocess_data(games, key)
	data = env_interaction(network, attn_masks, init_carry)
	return postprocess_data(data)

select_first = lambda x: x[0] if isinstance(x, jax.Array) else x
parallel_mean = lambda x: lax.pmean(x, "num_devices")

# TODO pretrain the transformer model!!!!!!!!! 

@partial(eqx.filter_pmap, 
        in_axes=(0, 0, None, None, None),
        axis_name="num_devices")
def train_agent(games, attn_masks, network, opt_state, key):
	data = tree_search(games, attn_masks, network, key)
	obs, search_policy, search_value, _ = jnp.split(data, split_idxs, axis=-1)
	search_policy = search_policy.reshape(BS, NUM_INTERMEDIATES)
	search_value = search_value.reshape(BS, 1)
	obs = obs.reshape(BS, *edges_shape)

	subkeys = jrand.split(key, obs.shape[0])
	val, grads = eqx.filter_value_and_grad(A0_loss, has_aux=True)(network, 
																	search_policy, 
																	search_value, 
																	obs,
																	args.policy_weight,
																	args.L2_weight,
																	subkeys)
	loss, aux = val
	loss = lax.pmean(loss, axis_name="num_devices")
	aux = lax.pmean(aux, axis_name="num_devices")
	grads = jtu.tree_map(parallel_mean, grads)

	updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(network, eqx.is_inexact_array))
	network = eqx.apply_updates(network, updates)
	return loss, aux, network, opt_state


pbar = tqdm(range(args.num_episodes))
rewards = []
for e in pbar:
	data_key, env_key, train_key, key = jrand.split(key, 4)
	games, attn_masks = game_generator(args.batchsize, 
										num_devices=jax.local_device_count(), 
										key=data_key)

	start_time = time.time()
	losses, aux, models, opt_states = train_agent(games, attn_masks, MODEL, opt_state, train_key)	
	print("train", time.time() - start_time)

	loss = losses[0]
	aux = aux[0]
	MODEL = jtu.tree_map(select_first, models, is_leaf=eqx.is_inexact_array)
	opt_state = jtu.tree_map(select_first, opt_states, is_leaf=eqx.is_inexact_array)

	wandb.log({"loss": loss.tolist(),
            	"policy_loss": aux[0].tolist(),
             	"value_loss": aux[1].tolist(),
              	"L2_reg": aux[2].tolist()})

	if e % args.test_frequency == 0:
		print(e)
		start_time = time.time()
		rews = eqx.filter_jit(differentiate)(MODEL, diff_env_interaction, key, *benchmark_games)
		print(rews)
		print("diff", time.time() - start_time)
 
		num_computations = {name:-nops for name, nops in zip(names, rews.tolist())}
		wandb.log(num_computations)
	pbar.set_description(f"loss: {loss}, return: {rews}")

