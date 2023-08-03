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


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="AlphaGrad_Test", help="Name of the experiment.")

parser.add_argument("--gpus", type=str, 
                    default="0,1,2,3", help="GPU identifier.")

parser.add_argument("--seed", type=int,
                    default=42, help="Random seed.")

parser.add_argument("--num_episodes", type=int, 
                    default=1000, help="Number of runs on random data.")

parser.add_argument("--num_simulations", type=int, 
                    default=50, help="Number of simulations.")

parser.add_argument("--batchsize", type=int, 
                    default=56, help="Learning batchsize.")

parser.add_argument("--policy_weight", type=float, 
                    default=.075, help="Contribution of policy.")

parser.add_argument("--L2_weight", type=float, 
                    default=1e-6, help="Contribution of L2 regularization.")

parser.add_argument("--entropy_weight", type=float, 
                    default=.1, help="Contribution of entropy regularization.")

parser.add_argument("--lr", type=float, 
                    default=1e-4, help="Learning rate.")

parser.add_argument("--num_inputs", type=int, 
                    default=20, help="Number input variables.")

parser.add_argument("--num_actions", type=int, 
                    default=105, help="Number of actions.")

parser.add_argument("--num_outputs", type=int, 
                    default=20, help="Number of output variables.")

parser.add_argument("--disable_wandb", action="store_const", const=True,
                    default=False, help="Use wandb for logging or not.")

parser.add_argument("--file_path", type=str,
                    default="./data/samples", help="Path to the dataset.")

parser.add_argument("--task_file_path", type=str,
                    default="./data/task_samples", help="Path to the task dataset.")

parser.add_argument("--benchmark_file_path", type=str,
                    default="./data/benchmark_samples", help="Path to the benchmark dataset.")

parser.add_argument("--num_layers", type=int,
                    default=6, help="Number of transformer blocks.")

parser.add_argument("--num_layers_policy", type=int,
                    default=2, help="Number of transformer blocks.")

parser.add_argument("--num_heads", type=int,
                    default=2, help="Number of attention heads.")

parser.add_argument("--embedding_dim", type=int,
                    default=128, help="Dimension of the token embeddings.")

parser.add_argument("--test_frequency", type=int,
                    default=5, help="The frequency with which we test the current policy.")

parser.add_argument("--checkpointing_frequency", type=int,
                    default=500, help="When to create a checkpoint for the model.")

parser.add_argument("--benchmarking_frequency", type=int,
                    default=100, help="When to create a checkpoint for the model.")

parser.add_argument("--load_model", type=str,
                    default=None, help="Path to the model weights that have to be loaded.")

parser.add_argument("--pid", type=int,
                    default=0, help="id of this process in the JAX virtual cluster.")

parser.add_argument("--num_nodes", type=int,
                    default=1, help="Number of physical nodes where we execute the computations.")

args = parser.parse_args()


# Distributed computing
if args.num_nodes > 1:
	jax.distributed.initialize(coordinator_address="134.94.166.3:1234",
								num_processes=args.num_nodes,
								process_id=args.pid)

print(jax.device_count())
print(jax.local_device_count())
print(jax.devices())
print(jax.local_devices())


import optax
import equinox as eqx

from tqdm import tqdm
from functools import partial

from graphax.vertex_game import step
from graphax.dataset import GraphDataset

from alphagrad.utils import (A0_loss,
    						get_masked_logits,
              				make_init_state,
                      		postprocess_data,
                        	make_batch,
                    		update_best_scores)
from alphagrad.data import (make_recurrent_fn,
							make_environment_interaction)

from alphagrad.sequential_transformer import SequentialTransformerModel
from alphagrad.evaluate import evaluate_tasks, make_reference_values, evaluate_benchmark

if not args.disable_wandb and args.pid == 0:
	wandb.login(key="local-f6fac6ab04ebeaa9cc3f9d44207adbb1745fe4a2", 
             	host="https://wandb.fz-juelich.de")
	wandb.init(entity="lohoff", project="AlphaGrad")
	wandb.run.name = args.name
	wandb.config = vars(args)

BS = args.batchsize*args.num_actions // jax.local_device_count()
NUM_INPUTS = args.num_inputs
NUM_INTERMEDIATES = args.num_actions
NUM_OUTPUTS = args.num_outputs

key = jrand.PRNGKey(args.seed)
INFO = [NUM_INPUTS, NUM_INTERMEDIATES, NUM_OUTPUTS]

nn_key, key = jrand.split(key, 2)
MODEL = SequentialTransformerModel(INFO, 
									args.embedding_dim, 
									args.num_layers, 
									args.num_heads,
									ff_dim=1024,
									num_layers_policy=args.num_layers_policy,
									policy_ff_dims=[1024, 512],
									value_ff_dims=[512, 256, 64], 
									key=nn_key)
if args.load_model is not None:
    MODEL = eqx.tree_deserialise_leaves(args.load_model, MODEL)

batched_step = jax.vmap(step)
batched_get_masked_logits = jax.vmap(get_masked_logits, in_axes=(0, 0))


recurrent_fn = make_recurrent_fn(MODEL,
                                batched_step, 
                                batched_get_masked_logits)

env_interaction = make_environment_interaction(INFO, 
                                            	args.num_simulations,
                                                recurrent_fn,
                                                batched_step,
												temperature=0)

eval_env_interaction = make_environment_interaction(INFO, 
													args.num_simulations,
													recurrent_fn,
													batched_step,
													temperature=0)

graph_dataset = GraphDataset(args.file_path)
print("Training datset size:", len(graph_dataset))
task_graph_dataset = GraphDataset(args.task_file_path, include_code=True)
benchmark_graph_dataset = GraphDataset(args.benchmark_file_path)


dataloader = DataLoader(graph_dataset, batch_size=args.batchsize, shuffle=True, num_workers=8, drop_last=True)
task_dataloader = DataLoader(task_graph_dataset, batch_size=8, shuffle=False, num_workers=1)
benchmark_dataloader = DataLoader(benchmark_graph_dataset, batch_size=100, shuffle=False, num_workers=4)

optim = optax.adamw(learning_rate=args.lr, weight_decay=1e-4)
opt_state = optim.init(eqx.filter(MODEL, eqx.is_inexact_array))

### needed to reassemble data
state_shape = (5, NUM_INPUTS+NUM_INTERMEDIATES+1, NUM_INTERMEDIATES)
state_idx = jnp.prod(jnp.array(state_shape))
policy_idx = state_idx + NUM_INTERMEDIATES
reward_idx = policy_idx + 1
split_idxs = (state_idx, policy_idx, reward_idx)


def tree_search(games, network, key):
	init_carry = make_init_state(games, key)
	data = env_interaction(network, init_carry)
	return postprocess_data(data)


select_first = lambda x: x[0] if isinstance(x, jax.Array) else x
parallel_mean = lambda x: lax.pmean(x, "num_devices")

@partial(eqx.filter_pmap, in_axes=(0, None, None, None), axis_name="num_devices", devices=jax.devices())
def train_agent(games, network, opt_state, key):
	data = tree_search(games, network, key)

	state, search_policy, search_value, _ = jnp.split(data, split_idxs, axis=-1)
	search_policy = search_policy.reshape(BS, NUM_INTERMEDIATES)
	search_value = search_value.reshape(BS, 1)
	state = state.reshape(BS, *state_shape)
 
	subkeys = jrand.split(key, BS)
	val, grads = eqx.filter_value_and_grad(A0_loss, has_aux=True)(network, 
																	search_policy, 
																	search_value, 
																	state,
																	args.policy_weight,
																	args.L2_weight,
																	args.entropy_weight,
																	subkeys)
	loss, aux = val
	loss = lax.pmean(loss, axis_name="num_devices")
	aux = lax.pmean(aux, axis_name="num_devices")
	grads = jtu.tree_map(parallel_mean, grads)

	updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(network, eqx.is_inexact_array))
	network = eqx.apply_updates(network, updates)
	return loss, aux, network, opt_state

pbar = tqdm(range(args.num_episodes))
best_rews = [-1000000. for _ in range(len(task_graph_dataset))]

print("Initial evaluation...")
single_best_performance = {}
names, rews, orders = evaluate_tasks(MODEL, task_dataloader, eval_env_interaction, key)
for name, rew, order in zip(names, rews, orders):
    single_best_performance[name] = (rew, order)

reference_values = make_reference_values(benchmark_dataloader, jax.local_devices())

counter = 0
for e in pbar:
	for edges in tqdm(dataloader):
		data_key, env_key, train_key, key = jrand.split(key, 4)
		games = make_batch(edges, num_devices=jax.local_device_count())

		start_time = time.time()
		losses, aux, models, opt_states= train_agent(games, MODEL, opt_state, train_key)	
		print("train", time.time() - start_time)

		loss = losses[0]
		aux = aux[0]
		MODEL = jtu.tree_map(select_first, models, is_leaf=eqx.is_inexact_array)
		opt_state = jtu.tree_map(select_first, opt_states, is_leaf=eqx.is_inexact_array)

		if not args.disable_wandb and args.pid == 0:
			wandb.log({"loss": loss.tolist(),
						"policy_loss": aux[0].tolist(),
						"value_loss": aux[1].tolist(),
						"L2_reg": aux[2].tolist(),
      					"entropy_reg": aux[3].tolist()})

		if counter % args.test_frequency == 0:
			st = time.time()
			names, rews, orders = evaluate_tasks(MODEL, task_dataloader, eval_env_interaction, key)

			update_best_scores(single_best_performance, names, rews, orders)
			print("eval time", time.time() - st)

			if not args.disable_wandb and args.pid == 0:	
				num_computations = {name:-int(nops) for name, nops in zip(names, rews)}
				wandb.log(num_computations)

			if (jnp.array(rews) >= jnp.array(best_rews)).all():
				print("Saving model...")
				best_rews = rews
				eqx.tree_serialise_leaves("./checkpoints/" + args.name + "_best.eqx", MODEL)
	
		if counter % args.checkpointing_frequency == 0:
			print("Checkpointing model...")
			eqx.tree_serialise_leaves("./checkpoints/" + args.name + "_chckpt.eqx", MODEL)

		if counter % args.benchmarking_frequency == 0:
			st = time.time()
			performance_delta = evaluate_benchmark(MODEL, 
													benchmark_dataloader, 
													reference_values, 
													jax.local_devices(), 
													eval_env_interaction, 
													key)
			print("Benchmarking time", time.time() - st)
			if not args.disable_wandb and args.pid == 0:
				wandb.log({"performance vs Markowitz": float(performance_delta)})

		counter += 1
		rewards = [int(r) for r in rews]
		pbar.set_description(f"loss: {loss}, performance: {performance_delta:.4f}, return: {rewards}")

