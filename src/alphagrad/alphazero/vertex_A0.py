import os
import argparse
import wandb
import time
from functools import partial, reduce

import numpy as np

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import flashbax as fbx
import optax
import equinox as eqx

from tqdm import tqdm

from alphagrad.config import setup_experiment
from alphagrad.experiments import make_benchmark_scores
from alphagrad.vertexgame import step
from alphagrad.alphazero.evaluate import evaluate
from alphagrad.utils import A0_loss, get_masked_logits, postprocess_data
from alphagrad.alphazero.environment_interaction import (make_recurrent_fn,
														make_environment_interaction)
from alphagrad.transformer.models import AlphaZeroModel

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="Test", help="Name of the experiment.")

parser.add_argument("--task", type=str,
                    default="RoeFlux_1d", help="Name of the task to run.")

parser.add_argument("--gpus", type=str, 
                    default="0,1,2,3", help="GPU ID's to use for training.")

parser.add_argument("--seed", type=int, 
                    default="250197", help="Random seed.")

parser.add_argument("--config_path", type=str, 
                    default=os.path.join(os.getcwd(), "config"), 
                    help="Path to the directory containing the configuration files.")

parser.add_argument("--wandb", type=str,
                    default="run", help="Wandb mode.")

parser.add_argument("--L2", type=float,
                    default=None, help="L2 regularization weight.")

args = parser.parse_args()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
key = jrand.PRNGKey(args.seed)

print("CPU devices:", jax.devices("cpu"))
print("GPU devices:", jax.devices("gpu"))


config, graph, graph_shape, task_fn = setup_experiment(args.task, args.config_path)
mM_order, scores = make_benchmark_scores(graph)

parameters = config["hyperparameters"]
ENTROPY_WEIGHT = parameters["entropy_weight"]
VALUE_WEIGHT = parameters["value_weight"]
EPISODES = parameters["episodes"]

NUM_ENVS = parameters["num_envs"]
PER_DEVICE_NUM_ENVS = NUM_ENVS // jax.device_count("gpu")

BATCHSIZE = parameters["batchsize"]
PER_DEVICE_BATCHSIZE = BATCHSIZE // jax.device_count("gpu")
LR = parameters["lr"]

GUMBEL_SCALE = parameters["A0"]["gumbel_scale"]
NUM_SIMULATIONS = parameters["A0"]["num_simulations"]
NUM_CONSIDERED_ACTIONS = parameters["A0"]["num_considered_actions"]
L2_WEIGHT = parameters["l2_weight"] if args.L2 is None else args.L2
REPLAY_BUFFER_SIZE = parameters["A0"]["replay_buffer_size"]
QTRANSFORM_PARAMS = parameters["A0"]["qtransform"]

ROLLOUT_LENGTH = int(graph_shape[1] - graph_shape[2])
OBS_SHAPE = reduce(lambda x, y: x*y, graph.shape)
NUM_ACTIONS = graph.shape[-1] # ROLLOUT_LENGTH # TODO fix this

run_config = {"entropy_weight": ENTROPY_WEIGHT, 
                "value_weight": VALUE_WEIGHT, 
                "episodes": EPISODES, 
                "num_envs": NUM_ENVS,
                "batchsize": BATCHSIZE, 
                "gumbel_scale": GUMBEL_SCALE, 
                "num_simulations": NUM_SIMULATIONS,
                "num_considered_actions": NUM_CONSIDERED_ACTIONS,
                "obs_shape": OBS_SHAPE, 
                "num_actions": NUM_ACTIONS, 
                "rollout_length": ROLLOUT_LENGTH, 
                "fwd_fmas": scores[0], 
                "rev_fmas": scores[1], 
                "out_fmas": scores[2]}

wandb.login(key="local-84c6642fa82dc63629ceacdcf326632140a7a899", 
            host="https://wandb.fz-juelich.de")
wandb.init(entity="ja-lohoff", project="AlphaGrad", group=args.task, 
           	mode=args.wandb, config=run_config)
wandb.run.name = "A0_" + args.task + "_" + args.name


key, model_key, init_key = jrand.split(key, 3)
model = AlphaZeroModel(graph_shape, 64, 3, 6,
						ff_dim=256,
						num_layers_policy=2,
						policy_ff_dims=[256, 128],
						value_ff_dims=[256, 128, 64], 
						key=model_key)

# init_fn = jnn.initializers.glorot_uniform() # jnn.initializers.orthogonal(jnp.sqrt(2))

# def init_weight(model, init_fn, key):
#     is_linear = lambda x: isinstance(x, eqx.nn.Linear)
#     get_weights = lambda m: [x.weight
#                             for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
#                             if is_linear(x)]
#     get_biases = lambda m: [x.bias
#                             for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
#                             if is_linear(x) and x.bias is not None]
#     weights = get_weights(model)
#     biases = get_biases(model)
#     new_weights = [init_fn(subkey, weight.shape)
#                     for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
#     new_biases = [jnp.zeros_like(bias) for bias in biases]
#     new_model = eqx.tree_at(get_weights, model, new_weights)
#     new_model = eqx.tree_at(get_biases, new_model, new_biases)
#     return new_model

# # Initialization could help with performance
# model = init_weight(model, init_fn, init_key)


# Setup of the necessary functions for the tree search
recurrent_fn = make_recurrent_fn(model, step, get_masked_logits)
env_interaction = make_environment_interaction(ROLLOUT_LENGTH, 
                                               	NUM_CONSIDERED_ACTIONS,
                                               	GUMBEL_SCALE,
                                            	NUM_SIMULATIONS,
                                                recurrent_fn,
                                                step,
                                                **QTRANSFORM_PARAMS)


eval = partial(evaluate, env_interaction)
pmap_evaluate = eqx.filter_pmap(eval,
								in_axes=(None, 0 , None), 
								axis_name="num_devices", 
								devices=jax.devices(), 
								donate="all")


def make_init_carry(graph, key):
    graphs = jnp.tile(graph[jnp.newaxis, ...], (PER_DEVICE_NUM_ENVS, 1, 1, 1))
    return (graphs, jnp.zeros(PER_DEVICE_NUM_ENVS), key)


def tree_search(graph, model, key):
	init_carry = make_init_carry(graph, key)
	data = env_interaction(model, init_carry)
	return postprocess_data(data)

pmap_tree_search = eqx.filter_pmap(tree_search,
                                   in_axes=(None, None, 0), 
									axis_name="num_devices", 
									devices=jax.devices("gpu"), 
									donate="all")

# Initialization of the optimizer
# TODO implement proper schedule because dips in the best_return cause model to
# break after a number of epochs
lr_schedule = LR # optax.linear_schedule(1e-3, LR, 50) 
optim = optax.chain(optax.adam(lr_schedule, b1=.9, eps=1e-7), 
                    optax.clip_by_global_norm(.5))
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


# Needed to reassemble data
state_shape = (5, graph_shape[0]+NUM_ACTIONS+1, NUM_ACTIONS)
state_idx = jnp.prod(jnp.array(state_shape))
policy_idx = state_idx + NUM_ACTIONS
reward_idx = int(policy_idx + 1)
split_idxs = (state_idx, policy_idx, reward_idx)


# Initializing the replay buffer
replay_buffer = fbx.make_item_buffer(max_length=REPLAY_BUFFER_SIZE, 
                                    min_length=1, 
                                    sample_batch_size=BATCHSIZE)

item_prototype =jnp.zeros(reward_idx+1, device=jax.devices("cpu")[0])


def fill_buffer(replay_buffer, buffer_state, samples):
	def loop_fn(buffer_state, sample):
		# This piece of code removes the final states from the buffer
		# new_buffer_state = lax.cond(sample[-1] > 0,
		#                             lambda bs: replay_buffer.add(bs, sample),
		#                             lambda bs: bs,
		#                             buffer_state)
		new_buffer_state = replay_buffer.add(buffer_state, sample)
		return new_buffer_state, None
	samples = samples.reshape(NUM_ENVS*ROLLOUT_LENGTH, -1)
	updated_buffer_state, _ = lax.scan(loop_fn, buffer_state, samples)
	return updated_buffer_state

fill_buf = partial(fill_buffer, replay_buffer)

sample_fn = replay_buffer.sample

@partial(jax.jit, donate_argnums=1)
def fill_and_sample(data, buffer_states, key):
	# Fill replay buffer
	buffer_states = fill_buf(buffer_states, data)
 
	# Sample from replay buffer
	samples = sample_fn(buffer_states, key)
	samples = samples.experience
	samples = samples.reshape(jax.device_count("gpu"), PER_DEVICE_BATCHSIZE, -1)
	return samples, buffer_states

# Helper functions
order_idx = 3*graph.shape[-2]*graph.shape[-1]
end_idx = order_idx+graph.shape[-1]-int(graph_shape[2])
@jax.jit
def get_best_return(data):
	act_seqs = data[:, :, -1, order_idx:end_idx].reshape(-1, ROLLOUT_LENGTH)
	# jax.debug.print("value_target={value_target}", value_target=data[0, 0, :, -2].flatten())
	returns = data[:, :, 0, -2].flatten()
 
	best_return = jnp.max(returns)
	best_act_seq = act_seqs[jnp.argmax(returns).astype(jnp.int32)] # TODO fix indexing here!
	return best_return, best_act_seq, returns


# Defining the training function
select_first = lambda x: x[0] if isinstance(x, jax.Array) else x
parallel_mean = lambda x: lax.pmean(x, "num_devices")

@partial(eqx.filter_pmap, 
        in_axes=(0, None, None, 0), 
        axis_name="num_devices", 
        devices=jax.devices("gpu"), 
        donate="all")
def train_agent(data, model, opt_state, key):
	state, search_policy, search_value, _ = jnp.split(data, split_idxs, axis=-1)
	state = state.reshape(PER_DEVICE_BATCHSIZE, *state_shape)
  
	subkeys = jrand.split(key, PER_DEVICE_BATCHSIZE)
	val, grads = eqx.filter_value_and_grad(A0_loss, has_aux=True)(model, 
																search_policy, 
																search_value, 
																state,
																VALUE_WEIGHT,
																L2_WEIGHT,
																ENTROPY_WEIGHT,
																subkeys)
	loss, aux = val
	loss = lax.pmean(loss, axis_name="num_devices")
	aux = lax.pmean(aux, axis_name="num_devices")
	grads = jtu.tree_map(parallel_mean, grads)

	updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
	model = eqx.apply_updates(model, updates)
	return loss, aux, model, opt_state


# Training loop
pbar = tqdm(range(EPISODES))
test_key, key = jrand.split(key, 2)

env_keys = jrand.split(key, BATCHSIZE)
print("Scores:", scores)
best_global_return = jnp.max(-jnp.array(scores))
best_global_act_seq = None

buffer_state = replay_buffer.init(item_prototype)
elim_order_table = wandb.Table(columns=["episode", "return", "elimination order"])

for episode in pbar:
	data_key, env_key, train_key, sample_key, key = jrand.split(key, 5)
	data_keys = jrand.split(data_key, jax.device_count("gpu"))
	train_keys = jrand.split(train_key, jax.device_count("gpu"))
 
	# start_time = time.time()
	data = pmap_tree_search(graph, model, data_keys)
	# print("tree search time", time.time() - start_time)
 
	# start_time = time.time()
	data = jax.device_get(data)
	# print("transfer time", time.time() - start_time)
	
	best_return, best_act_seq, returns = get_best_return(data)
 
	if best_return > best_global_return:
		best_global_return = best_return
		best_global_act_seq = best_act_seq
		print(f"New best return: {best_return}")
		# vertex_elimination_order = [int(i) for i in best_act_seq]
		print(f"New best action sequence: {best_act_seq}")
		# elim_order_table.add_data(episode, best_return, np.array(best_act_seq))
 
	# start_time = time.time()
	samples, buffer_state = fill_and_sample(data, buffer_state, sample_key)
	# print("sampling time", time.time() - start_time)

	# start_time = time.time()
	losses, aux, models, opt_states = train_agent(samples, model, opt_state, train_keys)
	# print("training time", time.time() - start_time)	

	loss = losses[0]
	aux = aux[0]
	model = jtu.tree_map(select_first, models, is_leaf=eqx.is_inexact_array)
	opt_state = jtu.tree_map(select_first, opt_states, is_leaf=eqx.is_inexact_array)

	wandb.log({"total loss": loss.tolist(),
				"policy loss": aux[0].tolist(),
				"value loss": aux[1].tolist(),
				"L2 loss": aux[2].tolist(),
				"entropy loss": aux[3].tolist(),
				"explained variance": aux[4].tolist(),
				"best_return": best_return,
    			"mean_return": jnp.mean(returns)})

	pbar.set_description(f"loss: {loss}, best_return: {best_return}, mean_return: {jnp.mean(returns)}")

