import os
import argparse
import wandb
import time
from functools import partial, reduce

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

from alphagrad.config import setup_joint_experiment
from alphagrad.experiments import make_benchmark_scores
from alphagrad.vertexgame import step, embed
from alphagrad.alphazero.evaluate import evaluate
from alphagrad.utils import (A0_loss, get_masked_logits, symlog, symexp, 
							default_value_transform, default_inverse_value_transform)
from alphagrad.alphazero.environment_interaction import (make_recurrent_fn,
														make_environment_interaction)
from alphagrad.transformer.models import AlphaZeroModel


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="Test", help="Name of the experiment.")

parser.add_argument("--task", type=str,
                    default="RoeFlux_1d", help="Name of the task to run.")

parser.add_argument("--gpus", type=str, 
                    default="0,1", help="GPU ID's to use for training.")

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


config, graphs, graph_shapes, task_fns = setup_joint_experiment(args.config_path)
NAMES = config["tasks"]

max_graph = max(graphs, key=lambda x: x.shape[-1])
max_graph_shape = max(graph_shapes, key=lambda x: x[1])

GRAPH_REPO = []
orders_scores = {}
for name, graph in zip(NAMES, graphs):
    key, subkey = jrand.split(key, 2)
    mM_order, scs = make_benchmark_scores(graph)
    orders_scores[name] = {"order": mM_order, 
                            "fwd_fmas":scs[0],
                            "rev_fmas":scs[1],
                            "mM_fmas":scs[2]}
    _graph = embed(subkey, graph, max_graph_shape)
    GRAPH_REPO.append(_graph)
    
GRAPH_REPO = jnp.stack(GRAPH_REPO, axis=0)

parameters = config["hyperparameters"]
LR = parameters["lr"]
EPISODES = parameters["episodes"]
ENTROPY_WEIGHT = 0. # parameters["entropy_weight"]
VALUE_WEIGHT = parameters["value_weight"]
L2_WEIGHT = parameters["l2_weight"] if args.L2 is None else args.L2
DISCOUNT = parameters["discount"]

NUM_ENVS = parameters["num_envs"]
PER_DEVICE_NUM_ENVS = NUM_ENVS // jax.device_count("gpu")

BATCHSIZE = parameters["batchsize"]
PER_DEVICE_BATCHSIZE = BATCHSIZE // jax.device_count("gpu")

GUMBEL_SCALE = parameters["A0"]["gumbel_scale"]
NUM_SIMULATIONS = parameters["A0"]["num_simulations"]
NUM_CONSIDERED_ACTIONS = parameters["A0"]["num_considered_actions"]
REPLAY_BUFFER_SIZE = parameters["A0"]["replay_buffer_size"]
QTRANSFORM_PARAMS = parameters["A0"]["qtransform"]
LOOKBACK = parameters["A0"]["lookback"]

ROLLOUT_LENGTH = int(max_graph_shape[-2] - max_graph_shape[-1])
OBS_SHAPE = reduce(lambda x, y: x*y, graph.shape)
NUM_ACTIONS = graph.shape[-1] # ROLLOUT_LENGTH # TODO fix this

run_config = {"seed": args.seed,
    			"entropy_weight": ENTROPY_WEIGHT, 
                "value_weight": VALUE_WEIGHT, 
                "l2_weight": L2_WEIGHT,
                "lr": LR,
                "episodes": EPISODES, 
                "num_envs": NUM_ENVS,
                "batchsize": BATCHSIZE, 
                "gumbel_scale": GUMBEL_SCALE, 
                "num_simulations": NUM_SIMULATIONS,
                "num_considered_actions": NUM_CONSIDERED_ACTIONS,
                "replay_buffer_size": REPLAY_BUFFER_SIZE,
                "lookback": LOOKBACK,
                "qtransform_params": QTRANSFORM_PARAMS,
                "obs_shape": OBS_SHAPE, 
                "num_actions": NUM_ACTIONS, 
                "rollout_length": ROLLOUT_LENGTH, 
                "scores": orders_scores}

wandb.login(key="local-84c6642fa82dc63629ceacdcf326632140a7a899", 
            host="https://wandb.fz-juelich.de")
wandb.init(entity="ja-lohoff", project="AlphaGrad", group=args.task, 
           	mode=args.wandb, config=run_config)
wandb.run.name = "A0_" + args.task + "_" + args.name


key, model_key, init_key = jrand.split(key, 3)
model = AlphaZeroModel(max_graph_shape, 64, 6, 6,
						ff_dim=256,
						num_layers_policy=2,
						policy_ff_dims=[256, 128],
						value_ff_dims=[256, 128, 64], 
						key=model_key)

# model = AlphaZeroModel(5*LOOKBACK, NUM_ACTIONS, key=model_key)

init_fn = jnn.initializers.orthogonal(jnp.sqrt(2))

def init_weight(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x)]
    get_biases = lambda m: [x.bias
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x) and x.bias is not None]
    weights = get_weights(model)
    biases = get_biases(model)
    new_weights = [init_fn(subkey, weight.shape)
                    for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    new_biases = [jnp.zeros_like(bias) for bias in biases]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_biases, new_model, new_biases)
    return new_model

# Initialization could help with performance
model = init_weight(model, init_fn, init_key)

def value_transform(x):
    return symlog(x) # default_value_transform(x) # 

def inverse_value_transform(x):
    return symexp(x) # default_inverse_value_transform(x) # 


# Setup of the necessary functions for the tree search
recurrent_fn = make_recurrent_fn(value_transform, 
                                inverse_value_transform, 
                                model, 
                                step, 
                                get_masked_logits)
env_interaction = make_environment_interaction(value_transform,
                                               	inverse_value_transform,
    											ROLLOUT_LENGTH, 
                                               	NUM_CONSIDERED_ACTIONS,
                                               	GUMBEL_SCALE,
                                            	NUM_SIMULATIONS,
                                                recurrent_fn,
                                                step,
                                                **QTRANSFORM_PARAMS)


# Setup loss function
loss_fn = partial(A0_loss, value_transform, inverse_value_transform)

eval = partial(evaluate, env_interaction)
pmap_evaluate = eqx.filter_pmap(eval,
								in_axes=(None, 0 , None), 
								axis_name="num_devices", 
								devices=jax.devices("gpu"), 
								donate="all")


def make_init_carry(graph, key):
    graphs = jnp.tile(graph[jnp.newaxis, ...], (PER_DEVICE_NUM_ENVS, LOOKBACK, 1, 1))
    return (graphs, jnp.zeros(PER_DEVICE_NUM_ENVS), key)


def tree_search(graph, model, key):
	init_carry = make_init_carry(graph, key)
	final_state, num_muls, data = env_interaction(model, init_carry)
	return final_state, num_muls, data # postprocess_data(data)

pmap_tree_search = eqx.filter_pmap(tree_search,
                                   in_axes=(None, None, 0), 
									axis_name="num_devices", 
									devices=jax.devices("gpu"), 
									donate="all")

# Initialization of the optimizer
# TODO implement proper schedule because dips in the best_return cause model to
# break after a number of epochs
lr_schedule = optax.cosine_decay_schedule(LR, EPISODES) # LR
optim = optax.chain(optax.adam(lr_schedule, b1=.9, eps=1e-7), 
                    optax.clip_by_global_norm(.5))
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


# Needed to reassemble data
state_shape = (5, max_graph_shape[0]+NUM_ACTIONS+1, NUM_ACTIONS)
state_idx = jnp.prod(jnp.array(state_shape))
policy_idx = state_idx + NUM_ACTIONS
reward_idx = int(policy_idx + 1)
value_idx = int(reward_idx + 1)
split_idxs = (state_idx, policy_idx, reward_idx, value_idx, value_idx+1)

# Initializing the replay buffer
replay_buffer = fbx.make_trajectory_buffer(max_length_time_axis=ROLLOUT_LENGTH, 
											min_length_time_axis=ROLLOUT_LENGTH, 
											sample_batch_size=BATCHSIZE,
											add_batch_size=NUM_ENVS,
											sample_sequence_length=LOOKBACK,
											period=1)

item_prototype =jnp.zeros(value_idx+2, device=jax.devices("cpu")[0])


def fill_buffer(replay_buffer, buffer_state, samples):
	updated_buffer_state = replay_buffer.add(buffer_state, samples)
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
	samples = samples.reshape(jax.device_count("gpu"), PER_DEVICE_BATCHSIZE, LOOKBACK, -1)
	return samples, buffer_states


# Helper functions
order_idx = 3*graph.shape[-2]*graph.shape[-1]
end_idx = order_idx+graph.shape[-1]-int(max_graph_shape[2])


@jax.jit
@jax.vmap
def _raw_values(data):
	raw_values = data[:, -2].flatten()
	return jnp.concatenate([data, raw_values[:, jnp.newaxis]], axis=-1)


@jax.jit
@jax.vmap
def _compute_return(data):
	def loop_fn(returns, reward):
		return (reward + DISCOUNT*returns), reward + DISCOUNT*returns

	rewards = data[:, -3].flatten()
	_, returns = lax.scan(loop_fn, 0., rewards[::-1])
	return jnp.concatenate([data, returns[::-1, jnp.newaxis]], axis=-1)


@partial(jax.jit, static_argnums=(1,))
@jax.vmap
def _compute_bootstrap(data, lookahead: int = 5):
	rewards = data[:, -3].flatten()
	values = data[:, -2].flatten()
	values = jnp.roll(values, -lookahead)
	values = values.at[-lookahead:].set(0.)

	n_step_return = rewards

	for i in range(1, lookahead):
		rew = jnp.roll(rewards, -i)
		rew = rew.at[-i:].set(0.)
		n_step_return += DISCOUNT**i*rew

	n_step_return += DISCOUNT**5*values
	return jnp.concatenate([data, n_step_return[:, jnp.newaxis]], axis=-1)

compute_value_targets = _compute_return


@jax.jit
def get_best_act_seq(final_state, num_muls):
	act_seqs = final_state[:, :, 3, 0, 0:ROLLOUT_LENGTH].reshape(NUM_ENVS, -1)
	best_num_muls = jnp.max(num_muls)
	best_act_seq = act_seqs[jnp.argmax(num_muls).astype(jnp.int32)] # TODO fix indexing here!
	return best_num_muls, best_act_seq


# Defining the training function
select_first = lambda x: x[0] if isinstance(x, jax.Array) else x
parallel_mean = lambda x: lax.pmean(x, "num_devices")

@partial(eqx.filter_pmap, 
        in_axes=(0, None, None, 0), 
        axis_name="num_devices", 
        devices=jax.devices("gpu"), 
        donate="all")
def train_agent(data, model, opt_state, key):
	state, search_policy, search_rewards, search_value, done, search_target = jnp.split(data, split_idxs, axis=-1)
	state = state.reshape(PER_DEVICE_BATCHSIZE, 5*LOOKBACK, *state_shape[1:])

	subkeys = jrand.split(key, PER_DEVICE_BATCHSIZE)
	val, grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, 
																search_policy[:, -1], 
																search_target[:, -1], 
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
best_global_metric = {name: {} for name in NAMES}
for name in NAMES:
    scores = orders_scores[name]
    fwd, rev, mM = scores["fwd_fmas"], scores["rev_fmas"], scores["mM_fmas"]
    _best_glob_ret = jnp.max(-jnp.array([fwd, rev, mM]))
    best_global_metric[name]["best_return"] = _best_glob_ret
    best_global_metric[name]["act_seq"] = None

buffer_state = replay_buffer.init(item_prototype)
elim_order_table = wandb.Table(columns=["episode", "return", "elimination order"])

for episode in pbar:
	data_key, env_key, train_key, sample_key, key = jrand.split(key, 5)
	data_keys = jrand.split(data_key, jax.device_count("gpu"))
	train_keys = jrand.split(train_key, jax.device_count("gpu"))
 
	start_time = time.time()
	final_state, num_muls, data = pmap_tree_search(graph, model, data_keys)
	print("tree search time", time.time() - start_time)
 
	# start_time = time.time()
	data = jax.device_get(data)
	data = data.reshape(NUM_ENVS, ROLLOUT_LENGTH, -1)
	data = compute_value_targets(data)
	# print("transfer time", time.time() - start_time)
 
	final_state = jax.device_get(final_state)
	num_muls = jax.device_get(num_muls)
	best_num_muls, best_act_seq = get_best_act_seq(final_state, num_muls)
 
	if best_num_muls > best_global_num_muls:
		best_global_num_muls = best_num_muls
		best_global_act_seq = best_act_seq
		print(f"New best return: {best_num_muls}")
		vertex_elimination_order = [int(i) for i in best_act_seq]
		print(f"New best action sequence: {vertex_elimination_order}")
		# elim_order_table.add_data(episode, best_return, np.array(best_act_seq))
 
	samples, buffer_state = fill_and_sample(data, buffer_state, sample_key)
	# TODO: Perform multiple training steps over the replay buffer
	
	start_time = time.time()
	losses, aux, models, opt_states = train_agent(samples, model, opt_state, train_keys)
	print("training time", time.time() - start_time)	

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
				"best_return": best_num_muls, # TODO maybe change naming once the algorithm works
    			"mean_return": jnp.mean(num_muls)})

	pbar.set_description(f"loss: {loss:.4f}, best_num_muls: {best_num_muls}, mean_num_muls: {jnp.mean(num_muls):.2f}")

