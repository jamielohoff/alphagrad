import time
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import flashbax as fbx
import optax
import equinox as eqx

import wandb
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

import alphagrad.utils as u
import alphagrad.sharding_utils as shu
from alphagrad.experiments import make_benchmark_scores
from alphagrad.vertexgame import step
import alphagrad.alphazero.tree_search as ts
from alphagrad.transformer.models import AlphaZeroModel


@hydra.main(version_base=None, config_path="config", config_name="Encoder.yaml")
def alphazero_experiment(cfg: DictConfig) -> None:
	key = jrand.PRNGKey(cfg.seed)

	print("CPU devices:", jax.devices("cpu"))
	print("GPU devices:", jax.devices("gpu"))

	# Build task and create scores with off-the-shelf benchmarks such as 
	# reverse-mode AD
	graph, graph_shape, task_fn = u.task_builder(cfg.task)
	mM_order, scores = make_benchmark_scores(graph)
	print(f"graph shape: {graph.shape}")

	NUM_DEVICES = jax.device_count("gpu")
	PER_DEVICE_NUM_ENVS = cfg.rl.num_envs // NUM_DEVICES
	PER_DEVICE_BATCHSIZE = cfg.rl.batch_size // NUM_DEVICES
 
	ROLLOUT_LENGTH = int(graph.shape[2] - graph_shape[2])
	OBS_SHAPE = shu.prod(graph.shape)
	NUM_ACTIONS = graph.shape[-1]
 
	run_config = {
		"fwd_fmas": scores[0], 
		"rev_fmas": scores[1], 
		"out_fmas": scores[2]
	}

	wandb.login(
		key="local-84c6642fa82dc63629ceacdcf326632140a7a899", 
		host="https://wandb.fz-juelich.de"
	)
	wandb.init(
		entity="ja-lohoff",
		project="AlphaGrad",
		group=cfg.task, 
		mode=cfg.wandb_mode,
		config=run_config
	)
	wandb.run.name = "A0_" + cfg.task + "_" + cfg.name


	key, model_key, init_key = jrand.split(key, 3)
	model = AlphaZeroModel(graph_shape, key=model_key, **cfg.rl.agent)

	# init_fn = jnn.initializers.truncated_normal(0.02)
	# TODO(jamielohoff): used orthogonal init in the past...check if the whole
	# thing works without that!

	# # NOTE: this is terribly unwieldy! flax does this a lot better!
	# def init_weight(model, init_fn, key):
	# 	is_linear = lambda x: isinstance(x, eqx.nn.Linear)
	# 	get_weights = lambda m: [x.weight
	# 							for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
	# 							if is_linear(x)]
	# 	get_biases = lambda m: [x.bias
	# 							for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
	# 							if is_linear(x) and x.bias is not None]
	# 	weights = get_weights(model)
	# 	biases = get_biases(model)
		
	# 	it = zip(weights, jax.random.split(key, len(weights)))
	# 	new_weights = [init_fn(subkey, weight.shape) for weight, subkey in it]
		
	# 	new_biases = [jnp.zeros_like(bias) for bias in biases]
	# 	new_model = eqx.tree_at(get_weights, model, new_weights)
	# 	new_model = eqx.tree_at(get_biases, new_model, new_biases)
		
	# 	return new_model

	# # Initialization could help with performance
	# model = init_weight(model, init_fn, init_key)
	value_transform, inverse_value_transform = u.get_value_tf(cfg.rl.value_transform)

	# TODO(jamielohoff): make this into one function! 
	# Setup of the necessary functions for the tree search
	tree_search_fn = ts.make_tree_search(
		model,
		step,
		ROLLOUT_LENGTH, 
		inverse_value_transform,
		**cfg.rl.tree_search
	)

	# Tree search init
	def make_init_carry(graph, key):
		tiling = (PER_DEVICE_NUM_ENVS, cfg.rl.lookback, 1, 1)
		graphs = jnp.tile(graph[jnp.newaxis, ...], tiling)
		return (graphs, jnp.zeros(PER_DEVICE_NUM_ENVS), key)

	# Tree search function
	@partial(eqx.filter_jit, donate="all")
	@partial(jax.vmap, in_axes=(None, None, 0))
	def tree_search(graph, model, key):
		init_carry = make_init_carry(graph, key)
		final_state, num_muls, data = tree_search_fn(model, init_carry)
		return final_state, num_muls, data # postprocess_data(data)

	# Initialization of the optimizer
	# TODO(jamielohoff): implement proper schedule because dips in the 
	# best_return cause model to break after a number of epochs
	lr_schedule = optax.cosine_decay_schedule(cfg.rl.lr, cfg.rl.episodes)
	optim = optax.chain(
		optax.adamw(lr_schedule, weight_decay=cfg.rl.wd),
		optax.clip_by_global_norm(1.)
	)
	opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


	# Needed to reassemble data
	state_shape = (5, int(graph_shape[0])+NUM_ACTIONS+1, NUM_ACTIONS)
	state_idx = jnp.prod(jnp.array(state_shape))
	policy_idx = state_idx + NUM_ACTIONS
	reward_idx = int(policy_idx + 1)
	value_idx = int(reward_idx + 1)
	split_idxs = (state_idx, policy_idx, reward_idx, value_idx, value_idx+1)


	# Initializing the replay buffer
	replay_buffer = fbx.make_trajectory_buffer(
		max_length_time_axis=ROLLOUT_LENGTH, 
		min_length_time_axis=ROLLOUT_LENGTH, 
		sample_batch_size=cfg.rl.batch_size,
		add_batch_size=cfg.rl.num_envs,
		sample_sequence_length=cfg.rl.lookback,
		period=1
	)

	item_prototype = jnp.zeros(value_idx+2, device=jax.devices("cpu")[0])


	# Fill the replay buffer
	def fill_buffer(replay_buffer, buffer_state, samples):
		updated_buffer_state = replay_buffer.add(buffer_state, samples)
		return updated_buffer_state


	fill_buf = partial(fill_buffer, replay_buffer)


	@partial(jax.jit, donate_argnums=1)
	def fill_and_sample(data, buffer_states, key):
		# Fill replay buffer
		buffer_states = fill_buf(buffer_states, data)

		# Sample from replay buffer
		samples = replay_buffer.sample(buffer_states, key)
		samples = samples.experience
		samples = samples.reshape(
			NUM_DEVICES, PER_DEVICE_BATCHSIZE, cfg.rl.lookback, -1
		)
		return samples, buffer_states


	# Helper functions
	order_idx = 3*graph.shape[-2]*graph.shape[-1]
	end_idx = order_idx+graph.shape[-1] - int(graph_shape[2])


	@jax.jit
	@jax.vmap
	def compute_value_targets(data):
		def loop_fn(returns, reward):
			val = reward + cfg.rl.discount*returns
			return val, val

		rewards = data[:, -3].flatten()
		_, returns = lax.scan(loop_fn, 0., rewards[::-1])
		return jnp.concatenate([data, returns[::-1, jnp.newaxis]], axis=-1)


	@jax.jit
	def get_best_act_seq(final_state, num_muls):
		act_seqs = final_state[:, :, 3, 0, 0:ROLLOUT_LENGTH]
		act_seqs = act_seqs.reshape(cfg.rl.num_envs, -1)

		best_num_muls = jnp.max(num_muls)
		best_act_seq = act_seqs[jnp.argmax(num_muls).astype(jnp.int32)]
		return best_num_muls, best_act_seq

	# Setup loss function
	loss_fn = partial(u.A0_loss, value_transform)

	# data and key need batch parallelism for agent training
	# TODO(jamielohoff): maybe we can make this more elegant with a shard_map?
	@partial(eqx.filter_jit, donate="all")
	def train_agent(model, opt_state, data, key):
		out = jnp.split(data, split_idxs, axis=-1)
		state, ts_policy, _, _, _, ts_target = out
		state = state.reshape(
      		NUM_DEVICES, PER_DEVICE_BATCHSIZE, 5*cfg.rl.lookback, *state_shape[1:]
        )
		ts_policy, ts_target = ts_policy[..., -1], ts_target[..., -1]

		@partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
		def per_device_grads(model, ts_policy, ts_target, state, key):
			keys = jrand.split(key, PER_DEVICE_BATCHSIZE)
			return eqx.filter_value_and_grad(loss_fn, has_aux=True)(
				model, ts_policy, ts_target, state, cfg.rl.value_weight, keys
			)
			
		val, grads = per_device_grads(model, ts_policy, ts_target, state, key)

		loss, aux = val
		loss = jnp.mean(loss, axis=0)
		aux = jnp.mean(aux, axis=0)
		grads = jtu.tree_map(u.parallel_mean, grads)

		updates, opt_state = optim.update(
			grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
		)
		print(grads, updates, opt_state)
		model = eqx.apply_updates(model, updates)
		return loss, aux, model, opt_state


	# training loop
	pbar = tqdm(range(cfg.rl.episodes))
	print("scores:", scores)
	print("minimal Markowitz order:", [int(o) for o in mM_order])
	best_global_num_muls = jnp.max(-jnp.array(scores))

	buffer_state = replay_buffer.init(item_prototype)

	# generate sharding for batch parallelism
	batch_sharding = lambda shape: shu.get_sharding(
    	shape, 
    	axes=0,
    	axes_names="batch",
		num_devices=NUM_DEVICES
    )

	# training loop
	for ep in pbar:
		ts_key, train_key, sample_key, key = jrand.split(key, 4)

		# shard random keys for automatic batch parallelism.
		# model and graph object are automatically replicated by JIT compiler
		train_keys = jrand.split(train_key, NUM_DEVICES)
		ts_keys = jrand.split(ts_key, NUM_DEVICES)
  
		key_sharding = batch_sharding(ts_keys.shape)
		ts_keys = jax.device_put(ts_keys, key_sharding)
		train_keys = jax.device_put(train_keys, key_sharding)

		start_time = time.time()
		final_state, num_muls, data = tree_search(graph, model, ts_keys)
		print("tree search time", time.time() - start_time)

		# device transfer
		data = jax.device_get(data)
		data = data.reshape(cfg.rl.num_envs, ROLLOUT_LENGTH, -1)
		data = compute_value_targets(data)

		final_state = jax.device_get(final_state)
		num_muls = jax.device_get(num_muls)
		best_num_muls, best_act_seq = get_best_act_seq(final_state, num_muls)

		if best_num_muls > best_global_num_muls:
			best_global_num_muls = best_num_muls
			print(f"new best return: {best_num_muls}")
			vertex_elimination_order = [int(i) for i in best_act_seq]
			print(f"new best action sequence: {vertex_elimination_order}")

		# fill sample buffer and sample new training data
		samples, buffer_state = fill_and_sample(data, buffer_state, sample_key)
  
		# shard the samples for parallel training
		data_sharding = batch_sharding(samples.shape)
		samples = jax.device_put(samples, data_sharding)
		
		start_time = time.time()
		losses, aux, models, opt_states = train_agent(
			model, opt_state, samples, train_keys
		)
		print("training time", time.time() - start_time)	

		loss, aux = losses[0], aux[0]
  
		# NOTE: this here could be a major bottleneck. PLS benchmark
		# we should just fix this so that we do not need these weird commands...
		model = jtu.tree_map(u.select_first, models, is_leaf=eqx.is_inexact_array)
		opt_state = jtu.tree_map(
			u.select_first, opt_states, is_leaf=eqx.is_inexact_array
		)

		wandb.log({"total loss": loss.tolist(),
					"policy loss": aux[0].tolist(),
					"value loss": aux[1].tolist(),
					"l2 loss": aux[2].tolist(),
					"entropy loss": aux[3].tolist(),
					"explained variance": aux[4].tolist(),
					"best_return": best_num_muls,
					"mean_return": jnp.mean(num_muls)})

		pbar.set_description(
			f"loss: {loss:.4f}, best_num_muls: {best_num_muls}, mean_num_muls: {jnp.mean(num_muls):.2f}"
		)


if __name__ == "__main__":
	alphazero_experiment()

