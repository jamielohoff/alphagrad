import jax
import jax.numpy as jnp

import equinox as eqx

from alphagrad.utils import make_init_state, make_batch
from graphax.transforms.markowitz import minimal_markowitz
from graphax import cross_country


# TODO documentation
def evaluate(network, env_interaction_fn, key, games):
    # Evaluates the model on a set of given tasks by returning the reward, i.e.
    # number of computations
	init_carry = make_init_state(games, key)
	output = env_interaction_fn(network, init_carry)
	rewards = output[:, -1, -2].flatten()

	order_idx = 3*games.shape[-2]*games.shape[-1]
	end_idx = games.shape[-1]
	order = output[:, -1, order_idx:order_idx+end_idx]

	return rewards, order


# TODO possibly pmap this? - for a larger test dataset
def evaluate_tasks(model, dataloader, interaction_fn, key):
	rewards, orders, task_names = [], [], []
	for names, edges in dataloader:
		_names = [name.decode("utf-8") for name in names]
		task_names.extend(_names)

		games = make_batch(edges, num_devices=1)
		_rews, _orders = eqx.filter_jit(evaluate)(model, interaction_fn, key, games)
		rewards.extend([r for r in _rews])
		orders.extend([o for o in _orders])
	return task_names, rewards, orders


# TODO possibly pmap this? - for a larger test dataset
def evaluate_benchmark(model, dataloader, reference_values, devices, interaction_fn, key):
	rewards = []
	for edges in dataloader:
		games = make_batch(edges, num_devices=len(devices))
		_rews, _ = eqx.filter_pmap(evaluate,
							in_axes=(None, None, None, 0), 
							axis_name="num_devices", 
							devices=devices)(model, interaction_fn, key, games)
		rewards.extend([-r for r in _rews.flatten()])

	n = len(rewards)
	perf = 0.
	for ag, cc in zip(rewards, reference_values):
		perf += cc/ag

	return 1. - perf/n


def make_reference_values(dataloader, devices):
	rewards = []
	vmap_minimal_markowitz = jax.vmap(minimal_markowitz, in_axes=0)
	vmap_cross_country = jax.vmap(cross_country, in_axes=(0, 0))
	for edges in dataloader:
		games = make_batch(edges, num_devices=len(devices))
		orders = jax.pmap(vmap_minimal_markowitz, 
                    		in_axes=(0,), 
							axis_name="num_devices", 
							devices=devices)(games)
		_, _rews = jax.pmap(vmap_cross_country, 
                     		in_axes=(0, 0),
                       		axis_name="num_devices", 
							devices=devices)(orders, games)
		rewards.extend([r for r in _rews.flatten()])

	return rewards

