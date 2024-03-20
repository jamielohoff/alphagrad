import jax

import equinox as eqx

from alphagrad.utils import make_init_state, make_batch
from alphagrad.vertexgame import minimal_markowitz, cross_country


# TODO documentation
def evaluate(network, env_interaction_fn, key, games):
    # Evaluates the model on a set of given tasks by returning the reward, i.e.
    # number of computations and also returns the elimination order
	init_carry = make_init_state(games, key)
	output = env_interaction_fn(network, init_carry)
	rewards = output[:, -1, -2].flatten()

	order_idx = 3*games.shape[-2]*games.shape[-1]
	end_idx = games.shape[-1]
	orders = output[:, -1, order_idx:order_idx+end_idx]

	return rewards, orders


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

# TODO check if this is actually correct
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
	for ag, mM in zip(rewards, reference_values):
		perf += ag/mM

	return 1. - perf/n


def make_Markowitz_reference_values(dataloader, devices):
	rewards = []
 
	@jax.vmap
	def fmas_minimal_markowitz(edges):
		order = minimal_markowitz(edges)
		_, fmas = cross_country(order, edges)
		return fmas

	for edges in dataloader:
		games = make_batch(edges, num_devices=len(devices))
		rews = jax.pmap(fmas_minimal_markowitz, 
                    		in_axes=(0,), 
							axis_name="num_devices", 
							devices=devices)(games)
		rewards.extend([r for r in rews.flatten()])

	return rewards

