"""
PPO implementation with insights from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
"""

import os
import argparse
from functools import partial, reduce

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from tqdm import tqdm
import wandb

import distrax
import optax
import equinox as eqx

from alphagrad.config import setup_experiment
from alphagrad.experiments import make_benchmark_scores
from alphagrad.vertexgame import step
from alphagrad.utils import symlog, symexp
from alphagrad.transformer.sequential_transformer import SequentialTransformerModel


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="Test", help="Name of the experiment.")

parser.add_argument("--task", type=str,
                    default="RoeFlux_1d", help="Name of the task to run.")

parser.add_argument("--gpus", type=str, 
                    default="0", help="GPU ID's to use for training.")

parser.add_argument("--seed", type=int, 
                    default="250197", help="Random seed.")

args = parser.parse_args()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
key = jrand.PRNGKey(args.seed)

# RoeFlux @ 340 mults
# [36, 24, 81, 75, 72, 59, 89, 57, 27, 52, 42, 10, 46, 91, 70, 44, 16, 99, 97, 
# 43, 95, 4, 37, 8, 3, 66, 41, 7, 67, 38, 14, 1, 77, 58, 83, 26, 74, 78, 50, 49, 
# 86, 15, 47, 73, 28, 64, 21, 31, 84, 0, 62, 18, 6, 94, 76, 87, 56, 13, 39, 34, 
# 82, 90, 25, 88, 65, 33, 80, 79, 32, 30, 92, 68, 85, 20, 71, 55, 12, 54, 93, 61, 
# 45, 9, 53, 60, 48, 69, 40, 51, 19, 2, 22, 5, 29, 23, 63, 11, 17, 35]

# RoeFlux @ 335 mults
# [38, 46, 44, 20, 95, 24, 89, 72, 83, 86, 8, 7, 82, 42, 3, 79, 91, 30, 52, 26, 
# 10, 4, 25, 18, 16, 80, 41, 28, 99, 9, 68, 50, 81, 71, 12, 75, 84, 67, 93, 53, 
# 43, 36, 47, 73, 37, 33, 21, 32, 45, 97, 85, 31, 59, 94, 27, 58, 78, 6, 61, 34, 
# 90, 92, 15, 88, 5, 62, 76, 49, 39, 2, 60, 56, 77, 14, 57, 55, 70, 54, 13, 74, 
# 64, 17, 87, 63, 0, 65, 66, 69, 51, 40, 1, 48, 22, 19, 23, 35, 29, 11]

# RoeFlux # 327 mults
# [38, 46, 44, 20, 95, 24, 89, 72, 83, 86, 8, 7, 82, 42, 3, 79, 91, 30, 52, 26, 
# 10, 4, 32, 18, 37, 80, 41, 97, 99, 9, 68, 50, 77, 28, 12, 78, 84, 67, 93, 53, 
# 31, 36, 25, 73, 85, 75, 21, 13, 94, 0, 27, 16, 59, 66, 62, 58, 56, 43, 39, 34, 
# 90, 92, 45, 88, 5, 15, 76, 49, 81, 2, 60, 57, 74, 14, 71, 33, 55, 54, 47, 61, 
# 6, 17, 65, 63, 87, 70, 64, 69, 51, 40, 1, 48, 22, 19, 23, 35, 29, 11]


# RobotArm @ 258 mults
# [104, 93, 18, 9, 94, 79, 112, 13, 45, 89, 92, 17, 63, 76, 6, 66, 111, 96, 107, 
# 34, 103, 60, 64, 19, 47, 91, 37, 24, 50, 97, 88, 100, 74, 108, 8, 51, 16, 110, 
# 53, 55, 44, 5, 33, 56, 77, 90, 41, 57, 87, 31, 54, 95, 27, 46, 106, 21, 109, 
# 59, 99, 102, 29, 72, 30, 20, 52, 68, 62, 28, 48, 10, 25, 105, 86, 58, 70, 1, 
# 73, 83, 49, 14, 98, 32, 38, 80, 78, 7, 82, 42, 43, 4, 22, 85, 40, 36, 39, 2, 
# 69, 12, 75, 15, 0, 71, 81, 3, 26, 35, 23, 11]


# graph, graph_shape, RoeFlux_1d = make_RoeFlux_1d() # make_graph(RobotArm_6DOF, *xs) # 
# graph = embed(key, graph, [6, 114, 6])


config, graph, graph_shape, task_fn = setup_experiment(args.task, args.name)
mM_order, scores = make_benchmark_scores(graph)


ENTROPY_WEIGHT = config["entropy_weight"]
VALUE_WEIGHT = config["value_weight"]
EPISODES = config["episodes"]
BATCHSIZE = config["batchsize"]

GAE_LAMBDA = config["ppo"]["gae_lambda"]
EPS = config["ppo"]["clip_params"]
MINIBATCHES = config["ppo"]["num_minibatches"]

ROLLOUT_LENGTH = graph_shape[1] - graph_shape[2]
OBS_SHAPE = reduce(lambda x, y: x*y, graph.shape)
NUM_ACTIONS = ROLLOUT_LENGTH # graph.shape[-1]
MINIBATCHSIZE = BATCHSIZE*ROLLOUT_LENGTH//MINIBATCHES

model = SequentialTransformerModel(graph_shape, 128, 3, 8,
									ff_dim=1024,
									num_layers_policy=2,
									policy_ff_dims=[1024, 512],
									value_ff_dims=[1024, 512, 256], 
									key=key)


wandb.login(key="local-84c6642fa82dc63629ceacdcf326632140a7a899", 
            host="https://wandb.fz-juelich.de")
wandb.init(entity="ja-lohoff", project="AlphaGrad")
wandb.run.name = "PPO" + args.task + args.name
wandb.config = {"entropy_weight": ENTROPY_WEIGHT, 
                "value_weight": VALUE_WEIGHT, 
                "episodes": EPISODES, 
                "batchsize": BATCHSIZE, 
                "gae_lambda": GAE_LAMBDA, 
                "eps": EPS, 
                "minibatches": MINIBATCHES, 
                "minibatchsize": MINIBATCHSIZE, 
                "obs_shape": OBS_SHAPE, 
                "num_actions": NUM_ACTIONS, 
                "rollout_length": ROLLOUT_LENGTH, 
                "fwd_fmas": scores[0], 
                "rev_fmas": scores[1], 
                "out_fmas": scores[2]}


# Definition of some RL metrics for diagnostics
def explained_variance(advantage, empirical_return):
    return 1. - jnp.var(advantage)/jnp.var(empirical_return)


# Function to calculate the entropy of a probability distribution
def entropy(prob_dist):
    return -jnp.sum(prob_dist*jnp.log(prob_dist + 1e-7), axis=-1)


@partial(jax.vmap, in_axes=(None, 0, 0, 0))
def get_log_probs_and_value(network, state, action, key):
    mask = 1. - state.at[1, 0, :].get()
    output = network(state, key=key)
    value = output[0]
    prob_dist = jnn.softmax(output[1:], axis=-1)
    masked_prob_dist = prob_dist*mask / (jnp.sum(prob_dist*mask, axis=-1) + 1e-7)

    log_prob = jnp.log(prob_dist[action] + 1e-7)
    return log_prob, masked_prob_dist, value, entropy(masked_prob_dist)


@jax.jit
@jax.vmap
def get_returns(trajectories):
    rewards = trajectories[:, OBS_SHAPE+1]
    dones = trajectories[:, OBS_SHAPE+2]
    discounts = trajectories[:, 2*OBS_SHAPE+NUM_ACTIONS+4]
    inputs = jnp.stack([rewards, dones, discounts]).T
    
    def loop_fn(episodic_return, traj):
        reward = traj[0]
        done = traj[1]
        discount = traj[2]
        # Simplest advantage estimate
        # The advantage estimate has to be done with the states and actions 
        # sampled from the old policy due to the importance sampling formulation
        # of PPO
        done = 1. - done
        episodic_return = reward + discount*episodic_return*done
        return episodic_return, episodic_return
    
    _, output = lax.scan(loop_fn, 0., inputs[::-1])
    return output[::-1]


# Calculates advantages using generalized advantage estimation
@jax.jit
@jax.vmap
def get_advantages(trajectories):
    rewards = trajectories[:, OBS_SHAPE+1]
    dones = trajectories[:, OBS_SHAPE+2]
    values = trajectories[:, 2*OBS_SHAPE+3]
    next_values = jnp.roll(values, -1, axis=0)
    discounts = trajectories[:, 2*OBS_SHAPE+NUM_ACTIONS+4]
    inputs = jnp.stack([rewards, dones, values, next_values, discounts]).T
    
    def loop_fn(carry, traj):
        episodic_return, lastgaelam = carry
        reward = traj[0]
        done = traj[1]
        value = symexp(traj[2])
        next_value = symexp(traj[3])
        discount = traj[4]
        # Simplest advantage estimate
        # The advantage estimate has to be done with the states and actions 
        # sampled from the old policy due to the importance sampling formulation
        # of PPO
        done = 1. - done
        episodic_return = reward + discount*episodic_return*done
        delta = reward + next_value*discount*done - value
        advantage = delta + discount*GAE_LAMBDA*lastgaelam*done
        estim_return = advantage + value
        
        next_carry = (episodic_return, advantage)
        new_sample = jnp.array([episodic_return, estim_return, advantage])
        return next_carry, new_sample
    _, output = lax.scan(loop_fn, (0., 0.), inputs[::-1])
    return jnp.concatenate([trajectories, output[::-1]], axis=-1)
    
    
@jax.jit
def shuffle_and_batch(trajectories, key):
    size = BATCHSIZE*ROLLOUT_LENGTH//MINIBATCHES
    trajectories = trajectories.reshape(-1, trajectories.shape[-1])
    trajectories = jrand.permutation(key, trajectories, axis=0)
    return trajectories.reshape(MINIBATCHES, size, trajectories.shape[-1])


def init_carry(keys):
    graphs = jnp.tile(graph[jnp.newaxis, ...], (len(keys), 1, 1, 1))
    return graphs


# Implementation of the RL algorithm
@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, None, 0, 0))
def rollout_fn(network, rollout_length, init_carry, key):
    keys = jrand.split(key, rollout_length)
    def step_fn(state, key):
        net_key, next_net_key, act_key = jrand.split(key, 3)
        
        output = network(state, key=net_key)
        value = output[0]
        prob_dist = jnn.softmax(output[1:], axis=-1)
        
        mask = 1. - state.at[1, 0, :].get()
        masked_prob_dist = prob_dist*mask / (jnp.sum(prob_dist*mask, axis=-1) + 1e-7)
        
        distribution = distrax.Categorical(probs=masked_prob_dist)
        action = distribution.sample(seed=act_key)
        
        next_state, reward, done = step(state, action)
        discount = 1.
        next_output = network(next_state, key=next_net_key)
        next_value = next_output[0]
        next_prob_dist = next_output[1:]
        
        new_sample = jnp.concatenate((state.flatten(),
                                    jnp.array([action]), 
                                    jnp.array([reward]), 
                                    jnp.array([done]),
                                    next_state.flatten(), 
                                    jnp.array([next_value]),
                                    masked_prob_dist, 
                                    jnp.array([discount]))) # (sars')
        
        return next_state, new_sample
    
    return lax.scan(step_fn, init_carry, keys)


def loss(network, trajectories, keys):
    state = trajectories[:, :OBS_SHAPE]
    state = state.reshape(-1, *graph.shape)
    actions = trajectories[:, OBS_SHAPE]
    actions = jnp.int32(actions)
    
    rewards = trajectories[:, OBS_SHAPE+1]
    next_state = trajectories[:, OBS_SHAPE+3:2*OBS_SHAPE+3]
    next_state = next_state.reshape(-1, *graph.shape)
    
    old_prob_dist = trajectories[:, 2*OBS_SHAPE+4:2*OBS_SHAPE+NUM_ACTIONS+4]
    discounts = trajectories[:, 2*OBS_SHAPE+NUM_ACTIONS+4]
    episodic_returns = trajectories[:, 2*OBS_SHAPE+NUM_ACTIONS+5]
    returns = trajectories[:, 2*OBS_SHAPE+NUM_ACTIONS+6]
    advantages = trajectories[:, 2*OBS_SHAPE+NUM_ACTIONS+7]
    
    log_probs, prob_dist, values, entropies = get_log_probs_and_value(network, state, actions, keys)
    _, _, next_values, _ = get_log_probs_and_value(network, next_state, actions, keys)
    norm_adv = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-7)
    
    # Losses
    old_log_probs = jax.vmap(lambda dist, a: jnp.log(dist[a] + 1e-7))(old_prob_dist, actions)
    ratio = jnp.exp(log_probs - old_log_probs)
    clipping_objective = jnp.minimum(ratio*norm_adv, jnp.clip(ratio, 1.-EPS, 1.+EPS)*norm_adv)
    ppo_loss = jnp.mean(-clipping_objective)
    entropy_loss = jnp.mean(entropies)
    value_loss = .5*jnp.mean((symlog(returns) - values)**2)
    
    # Metrics
    dV = episodic_returns - rewards - discounts*symexp(next_values) # assess fit quality
    fit_quality = jnp.mean(jnp.abs(dV))
    explained_var = explained_variance(advantages, returns)
    kl_div = jnp.mean(optax.kl_divergence(jnp.log(prob_dist + 1e-7), old_prob_dist))
    total_loss = ppo_loss
    total_loss += VALUE_WEIGHT*value_loss
    total_loss -= ENTROPY_WEIGHT*entropy_loss
    return total_loss, [kl_div, entropy_loss, fit_quality, explained_var]
    

@eqx.filter_jit
def train_agent(network, opt_state, trajectories, keys):  
    grads, metrics = eqx.filter_grad(loss, has_aux=True)(network, trajectories, keys)   
    updates, opt_state = optim.update(grads, opt_state)
    network = eqx.apply_updates(network, updates)
    return network, opt_state, metrics


@eqx.filter_jit
def test_agent(network, rollout_length, keys):
    env_carry = init_carry(keys)
    _, trajectories = rollout_fn(network, rollout_length, env_carry, keys)
    returns = get_returns(trajectories)
    best_return = jnp.max(returns[:, 0], axis=-1)
    idx = jnp.argmax(returns[:, 0], axis=-1)
    best_act_seq = trajectories[idx, :, OBS_SHAPE]
    return best_return, best_act_seq


model_key, key = jrand.split(key, 2)
ortho_init = jnn.initializers.orthogonal(jnp.sqrt(2))


def init_ortho_weight(model, init_fn, key):
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

# Orthogonal initialization is supposed to help with PPO
model = init_ortho_weight(model, ortho_init, model_key)

# Define optimizer
# schedule = optax.linear_schedule(3e-4, 0., 4000)
optim = optax.chain(optax.adam(2.5e-4), optax.clip_by_global_norm(.5))
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


# Training loop
pbar = tqdm(range(EPISODES))
ret, entropy_evo, value_fit_quality, expl_var, kl, nsamples = [], [], [], [], [], []
test_key = jrand.PRNGKey(1234)
test_keys = jrand.split(test_key, 8)
samplecounts = 0

env_keys = jrand.split(key, BATCHSIZE)
env_carry = init_carry(env_keys)
best_global_return = jnp.max(-jnp.array(scores))
best_global_act_seq = None

elim_order_table = wandb.Table(columns=["episode", "best return", "best action sequence"])

for episode in pbar:
    subkey, key = jrand.split(key, 2)
    keys = jrand.split(key, BATCHSIZE)  
    env_carry = jax.jit(init_carry)(keys)
    env_carry, trajectories = rollout_fn(model, ROLLOUT_LENGTH, env_carry, keys)

    trajectories = get_advantages(trajectories)
    batches = shuffle_and_batch(trajectories, subkey)
    
    # We perform multiple descent steps on a subset of the same trajectory sample
    # This severely increases data efficiency
    # Furthermore, PPO utilizes the 'done' property to continue already running
    # environments
    for i in range(MINIBATCHES):
        subkeys = jrand.split(key, MINIBATCHSIZE)
        model, opt_state, metrics = train_agent(model, opt_state, batches[i], subkeys)   
    samplecounts += BATCHSIZE*ROLLOUT_LENGTH
    
    kl_div, policy_entropy, fit_quality, explained_var = metrics
    best_return, best_act_seq = test_agent(model, ROLLOUT_LENGTH, test_keys)
    
    if best_return > best_global_return:
        best_global_return = best_return
        best_global_act_seq = best_act_seq
        print(f"New best return: {best_return}")
        vertex_elimination_order = [int(i) for i in best_act_seq]
        print(f"New best action sequence: {vertex_elimination_order}")
        elim_order_table.add_data(episode, best_return, best_act_seq)
        
    
    # Tracking different RL metrics
    wandb.log({"return": best_return,
                "KL divergence": kl_div,
                "entropy evolution": policy_entropy,
                "explained variance": expl_var,
                "value function fit quality": fit_quality,
                "sample count": samplecounts})
        
    pbar.set_description(f"entropy: {policy_entropy:.4f}, returns: {best_return}, fit_quality: {fit_quality:.2f}, expl_var: {explained_var:.4}, kl_div: {kl_div:.4f}")
        
vertex_elimination_order = [int(i) for i in best_act_seq]
print(f"Best vertex elimination sequence after {EPISODES} episodes is {vertex_elimination_order} with {best_global_return} multiplications.")

