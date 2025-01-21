"""
Implementation of PPO with insights from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
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

from multiprocessing import set_start_method, Pool

from graphax.examples import RoeFlux_1d
from alphagrad.utils import (entropy, explained_variance, symlog, symexp,
                            default_value_transform, default_inverse_value_transform)
from alphagrad.vertexgame import forward, reverse, cross_country
from alphagrad.vertexgame.runtime_game import RuntimeGame, _get_reward2
from alphagrad.vertexgame.transforms import minimal_markowitz
from alphagrad.transformer.models import PPOModel
from alphagrad.config import setup_experiment

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="Test", help="Name of the experiment.")

parser.add_argument("--task", type=str,
                    default="RoeFlux_1d", help="Name of the task to run.")

parser.add_argument("--gpus", type=str, 
                    default="0", help="GPU ID's to use for training.")

parser.add_argument("--seed", type=int, 
                    default="250197", help="Random seed.")

parser.add_argument("--config_path", type=str, 
                    default=os.path.join(os.getcwd(), "config"), 
                    help="Path to the directory containing the configuration files.")

parser.add_argument("--wandb", type=str,
                    default="run", help="Wandb mode.")

args = parser.parse_args()

key = jrand.PRNGKey(args.seed)
model_key, init_key, key = jrand.split(key, 3)


config, graph, graph_shape, task_fn = setup_experiment(args.task, args.config_path, prefix="Runtime_")

parameters = config["hyperparameters"]
ENTROPY_WEIGHT = parameters["entropy_weight"]
VALUE_WEIGHT = parameters["value_weight"]
EPISODES = parameters["episodes"]
NUM_ENVS = parameters["num_envs"]
LR = parameters["lr"]

GAE_LAMBDA = parameters["ppo"]["gae_lambda"]
EPS = parameters["ppo"]["clip_param"]
MINIBATCHES = parameters["ppo"]["num_minibatches"]

ROLLOUT_LENGTH = parameters["ppo"]["rollout_length"]
OBS_SHAPE = reduce(lambda x, y: x*y, graph.shape)
NUM_ACTIONS = graph.shape[-1] # ROLLOUT_LENGTH # TODO fix this
MINIBATCHSIZE = NUM_ENVS*ROLLOUT_LENGTH//MINIBATCHES

# Make runtime measurements
NUM_MEASUREMENTS = config["runtime_measurements"]["num_measurements"]
BURNIN = config["runtime_measurements"]["measurement_burnin"]
MEASUREMENT_BATCHSIZE = config["runtime_measurements"]["measurement_batchsize"]
REWARD_SCALE = config["runtime_measurements"]["reward_scale"]
DEVICE = jax.devices("cpu")[0] # Change this to create hardware-aware algorithm
FUNCTION = RoeFlux_1d
xs = [.01, .02, .02, .01, .03, .03]
xs = [jax.device_put(jnp.ones(1)*x, device=DEVICE) for x in xs]
env = RuntimeGame(MEASUREMENT_BATCHSIZE, NUM_MEASUREMENTS, BURNIN, FUNCTION, 
                  *xs, reward_scale=REWARD_SCALE)

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
                    for weight, subkey in zip(weights, jrand.split(key, len(weights)))]
    new_biases = [jnp.zeros_like(bias) for bias in biases]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_biases, new_model, new_biases)
    return new_model

    
run_config = {
    "seed": args.seed,
    "entropy_weight": ENTROPY_WEIGHT, 
    "value_weight": VALUE_WEIGHT, 
    "lr": LR,
    "episodes": EPISODES, 
    "batchsize": NUM_ENVS, 
    "gae_lambda": GAE_LAMBDA, 
    "eps": EPS, 
    "minibatches": MINIBATCHES, 
    "minibatchsize": MINIBATCHSIZE, 
    "obs_shape": OBS_SHAPE, 
    "num_actions": NUM_ACTIONS, 
    "rollout_length": ROLLOUT_LENGTH
}


# Value scaling functions
def reward_normalization_fn(x):
    return x # default_value_transform(x) # symlog(x)

def inverse_reward_normalization_fn(x):
    return x # default_inverse_value_transform(x) # symexp(x)


# Definition of some RL metrics for diagnostics
def get_num_clipping_triggers(ratio):
    _ratio = jnp.where(ratio <= 1.+EPS, ratio, 0.)
    _ratio = jnp.where(ratio >= 1.-EPS, 1., 0.)
    return jnp.sum(_ratio)


@partial(jax.vmap, in_axes=(None, 0, 0, 0))
def get_log_probs_and_value(network, state, action, key):
    mask = 1. - state.at[1, 0, :].get()
    output = network(state, key=key)
    value = output[0]
    logits = output[1:]
    prob_dist = jnn.softmax(logits, axis=-1, where=mask)

    log_prob = jnp.log(prob_dist[action] + 1e-7)
    return log_prob, prob_dist, value, entropy(prob_dist)


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
        done = 1.
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
        value = traj[2]
        next_value = traj[3]
        discount = traj[4]
        # Simplest advantage estimate
        # The advantage estimate has to be done with the states and actions 
        # sampled from the old policy due to the importance sampling formulation
        # of PPO
        done = 1.
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
    size = NUM_ENVS*ROLLOUT_LENGTH//MINIBATCHES
    trajectories = trajectories.reshape(-1, trajectories.shape[-1])
    trajectories = jrand.permutation(key, trajectories, axis=0)
    return trajectories.reshape(MINIBATCHES, size, trajectories.shape[-1])


@jax.jit
def init_carry(keys):
    graphs = jnp.tile(env.graph[jnp.newaxis, ...], (len(keys), 1, 1, 1))
    act_seqs = jnp.zeros((len(keys), env.num_actions), dtype=jnp.int32)
    return (graphs, act_seqs)


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in tqdm(xs):
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)


# Multi-processing env_steps function for parallel compilation of different
# vertex elimination orders
def _init_envs(obs, act_seq):
    """
    Worker initializer to set up the global state.
    """
    global process_obs
    global process_act_seq
    process_obs = obs
    process_act_seq = act_seq
    
    
def reset_envs():
    obs, act_seq = env.reset()
    init_obs = jnp.stack([obs for _ in range(NUM_ENVS)])
    init_act_seq = jnp.stack([act_seq for _ in range(NUM_ENVS)])
    for _ in range(NUM_ENVS):
        pool.apply_async(_init_envs, args=(obs, act_seq))
    return (init_obs, init_act_seq)  
    
    
def step_fn(i, obs, action):
    return i, env.step(obs, action)

    
# Multi-processing step
def env_steps(states, actions):
    states = jax.device_put(states, jax.devices("cpu")[0])
    actions = jax.device_put(actions, jax.devices("cpu")[0])
    states = [(states[0][i], states[1][i]) for i in range(NUM_ENVS)]
    
    out = {}
    jobs = [pool.apply_async(step_fn, args=(i, state, action)) 
            for i, (state, action) in enumerate(zip(states, actions))]

    # Collect results and maintain order
    for job in jobs:
        key, result = job.get()
        out[key] = result
    
    next_states, rewards, dones = zip(*out.values())
    
    next_obs, next_act_seqs = zip(*list(next_states))
    next_states = (jnp.stack(list(next_obs)), jnp.stack(list(next_act_seqs)))
    
    rewards = jnp.array(list(rewards))
    dones = jnp.array(list(dones))
    
    return next_states, rewards, dones

# Function to predict the actions from the neural network policy
@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, 0, 0))
def get_actions_and_values(network, obs, key):
    net_key, act_key = jrand.split(key, 2)
    mask = 1. - obs.at[1, 0, :].get()
    
    output = network(obs, key=net_key)
    value = output[0]
    logits = output[1:]
    prob_dist = jnn.softmax(logits, axis=-1, where=mask)
    
    distribution = distrax.Categorical(probs=prob_dist)
    action = distribution.sample(seed=act_key)
    return action, prob_dist, value


# Implementation of the RL algorithm
# @eqx.filter_jit
# @partial(jax.vmap, in_axes=(None, None, 0, 0))
# NOTE: Painfully vmapped by hand
def rollout_fn(network, rollout_length, init_carry, key):
    keys = jrand.split(key, rollout_length)
    
    def step_fn(states, key):
        obs, act_seqs = states
        next_net_key, key = jrand.split(key, 2)
        keys = jrand.split(key, NUM_ENVS)
        actions, prob_dists, _ = get_actions_and_values(network, obs, keys)
        
        next_states, rewards, dones = env_steps(states, actions)
        next_obs, next_act_seqs = next_states
        discounts = 0.995*jnp.ones(NUM_ENVS) # TODO adjust this        
        
        next_net_keys = jrand.split(next_net_key, NUM_ENVS)
        _, _, next_values = get_actions_and_values(network, next_obs, next_net_keys)
        
        new_sample = jnp.concatenate((
            obs.reshape(NUM_ENVS, -1),
            actions[:, jnp.newaxis], 
            rewards[:, jnp.newaxis], 
            dones[:, jnp.newaxis],
            next_obs.reshape(NUM_ENVS, -1), 
            next_values[:, jnp.newaxis],
            prob_dists, 
            discounts[:, jnp.newaxis]
        ), axis=1) # (sars')
        return next_states, new_sample
    
    return scan(step_fn, init_carry, keys)


# TODO Need better management of random seeds
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
    
    num_triggers = get_num_clipping_triggers(ratio)
    trigger_ratio = num_triggers / len(ratio)
    
    clipping_objective = jnp.minimum(ratio*norm_adv, jnp.clip(ratio, 1.-EPS, 1.+EPS)*norm_adv)
    ppo_loss = jnp.mean(-clipping_objective)
    entropy_loss = jnp.mean(entropies)
    value_loss = jnp.mean((values - reward_normalization_fn(returns))**2)
    
    # Metrics
    dV = returns - rewards - discounts*inverse_reward_normalization_fn(next_values) # assess fit quality
    fit_quality = jnp.mean(jnp.abs(dV))
    explained_var = explained_variance(advantages, returns)
    kl_div = jnp.mean(optax.kl_divergence(jnp.log(prob_dist + 1e-7), old_prob_dist))
    total_loss = ppo_loss
    total_loss += VALUE_WEIGHT*value_loss
    total_loss -= ENTROPY_WEIGHT*entropy_loss
    return total_loss, [kl_div, entropy_loss, fit_quality, explained_var,ppo_loss, 
                        VALUE_WEIGHT*value_loss, ENTROPY_WEIGHT*entropy_loss, 
                        total_loss, trigger_ratio]


@eqx.filter_jit
def train_agent(network, opt_state, trajectories, keys):  
    grads, metrics = eqx.filter_grad(loss, has_aux=True)(network, trajectories, keys)   
    updates, opt_state = optim.update(grads, opt_state)
    network = eqx.apply_updates(network, updates)
    return network, opt_state, metrics


def test_agent(network, rollout_length, keys):
    env_carry = init_carry(keys)
    _, trajectories = rollout_fn(network, rollout_length, env_carry, keys[0])
    returns = get_returns(trajectories)
    print(returns)
    best_return = jnp.max(returns[:, 0], axis=-1)
    idx = jnp.argmax(returns[:, 0], axis=-1)
    best_act_seq = trajectories[idx, :, OBS_SHAPE]
    return best_return, best_act_seq, returns[:, 0]


if __name__ == '__main__':
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    
    with open("../../../wandb_key.txt") as f:
        wandb_key = f.readline()
    wandb.login(key=wandb_key, 
                host="https://wandb.fz-juelich.de")
    wandb.init(entity="ja-lohoff", project="AlphaGrad", 
                group="Runtime_" + args.task, config=run_config,
                mode=args.wandb)
    wandb.run.name = "PPO_runtime_" + args.task + "_" + args.name

    _, fwd_fmas = forward(env.graph)
    _, rev_fmas = reverse(env.graph)
    mM_order = minimal_markowitz(env.graph, int(env.graph.at[0, 0, 1].get()))
    print("mM_order", [int(i) for i in mM_order])
    _, cc_fmas = cross_country(mM_order, env.graph)
    print("number of operations:", fwd_fmas, rev_fmas, cc_fmas)

    # NOTE: We need to fix the elimination orders here
    act_seq = [i-1 for i in mM_order]
    runtimes = partial(_get_reward2, MEASUREMENT_BATCHSIZE, NUM_MEASUREMENTS, BURNIN, FUNCTION, *xs)
    cc_time = runtimes(act_seq=act_seq)
    fwd_time = runtimes(act_seq="fwd")
    rev_time = runtimes(act_seq="rev")
    print("runtimes:", fwd_time, rev_time, cc_time)
    
    # Creating the model
    model = PPOModel(
        graph_shape, 64, 6, 8,
        ff_dim=256,
        num_layers_policy=2,
        policy_ff_dims=[256, 256],
        value_ff_dims=[256, 128, 64], 
        key=key
    )
    
    # Initialization could help with performance
    model = init_weight(model, init_fn, init_key)
    
    # Set up the multiprocessing pool
    # NOTE: It is imperative to choose "spawn" as starting method!
    set_start_method("spawn")
    
    pool = Pool(
        processes=NUM_ENVS,
        initializer=_init_envs,
        initargs=(None, None)  # Placeholder for state initialization
    )
    # Define optimizer
    # TODO test L2 norm and stationary ADAM for better stability
    schedule = LR # optax.cosine_decay_schedule(LR, 5000, 0.)
    optim = optax.chain(optax.adam(schedule, b1=.9, eps=1e-7), 
                        optax.clip_by_global_norm(.5))
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


    # Training loop
    pbar = tqdm(range(EPISODES))
    ret, entropy_evo, value_fit_quality, expl_var, kl, nsamples = [], [], [], [], [], []
    test_key = jrand.PRNGKey(1234)
    test_keys = jrand.split(test_key, NUM_ENVS) # NOTE only 4 test envs
    samplecounts = 0

    env_keys = jrand.split(key, NUM_ENVS)
    env_carry = init_carry(env_keys)
    best_global_return = jnp.max(-jnp.array([fwd_time, rev_time, cc_time]))
    best_global_act_seq = None

    for episode in pbar:
        test_key, subkey, key = jrand.split(key, 3)
        keys = jrand.split(key, NUM_ENVS)  
        env_carry = init_carry(keys)
        env_carry, trajectories = rollout_fn(model, ROLLOUT_LENGTH, env_carry, key)
        trajectories = jnp.swapaxes(trajectories, 0, 1)
        trajectories = get_advantages(trajectories)
        batches = shuffle_and_batch(trajectories, subkey)
        
        # We perform multiple descent steps on a subset of the same trajectory sample
        # This severely increases data efficiency
        # Furthermore, PPO utilizes the 'done' property to continue already running
        # environments
        for i in range(MINIBATCHES):
            subkeys = jrand.split(key, MINIBATCHSIZE)
            model, opt_state, metrics = train_agent(model, opt_state, batches[i], subkeys)   
        samplecounts += NUM_ENVS*ROLLOUT_LENGTH
    
        kl_div, policy_entropy, fit_quality, explained_var, ppo_loss, value_loss, entropy_loss, total_loss, clipping_trigger_ratio = metrics
        
        test_keys = jrand.split(test_key, NUM_ENVS)
        best_return, best_act_seq, returns = test_agent(model, ROLLOUT_LENGTH, test_keys)
        
        if best_return > best_global_return:
            best_global_return = best_return
            best_global_act_seq = best_act_seq
            print(f"New best return: {best_return}")
            vertex_elimination_order = [int(i) for i in best_act_seq]
            print(f"New best action sequence: {vertex_elimination_order}")
            # elim_order_table.add_data(episode, best_return, np.array(best_act_seq))
        
        # Tracking different RL metrics
        wandb.log({
            "best_return": best_return,
            "mean_return": jnp.mean(returns),
            "KL divergence": kl_div,
            "entropy evolution": policy_entropy,
            "value function fit quality": fit_quality,
            "explained variance": explained_var,
            "sample count": samplecounts,
            "ppo loss": ppo_loss,
            "value loss": value_loss,
            "entropy loss": entropy_loss,
            "total loss": total_loss,
            "clipping trigger ratio": clipping_trigger_ratio
        })
            
        pbar.set_description(f"entropy: {policy_entropy:.4f}, best_return: {best_return}, mean_return: {jnp.mean(returns)}, fit_quality: {fit_quality:.2f}, expl_var: {explained_var:.4f}, kl_div: {kl_div:.4f}")
            
    vertex_elimination_order = [int(i) for i in best_act_seq]
    print(f"Best vertex elimination sequence after {EPISODES} episodes is {vertex_elimination_order} with {best_global_return} multiplications.")

