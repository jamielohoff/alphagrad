"""
Implementation of PPO with insights from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
"""

import os
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

import graphax as gx
from graphax.examples import RoeFlux_1d, RobotArm_6DOF, EncoderDecoder, ADALIF_SNN, f, g, Helmholtz
from alphagrad.vertexgame import step, make_graph, forward, reverse, cross_country
from alphagrad.vertexgame.runtime_game import RuntimeGame, _get_reward
from alphagrad.vertexgame.transforms import minimal_markowitz, embed
from alphagrad.utils import symlog, symexp
from alphagrad.sequential_transformer import SequentialTransformerModel

import matplotlib.pyplot as plt


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

key = jrand.PRNGKey(250197)
NUM_MEASUREMENTS = 100
DEVICE = jax.devices("cpu")[0] # Change this to create hardware-aware algorithm
FUNCTION = RoeFlux_1d
xs = [.01, .02, .02, .01, .03, .03]
xs = [jax.device_put(jnp.ones(1)*x, device=DEVICE) for x in xs]
env = RuntimeGame(NUM_MEASUREMENTS, FUNCTION, *xs)
i = env.graph.at[0, 0, 0].get()
v = env.graph.at[0, 0, 1].get() + env.graph.at[0, 0, 2].get()
o = env.graph.at[0, 0, 2].get()
INFO = jnp.array([i, v, o])
print("info", INFO)


_, fwd_fmas = forward(env.graph)
_, rev_fmas = reverse(env.graph)
mM_order = minimal_markowitz(env.graph, int(env.graph.at[0, 0, 1].get()))
print("mM_order", [int(i) for i in mM_order])
out, _ = cross_country(mM_order, env.graph)
print("number of operations:", fwd_fmas, rev_fmas, out[1])

act_seq = [i-1 for i in mM_order]
cc_time = _get_reward(NUM_MEASUREMENTS, FUNCTION, *xs, act_seq=act_seq)
fwd_time = _get_reward(NUM_MEASUREMENTS, FUNCTION, *xs, act_seq="fwd")
rev_time = _get_reward(NUM_MEASUREMENTS, FUNCTION, *xs, act_seq="rev")


FNAME = "Runtime_RobotArm_vertex_ppo.png"
ENTROPY_WEIGHT = 0.
VALUE_WEIGHT = 1.
EPISODES = 1000
BATCHSIZE = 16
ROLLOUT_LENGTH = env.graph.shape[-1] - int(o)
GAE_LAMBDA = 0.95
OBS_SHAPE = reduce(lambda x, y: x*y, env.graph.shape)
NUM_ACTIONS = env.graph.shape[-1]
EPS = 0.2 # clipping parameter for PPO
MINIBATCHES = 4
MINIBATCHSIZE = BATCHSIZE*ROLLOUT_LENGTH//MINIBATCHES
model = SequentialTransformerModel(INFO, 128, 3, 8,
									ff_dim=1024,
									num_layers_policy=2,
									policy_ff_dims=[1024, 512],
									value_ff_dims=[1024, 512, 256], 
									key=key)


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
    masked_prob_dist = prob_dist*mask / (jnp.sum(prob_dist*mask, axis=-1, keepdims=True) + 1e-7)

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
    size = BATCHSIZE*ROLLOUT_LENGTH//MINIBATCHES
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


# NOTE this is the bottleneck
# this should be parallelized somehow...
def env_steps(states, actions):
    next_obs, next_act_seqs, rewards, dones = [], [], [], []
    _states = [(states[0][i], states[1][i]) for i in range(BATCHSIZE)]
    
    out = [env.step(state, action) for state, action in zip(_states, actions)]
    
    next_obs = [o[0][0] for o in out]
    next_act_seqs = [o[0][1] for o in out]
    rewards = [o[1] for o in out]
    dones = [o[2] for o in out]
    next_states = (jnp.stack(next_obs), jnp.stack(next_act_seqs))
    return next_states, jnp.array(rewards), jnp.array(dones)


@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, 0, 0))
def get_actions(network, obs, key):
    net_key, act_key = jrand.split(key, 2)
    output = eqx.filter_jit(network)(obs, key=net_key)
    value = output[0]
    prob_dist = jnn.softmax(output[1:], axis=-1)
    
    mask = 1. - obs.at[1, 0, :].get()
    masked_prob_dist = prob_dist*mask / (jnp.sum(prob_dist*mask, axis=-1, keepdims=True) + 1e-7)
    distribution = distrax.Categorical(probs=masked_prob_dist)
    action = distribution.sample(seed=act_key)
    return action, masked_prob_dist


# Implementation of the RL algorithm
# @eqx.filter_jit
# @partial(jax.vmap, in_axes=(None, None, 0, 0))
# Painfully vmapped by hand
def rollout_fn(network, rollout_length, init_carry, key):
    keys = jrand.split(key, rollout_length)
    def step_fn(states, key):
        obs, act_seqs = states
        next_net_key, key = jrand.split(key, 2)
        keys = jrand.split(key, BATCHSIZE)
        actions, masked_prob_dists = get_actions(network, obs, keys)
        
        next_states, rewards, dones = env_steps(states, actions)
        next_obs, next_act_seqs = next_states
        discounts = 0.995*jnp.ones(BATCHSIZE) # TODO adjust this
        
        next_net_keys = jrand.split(next_net_key, BATCHSIZE)
        next_output = eqx.filter_jit(jax.vmap(network))(next_obs, key=next_net_keys)
        next_values = next_output[:, 0]
        next_prob_dists = next_output[:, 1:]
        
        new_sample = jnp.concatenate((obs.reshape(BATCHSIZE, -1),
                                    actions[:, jnp.newaxis], 
                                    rewards[:, jnp.newaxis], 
                                    dones[:, jnp.newaxis],
                                    next_obs.reshape(BATCHSIZE, -1), 
                                    next_values[:, jnp.newaxis],
                                    masked_prob_dists, 
                                    discounts[:, jnp.newaxis]), axis=1) # (sars')
        return next_states, new_sample
    
    return scan(step_fn, init_carry, keys)


def loss(network, trajectories, keys):
    state = trajectories[:, :OBS_SHAPE]
    state = state.reshape(-1, *env.graph.shape)
    actions = trajectories[:, OBS_SHAPE]
    actions = jnp.int32(actions)
    
    rewards = trajectories[:, OBS_SHAPE+1]
    next_state = trajectories[:, OBS_SHAPE+3:2*OBS_SHAPE+3]
    next_state = next_state.reshape(-1, *env.graph.shape)
    
    old_prob_dist = trajectories[:, 2*OBS_SHAPE+4:2*OBS_SHAPE+NUM_ACTIONS+4]
    discounts = trajectories[:, 2*OBS_SHAPE+NUM_ACTIONS+4]
    episodic_returns = trajectories[:, 2*OBS_SHAPE+NUM_ACTIONS+5]
    returns = trajectories[:, 2*OBS_SHAPE+NUM_ACTIONS+6]
    advantages = trajectories[:, 2*OBS_SHAPE+NUM_ACTIONS+7]
    
    log_probs, prob_dist, values, entropies = get_log_probs_and_value(network, state, actions, keys)
    _, _, next_values, _ = get_log_probs_and_value(network, next_state, actions, keys)
    norm_adv = (advantages - jnp.mean(advantages))/(jnp.std(advantages) + 1e-7)
    
    # Losses
    old_log_probs = jax.vmap(lambda dist, a: jnp.log(dist[a] + 1e-7))(old_prob_dist, actions)
    ratio = jnp.exp(log_probs - old_log_probs)
    clipping_objective = jnp.minimum(ratio*norm_adv, jnp.clip(ratio, 1.-EPS, 1.+EPS)*norm_adv)
    ppo_loss = jnp.mean(-clipping_objective)
    entropy_loss = jnp.mean(entropies)
    value_loss = .5*jnp.mean((returns - values)**2)
    
    # Metrics
    dV = episodic_returns - rewards - discounts*next_values # assess fit quality
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


# @eqx.filter_jit
def test_agent(network, rollout_length, keys):
    env_carry = init_carry(keys)
    _, trajectories = rollout_fn(network, rollout_length, env_carry, keys[0])
    trajectories = jnp.swapaxes(trajectories, 0, 1)
    returns = jax.jit(get_returns)(trajectories)
    best_return = jnp.max(returns[:, 1], axis=-1)
    idx = jnp.argmax(returns[:, -1], axis=-1)
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
optim = optax.chain(optax.adam(5e-4), optax.clip_by_global_norm(.5))
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

# Tracking different RL metrics
def plot_metrics(num_samples, ret, entropy_evo, value_fit_quality, expl_var, kl):
    fig, axs = plt.subplots(2, 3)
    fig.suptitle("Reinforcement learning metrics")
    axs[0, 0].set_title("return")
    axs[0, 0].plot(num_samples, ret)
    axs[0, 0].axhline(y=rev_time, color = 'r', linestyle = '--') 
    axs[0, 0].axhline(y=cc_time, color = 'g', linestyle = '--')
    axs[0, 0].axhline(y=fwd_time, color = 'k', linestyle = '--')
    
    axs[0, 1].set_title("kl div")
    axs[0, 1].plot(num_samples, kl)

    axs[0, 2].set_title("mean entropy")
    axs[0, 2].plot(num_samples, entropy_evo)

    axs[1, 0].set_title("explained variance")
    axs[1, 0].plot(num_samples, expl_var)
    
    axs[1, 1].set_title("value fit quality")
    axs[1, 1].plot(num_samples, value_fit_quality)
    
    axs[1, 2].set_title("relative gain")
    rel_gain = [(cc_time - ret[i])/cc_time for i in range(len(ret))]
    axs[1, 2].plot(num_samples, rel_gain)
        
    fig.tight_layout()
    plt.savefig(FNAME)
    plt.close()


# Training loop
pbar = tqdm(range(EPISODES))
ret, entropy_evo, value_fit_quality, expl_var, kl, nsamples = [], [], [], [], [], []
test_key = jrand.PRNGKey(1234)
test_keys = jrand.split(test_key, BATCHSIZE) # NOTE only 4 test envs
samplecounts = 0

env_keys = jrand.split(key, BATCHSIZE)
env_carry = init_carry(env_keys)
best_global_return = -100000.
best_global_act_seq = None

for episode in pbar:
    subkey, key = jrand.split(key, 2)
    keys = jrand.split(key, BATCHSIZE)  
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
    samplecounts += BATCHSIZE*ROLLOUT_LENGTH
   
    kl_div, policy_entropy, fit_quality, explained_var = metrics
    best_return, best_act_seq = test_agent(model, ROLLOUT_LENGTH, test_keys)
    
    if best_return > best_global_return:
        best_global_return = best_return
        best_global_act_seq = best_act_seq
        print(f"New best return: {best_return}")
        
        vertex_elimination_order = [int(i) for i in best_act_seq]
        _, fmas = jax.jit(cross_country)(vertex_elimination_order, env.graph)
        print(f"New best action sequence: {vertex_elimination_order} with {sum(fmas)} multiplications.")
    
    nsamples.append(samplecounts)
    kl.append(kl_div)
    ret.append(best_return)
    entropy_evo.append(policy_entropy)
    expl_var.append(explained_var)
    value_fit_quality.append(fit_quality)
        
    pbar.set_description(f"entropy: {policy_entropy:.4f}, returns: {best_return}, fit_quality: {fit_quality:.2f}, expl_var: {explained_var:.4}, kl_div: {kl_div:.4f}")
    plot_metrics(nsamples, ret, entropy_evo, value_fit_quality, expl_var, kl)
        
vertex_elimination_order = [int(i) for i in best_act_seq]
print(f"Best vertex elimination sequence after {EPISODES} episodes is {vertex_elimination_order} with {best_global_return} multiplications.")

