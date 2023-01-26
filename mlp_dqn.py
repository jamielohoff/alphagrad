import os
import copy
import time
import functools as ft
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import chex
import equinox as eqx
import rlax
import optax

from graphax.game import VertexGame
from graphax.elimination import forward, reverse
from graphax.examples.random import construct_random_graph
from graphax.examples.helmholtz import construct_Helmholtz

from replay_memory import ReplayMemory


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

key = jrand.PRNGKey(123)
EPISODES = 250
LR = 1e-3
BATCHSIZE = 64
REPLAY_SIZE = 512
EPS = .75 # the higher the more exploration
NINPUTS = 20
NINTERMEDIATES = 50
NOUTPUTS = 2
UPDATE_FREQUENCY = 5
GAMMA = .9

GS, nedges = construct_random_graph(NINPUTS, NINTERMEDIATES, NOUTPUTS, key)

# create a replay buffer
replay_buffer = ReplayMemory(REPLAY_SIZE)

forward_gs = copy.deepcopy(GS)
_, ops = forward(forward_gs, nedges, NINTERMEDIATES, NINPUTS)
print("forward-mode:", ops)

reverse_gs = copy.deepcopy(GS)
_, ops = reverse(reverse_gs, nedges, NINTERMEDIATES, NINPUTS)
print("reverse-mode:", ops)

env = VertexGame(GS, nedges, NINPUTS, NINTERMEDIATES, NOUTPUTS)

q_net = eqx.nn.MLP(NINTERMEDIATES, NINTERMEDIATES, 256, 2, key=key)
target_net = copy.deepcopy(q_net)


@eqx.filter_jit
def get_action(model, obs, key):
    eps_key, sample_key = jrand.split(key, 2)
    rn = jrand.uniform(eps_key, (1,))[0]
    q = model(obs, key=key)
    masked_q = jnp.where(jnp.less(obs, 1.), q, -500.)
    action = lax.cond(rn > EPS, 
                    lambda q: jnp.argmax(q),
                    lambda q: jrand.categorical(sample_key, logits=q),
                    masked_q)
    return action


def predict(model, obs, key):
    return lax.stop_gradient(model(obs, key=key))


def rollout(model, size, key):
    gs = env.reset()
    def q_step(gs, key):
        action = get_action(model, gs.state, key)
        next_gs, reward, done = env.step(gs, action)
        return next_gs, (gs.state, action, next_gs.state, reward, done)

    keys = jrand.split(key, size)
    _, transitions = lax.scan(q_step, gs, keys, size)
    
    return transitions


# Needs optimization
def initialize_replay_buffer(model, size, key):
    transitions = rollout(model, size, key)
    
    for i in range(size):
        replay_buffer.push((transitions[0][i],
                            transitions[1][i],
                            transitions[2][i],
                            transitions[3][i],
                            transitions[4][i]))
        

# TODO something does not seem to work
@ft.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0))
def calc_loss(q_model, 
                target_model, 
                state, 
                action, 
                next_state, 
                reward, 
                done,
                key):
    keys = jrand.split(key, 3)
    q_values = q_model(state, key=keys[0])
    
    def q_values_done(q_values):
        return q_values.at[action].set(reward)
    
    def q_values_next(q_values):
        q_values_next = predict(target_model, next_state, keys[1])
        rew = reward + GAMMA*q_values_next.max()
        return q_values.at[action].set(rew)
    
    q_target = lax.cond(done, 
                        lambda q: q_values_done(q),
                        lambda q: q_values_next(q),
                        q_values)

    q_prediction = q_model(state, key=keys[2])
    # calculate MSE loss
    return jnp.square(q_prediction - q_target).sum()


@eqx.filter_value_and_grad
def loss_and_grad(q_model, target_model, samples, key):
    keys = jrand.split(key, BATCHSIZE)
    states, actions, next_states, rewards, dones = samples
    loss_batch = calc_loss(q_model, 
                        target_model, 
                        states,
                        actions, 
                        next_states,
                        rewards,
                        dones,
                        keys)
    return loss_batch.mean()


@eqx.filter_jit
def train(q_model, target_model, optim, opt_state, samples, key):
    loss, grads = loss_and_grad(q_model, target_model, samples, key)
    updates, opt_state = optim.update(grads, opt_state)
    q_model = eqx.apply_updates(q_model, updates)
    return q_model, opt_state, loss


start = time.time()
initialize_replay_buffer(q_net, REPLAY_SIZE, key)
end = time.time()
print(end - start, "buffer init time")


# optimizer
optim = optax.adam(LR)
opt_state = optim.init(eqx.filter(q_net, eqx.is_inexact_array))

# Main training loop
losses_list, reward_list, episode_len_list, epsilon_list  = [], [], [], []

pbar = tqdm(range(EPISODES))
for episode in pbar:
    if episode % UPDATE_FREQUENCY == 0:
        # copy params from q_net pytree
        target_net = jtu.tree_map(lambda x: x, q_net)

    gs = env.reset()
    done, losses, ep_len, rew = False, 0., 0, 0.
    # TODO this way, the agent plays exactly one game!
    while not done:
        # agent interacts with the env for NINTERMEDIATES steps and refills the replay buffer        
        ep_len += 1 
        act_key, key = jrand.split(key, 2)
        action = get_action(q_net, gs.state, act_key)
        
        next_gs, reward, done = env.step(gs, action)
        
        replay_buffer.push((gs.state, action, next_gs.state, reward, done))
       
        gs = next_gs
        rew += reward
        
        samples = replay_buffer.sample(BATCHSIZE)
        
        q_net, opt_state, loss = train(q_net, target_net, optim, opt_state, samples, key)
        losses += loss 
    # epsilon is adjusted for exploration vs. exploitation     
    EPS = max(EPS*0.99, 0.01)
    
    losses_list.append(losses/ep_len)
    reward_list.append(rew)
    episode_len_list.append(ep_len)
    epsilon_list.append(EPS)
    pbar.set_description(f"episode: {episode}, loss: {losses}, return: {rew}, eps: {EPS}")
    
print("Average reward:", np.mean(reward_list))
print("Average duration:", np.mean(episode_len_list))

plt.plot(reward_list)
plt.title("DQN on Helmholtz Free Energy")
plt.xlabel("episodes")
plt.ylabel("return/# of computations")
plt.savefig("rews.png")

