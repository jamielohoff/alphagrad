import os
import copy
import time
import functools as ft
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import chex
import equinox as eqx
import rlax
import optax

from graphax.game import VertexGame
from graphax.elimination import forward, reverse, eliminate
from graphax.examples.random import construct_random_graph
from graphax.examples.helmholtz import construct_Helmholtz

from replay_memory import EdgeReplayMemory
from transformer import PositionalEncoder
from encoder import Encoder


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

key = jrand.PRNGKey(123)
EPISODES = 500
LR = 3.
BATCHSIZE = 64
REPLAY_SIZE = 512
EPS = .75 # the higher the more exploration
NINPUTS = 10
NINTERMEDIATES = 20
NOUTPUTS = 1
M = (NINPUTS+NINTERMEDIATES)*(NINTERMEDIATES+NOUTPUTS)
UPDATE_FREQUENCY = 10
GAMMA = 1.

GS = construct_random_graph(NINPUTS, NINTERMEDIATES, NOUTPUTS, key)

# create a replay buffer
replay_buffer = EdgeReplayMemory(REPLAY_SIZE)

forward_gs = copy.deepcopy(GS)
_, ops = forward(forward_gs, GS.get_info())
print("forward-mode:", ops)

reverse_gs = copy.deepcopy(GS)
_, ops = reverse(reverse_gs, GS.get_info())
print("reverse-mode:", ops)

env = VertexGame(GS)   

@eqx.filter_jit
def random_policy(env, key):
    gs = env.reset()
    def random_step(gs, key):
        sample_key, key = jrand.split(key, 2)
        one_hot_state = jnn.one_hot(gs.state-1, num_classes=NINTERMEDIATES)
        action_mask = one_hot_state.sum(axis=0)

        masked_q = jnp.where(jnp.less(action_mask, 1.), jnp.ones(NINTERMEDIATES), -500.)
        action = jrand.categorical(sample_key, logits=masked_q)
        next_gs, reward, done = env.step(gs, action)
        
        return next_gs, reward

    keys = jrand.split(key, NINTERMEDIATES)
    _, rewards = lax.scan(random_step, gs, keys, NINTERMEDIATES)
    return rewards.sum() 

best_rew = -1000.
for key in jrand.split(key, 500):
    rew = random_policy(env, key)
    best_rew = rew if rew > best_rew else best_rew
print("Random search:", best_rew)

class SimpleEncoderModel(eqx.Module):
    in_dim: int
    pos_encoder: PositionalEncoder
    embedding: eqx.nn.Embedding
    encoder: Encoder
    linear: eqx.nn.Linear
    
    def __init__(self, 
                num_layers,  
                num_heads, 
                in_dim, 
                ff_dim, 
                dropout=.2, *,
                key) -> None:
        super().__init__()
        self.in_dim = in_dim
        keys = jrand.split(key, 3)
        self.pos_encoder = PositionalEncoder(in_dim)
        self.embedding = eqx.nn.Embedding(M, in_dim, key=keys[0])
        self.encoder = Encoder(num_layers, 
                                num_heads,
                                in_dim, 
                                ff_dim, 
                                dropout=dropout,
                                key=keys[1])
        linear = eqx.nn.Linear(in_dim, NINTERMEDIATES, key=keys[2])
        self.linear = jax.vmap(linear, in_axes=(0,))
        
    def __call__(self, xs: chex.Array, key: chex.PRNGKey, mask=None):
        xs = xs.astype(jnp.int32)
        xs = self.embedding(xs) #  * jnp.sqrt(self.in_dim)
        xs = self.pos_encoder(xs)
        xs = self.encoder(xs, key=key, mask=mask)
        return self.linear(xs)


q_net = SimpleEncoderModel(1, 6, 32, 1024, key=key)
target_net = copy.deepcopy(q_net)

def generate_square_subsequent_mask(num_heads: int, sz: int) -> chex.Array:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    mask = np.tril(np.ones((sz, sz)), k=1)[jnp.newaxis, :, :]
    return jnp.repeat(mask, num_heads, axis=0).astype(bool)


@eqx.filter_jit
def get_action(model, state, edges, key):
    eps_key, sample_key = jrand.split(key, 2)
    rn = jrand.uniform(eps_key, (1,))[0]
    num_steps = jnp.count_nonzero(state)

    obs = edges.flatten()
    q = model(obs, key=key)
    one_hot_state = jnn.one_hot(state-1, num_classes=NINTERMEDIATES)
    action_mask = one_hot_state.sum(axis=0)

    masked_q = jnp.where(jnp.less(action_mask, 1.), q[num_steps-1], -500.)
    action = lax.cond(rn > EPS, 
                    lambda q: jnp.argmax(q),
                    lambda q: jrand.categorical(sample_key, logits=q),
                    masked_q)
    return action


def predict(model, state, edges, key):
    obs = edges.flatten()
    return lax.stop_gradient(model(obs, key=key))


def rollout(model, size, key):
    gs = env.reset()
    def q_step(gs, key):
        action = get_action(model, gs.state, gs.edges, key)
        next_gs, reward, done = env.step(gs, action)
        return next_gs, (gs.state, gs.edges, action, next_gs.state, next_gs.edges, reward, done)

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
                            transitions[4][i],
                            transitions[5][i],
                            transitions[6][i]))
        

@ft.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0))
def calc_loss(q_model, 
                target_model, 
                state, 
                edges,
                action, 
                next_state, 
                next_edges,
                reward, 
                done,
                key):
    keys = jrand.split(key, 3)
    num_steps = jnp.count_nonzero(state)
    obs = edges.flatten()
    q_values = q_model(obs, key=keys[0])
    
    def q_values_done(q_values):
        return q_values.at[action].set(reward)
    
    def q_values_next(q_values):
        q_values_next = predict(target_model, next_state, next_edges, keys[1])
        rew = reward + GAMMA*q_values_next.max()
        return q_values.at[action].set(rew)
    
    q_target = lax.cond(done, 
                        lambda q: q_values_done(q),
                        lambda q: q_values_next(q),
                        q_values.at[num_steps].get())

    q_prediction = q_model(obs, key=keys[2]).at[num_steps].get()
    # calculate MSE loss
    return jnp.square(q_prediction - q_target).sum()


@eqx.filter_value_and_grad
def loss_and_grad(q_model, target_model, samples, key):
    keys = jrand.split(key, BATCHSIZE)
    states, edges, actions, next_states, next_edges, rewards, dones = samples
    loss_batch = calc_loss(q_model, 
                        target_model, 
                        states,
                        edges,
                        actions, 
                        next_states,
                        next_edges,
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
schedule = optax.exponential_decay(init_value=LR, 
                                    transition_steps=NINTERMEDIATES,
                                    decay_rate=.95, 
                                    staircase=True)
optim = optax.chain(optax.adam(learning_rate=schedule)) # optax.clip_by_global_norm(.5), 
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
        action = get_action(q_net, gs.state, gs.edges, act_key)
        
        next_gs, reward, done = env.step(gs, action)
        replay_buffer.push((gs.state, gs.edges, action, next_gs.state, next_gs.edges, reward, done))
        
        gs = next_gs
        rew += reward
        
        samples = replay_buffer.sample(BATCHSIZE)
        
        q_net, opt_state, loss = train(q_net, target_net, optim, opt_state, samples, key)
        losses += loss 
    # epsilon is adjusted for exploration vs. exploitation     
    EPS = EPS*0.99
    
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

