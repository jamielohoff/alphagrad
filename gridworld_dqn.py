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

import equinox as eqx
import optax

from replay_memory import ReplayMemory
from gridworld import GridworldGame2D, GridworldState


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

key = jrand.PRNGKey(123)
EPISODES = 300
LR = 1e-3
BATCHSIZE = 64
REPLAY_SIZE = 512
EPS = .5 # the higher the more exploration
UPDATE_FREQUENCY = 10
GAMMA = 1.

goal = jnp.array([0, 4])
world = jnp.array( [[0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 0, 0]])  

env = GridworldGame2D(world, goal)
replay_buffer = ReplayMemory(REPLAY_SIZE)

q_net = eqx.nn.MLP(25, 4, 32, 2, key=key)
target_net = copy.deepcopy(q_net)


@eqx.filter_jit
def get_action(model, obs, key):
    eps_key, sample_key = jrand.split(key, 2)
    rn = jrand.uniform(eps_key, (1,))[0]
    qvalues = model(obs, key=key)
    # epsilon-greedy is sufficient for Q-Learning
    action = lax.cond(rn > EPS, 
                    lambda: jnp.argmax(qvalues),
                    lambda: jrand.categorical(sample_key, logits=qvalues))
    return action


def predict(model, obs, key):
    return lax.stop_gradient(model(obs, key=key))


def rollout(model, size, key):
    def q_step(carry, key):
        state, obs = carry
        action = get_action(model, obs, key)
        next_state, next_obs, reward, done = env.step(state, action)
        next_carry = (next_state, next_obs)
        return next_carry, (obs, action, next_obs, reward, done)

    state = env.reset(key)
    obs = env.get_obs(state.position)
    init_carry = (state, obs)
    keys = jrand.split(key, size)
    _, transitions = lax.scan(q_step, init_carry, keys, size)
    
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
                obs, 
                action, 
                next_obs, 
                reward, 
                done,
                key):
    keys = jrand.split(key, 3)
    q_values = q_model(obs, key=keys[0])
    
    # Q-Learning algorithm implementation
    
    def q_values_done(q_values):
        return q_values.at[action].set(reward)
    
    def q_values_next(q_values):
        q_values_next = predict(target_model, next_obs, keys[1])
        # TD evaluation using  maximum of action-value function
        rew = reward + GAMMA*q_values_next.max()
        return q_values.at[action].set(rew)
    
    q_target = lax.cond(done, 
                        lambda q: q_values_done(q),
                        lambda q: q_values_next(q),
                        q_values)

    q_prediction = q_model(obs, key=keys[2])
    # calculate MSE loss which has the correct gradient but has no deeper meaning
    return jnp.square(q_prediction - q_target).sum()


@eqx.filter_value_and_grad
def loss_and_grad(q_model, target_model, samples, key):
    keys = jrand.split(key, BATCHSIZE)
    obs, actions, next_obs, rewards, dones = samples
    loss_batch = calc_loss(q_model, 
                        target_model, 
                        obs,
                        actions, 
                        next_obs,
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

def solve_maze(network, starting_point):
    state = GridworldState(t=0, position=starting_point)
    obs = env.get_obs(state.position)

    rew = 0
    solved = False
    while not solved:
        qvalues = network(obs)
        action = jnp.argmax(qvalues).astype(jnp.int32)
        state, obs, reward, done = env.step(state, action)
        rew += reward
        solved = done

        print(state.position, action, qvalues)
    print(state.t)
    return rew

pbar = tqdm(range(EPISODES))
for episode in pbar:
    if episode % UPDATE_FREQUENCY == 0:
        # copy params from q_net pytree
        target_net = jtu.tree_map(lambda x: x, q_net)

    reset_key, key = jrand.split(key)
    state = env.reset(reset_key)
    obs = env.get_obs(state.position)
    done, losses, ep_len, rew = False, 0., 0, 0.
    # TODO this way, the agent plays exactly one game!
    while not done:
        # agent interacts with the env for NINTERMEDIATES steps and refills the replay buffer        
        ep_len += 1 
        act_key, key = jrand.split(key, 2)
        action = get_action(q_net, obs, act_key)
        
        next_state, next_obs, reward, done = env.step(state, action)
        
        replay_buffer.push((obs, action, next_obs, reward, done))

        state = next_state
        obs = next_obs
        rew += reward
        
        samples = replay_buffer.sample(BATCHSIZE)
        
        q_net, opt_state, loss = train(q_net, target_net, optim, opt_state, samples, key)
        losses += loss
        if ep_len >= 100:
            break
    # epsilon is adjusted for exploration vs. exploitation     
    EPS = max(EPS*0.99, 0.001)
    
    rew = solve_maze(q_net, jnp.array([0, 0]))
    
    losses_list.append(losses/ep_len)
    reward_list.append(rew)
    episode_len_list.append(ep_len)
    epsilon_list.append(EPS)
    pbar.set_description(f"episode: {episode}, loss: {losses}, return: {rew}, eps: {EPS}")
    
print("Average reward:", np.mean(reward_list))
print("Average duration:", np.mean(episode_len_list))

plt.plot(reward_list)
plt.title("DQN on Gridworld")
plt.xlabel("episodes")
plt.ylabel("return/# of computations")
plt.savefig("gridworld_rews.png")

