import os
import copy
import time
import argparse
import functools as ft

import wandb
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

from graphax.game import VertexGame
from graphax.elimination import forward, reverse
from graphax.examples.random import construct_random_graph
from graphax.examples.helmholtz import construct_Helmholtz

from replay_memory import ReplayMemory


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, 
                    default="Maze_DQN_test", help="Name of the experiment.")

parser.add_argument("--gpu", type=str, 
                    default="0", help="GPU identifier.")

parser.add_argument("--seed", type=int,
                    default=123, help="Random seed.")

parser.add_argument("--episodes", type=int, 
                    default=150, help="Number of runs on random data.")

parser.add_argument("--replay_size", type=int, 
                    default=2048, help="Size of the replay buffer.")

parser.add_argument("--rollout_length", type=int, default=50, 
                    help="Duration of the rollout phase of the MCTS algorithm.")

parser.add_argument("--batchsize", type=int, 
                    default=256, help="Learning batchsize.")

parser.add_argument("--rollout_batchsize", type=int,
                    default=16, 
                    help="Batchsize for environment interaction.")

parser.add_argument("--update_frequency", type=int, 
                    default=0., help="Update frequency of the target network.")

parser.add_argument("--lr", type=float, 
                    default=1e-2, help="Learning rate.")

parser.add_argument("--gamma", type=float, 
                    default=.99, help="Discount rate.")

parser.add_argument("--eps", type=float, 
                    default=.75, help="Initial exploration rate for epsilon-greedy.")

args = parser.parse_args()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

wandb.init("Maze_DQN")
wandb.run.name = args.name
wandb.config = vars(args)

key = jrand.PRNGKey(args.seed)

# create a replay buffer
replay_buffer = ReplayMemory(args.replay_size)

NUM_INPUTS = args.num_inputs
NUM_INTERMEDIATES = args.num_intermediates
NUM_OUTPUTS = args.num_outputs
EPS = args.eps

key = jrand.PRNGKey(args.seed)
GS = construct_Helmholtz()


forward_gs = copy.deepcopy(GS)
_, ops = forward(forward_gs, GS.get_info())
print("forward-mode:", ops)


reverse_gs = copy.deepcopy(GS)
_, ops = reverse(reverse_gs, GS.get_info())
print("reverse-mode:", ops)

env = VertexGame(GS)

q_net = eqx.nn.MLP(15*15, 12, 64, 3, key=key)
target_net = copy.deepcopy(q_net)


@eqx.filter_jit
def get_action(model, obs, key):
    eps_key, sample_key = jrand.split(key, 2)
    rn = jrand.uniform(eps_key, (1,))[0]
    q = model(obs, key=key)
    masked_q = jnp.where(jnp.less(obs, 1.), q, -100000.)
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
        rew = reward + args.gamma*q_values_next.max()
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
    keys = jrand.split(key, args.batchsize)
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
initialize_replay_buffer(q_net, args.replay_size, key)
end = time.time()
print(end - start, "buffer init time")


# optimizer
optim = optax.adam(args.lr)
opt_state = optim.init(eqx.filter(q_net, eqx.is_inexact_array))

# Main training loop
losses_list, reward_list, episode_len_list, epsilon_list  = [], [], [], []

pbar = tqdm(range(args.episodes))
for episode in pbar:
    if episode % args.update_frequency == 0:
        # copy params from q_net pytree
        target_net = jtu.tree_map(lambda x: x, q_net)

    gs = env.reset(key)
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
        
        samples = replay_buffer.sample(args.batchsize)
        
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

