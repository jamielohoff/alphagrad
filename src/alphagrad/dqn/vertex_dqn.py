import os
import copy
from tqdm import tqdm
from functools import partial, reduce
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import wandb

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import equinox as eqx
import distrax
import optax
import flashbax as fbx
from chex import Array, PRNGKey

from graphax.examples import RoeFlux_1d, RobotArm_6DOF, EncoderDecoder, ADALIF_SNN, f, g
from alphagrad.vertexgame import step, make_graph, forward, reverse, cross_country
from alphagrad.vertexgame.transforms import minimal_markowitz

from alphagrad.transformer import MLP
from alphagrad.transformer import Encoder
from alphagrad.transformer import PositionalEncoder

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(3)


# RObotArm at 226 mults wtf...

key = jrand.PRNGKey(42)
FNAME = "RobotArm_vertex_dqn.png"
NUM_ENVS = 8
EPISODES = 500
LR = 1e-3
BATCHSIZE = 256
REPLAY_SIZE = 100000
EPS = .65 # the higher the more exploration
GAMMA = 1.

# Instantiate the environment & its settings.
xs = [.01, .02, .02, .01, .03, .03]
graph = make_graph(RobotArm_6DOF, *xs) # make_graph(RoeFlux_1d, *xs) # 

i = graph.at[0, 0, 0].get()
v = graph.at[0, 0, 1].get() + graph.at[0, 0, 2].get()
o = graph.at[0, 0, 2].get()
INFO = jnp.array([i, v, o])
ROLLOUT_LENGTH = int(graph.at[0, 0, 1].get())
print(ROLLOUT_LENGTH)
print("info", INFO)

_, fwd_fmas = forward(graph)
_, rev_fmas = reverse(graph)
mM_order = minimal_markowitz(graph, int(graph.at[0, 0, 1].get()))
print("mM_order", [int(i) for i in mM_order])
out, _ = cross_country(mM_order, graph)
print(fwd_fmas, rev_fmas, out[1])


# Initialize replay buffer
OBS_SHAPE = reduce(lambda x, y: x*y, graph.shape)
NUM_ACTIONS = graph.shape[-1]
item_prototype = jnp.zeros(2*OBS_SHAPE+3, dtype=jnp.int32)
item_prototype = jax.device_put(item_prototype, jax.devices("cpu")[0])
replay_buffer = fbx.make_item_buffer(max_length=REPLAY_SIZE, 
                                     min_length=1, 
                                     sample_batch_size=BATCHSIZE)
buffer_state = replay_buffer.init(item_prototype)


# Model definition
class SequentialTransformer(eqx.Module):
    num_heads: int
    pos_enc: PositionalEncoder
    encoder: Encoder
    value_head: MLP
    
    def __init__(self, 
                in_dim: int,
                seq_len: int,
                num_layers: int,
                num_heads: int,
                ff_dim: int = 1024,
                value_ff_dims: Sequence[int] = [512, 256],
                key: PRNGKey = None) -> None:
        super().__init__()  
        self.num_heads = num_heads      
        e_key, v_key = jrand.split(key, 2)
        
        self.pos_enc = PositionalEncoder(in_dim, seq_len)
        
        self.encoder = Encoder(num_layers=num_layers,
                                num_heads=num_heads,
                                in_dim=in_dim,
                                ff_dim=ff_dim,
                                key=e_key)
        
        self.value_head = MLP(in_dim, 1, value_ff_dims, key=v_key)
        
    def __call__(self, xs: Array, mask: Array = None, key: PRNGKey = None) -> Array:                    
        # Transpose inputs for equinox attention mechanism
        xs = self.pos_enc(xs).T
        mask = mask.T
        
        # Replicate mask and apply encoder
        replicated_mask = jnp.tile(mask[jnp.newaxis, :, :], (self.num_heads, 1, 1))
        xs = self.encoder(xs, mask=replicated_mask, key=key)
        
        return jax.vmap(self.value_head)(xs).flatten()
    
    
class SequentialTransformerModel(eqx.Module):
    embedding: eqx.nn.Conv2d
    projection: Array
    output_token: Array
    transformer: SequentialTransformer
    
    def __init__(self, 
                info: Sequence[int],
                embedding_dim: int,
                num_layers: int,
                num_heads: int,
                key: PRNGKey = None,
                **kwargs) -> None:
        super().__init__()
        embed_key, token_key, proj_key, tf_key = jrand.split(key, 4)
        self.embedding = eqx.nn.Conv2d(info[1], info[1], (5, 1), stride=(1, 1), key=embed_key)
        self.projection = jrand.normal(proj_key, (info[0]+info[1], embedding_dim))
        self.output_token = jrand.normal(token_key, (info[0]+info[1], 1))
        self.transformer = SequentialTransformer(embedding_dim,
                                                info[1], 
                                                num_layers, 
                                                num_heads, 
                                                key=tf_key, 
                                                **kwargs)
    
    def __call__(self, xs: Array, key: PRNGKey = None) -> Array:
        output_mask = xs.at[2, 0, :].get()
        vertex_mask = xs.at[1, 0, :].get() - output_mask
        attn_mask = jnp.logical_or(vertex_mask.reshape(1, -1), vertex_mask.reshape(-1, 1))
        
        output_token_mask = jnp.where(xs.at[2, 0, :].get() > 0, self.output_token, 0.)
        edges = xs.at[:, 1:, :].get() + output_token_mask[jnp.newaxis, :, :]
        edges = edges.astype(jnp.float32)
        
        embeddings = self.embedding(edges.transpose(2, 0, 1)).squeeze()
        embeddings = jax.vmap(jnp.matmul, in_axes=(0, None))(embeddings, self.projection)
        return self.transformer(embeddings.T, mask=attn_mask, key=key)
        

# Initialize the environment
q_net = SequentialTransformerModel(INFO, 128, 3, 8, key)
target_net = jtu.tree_map(lambda x: x, q_net) # copy.deepcopy(q_net)


# Filtering low-value samples, TODO define low-value samples
def fill_buffer(buffer_state, samples):
    def loop_fn(buffer_state, sample):
        # new_buffer_state = lax.cond(sample[-1] > 0,
        #                             lambda bs: replay_buffer.add(bs, sample),
        #                             lambda bs: bs,
        #                             buffer_state)
        new_buffer_state = replay_buffer.add(buffer_state, sample)
        return new_buffer_state, None
    updated_buffer_state, _ = lax.scan(loop_fn, buffer_state, samples)
    return updated_buffer_state


def get_action(model, state, key):
    eps_key, sample_key = jrand.split(key, 2)
    rn = jrand.uniform(eps_key, ())
    qvalues = model(state, key=key)
    
    mask = 1. - state.at[1, 0, :].get()
    masked_qvalues = jnp.where(mask > 0, qvalues, -jnp.inf)
    
    distribution = distrax.Categorical(logits=masked_qvalues)
    # Epsilon-greedy is sufficient for Q-Learning
    action = lax.select(rn > EPS, 
                        jnp.argmax(masked_qvalues),
                        distribution.sample(seed=sample_key))
    return action


def init_carry(keys):
    return jnp.tile(graph[jnp.newaxis, ...], (len(keys), 1, 1, 1))


def predict(model, obs, key):
    return lax.stop_gradient(model(obs, key=key))
        

# TODO something does not seem to work
@partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0))
def loss_fn(q_model, target_model, state, action, next_obs, reward, done, key):
    keys = jrand.split(key, 3)
    q_values = q_model(state, key=keys[0])
    
    # Q-Learning algorithm implementation
    def q_values_done(q_values):
        return q_values.at[action.astype(jnp.int32)].set(reward)
    
    def q_values_next(q_values):
        q_values_next = predict(target_model, next_obs, keys[1])
        # TD evaluation using  maximum of action-value function
        rew = reward + GAMMA*q_values_next.max()
        return q_values.at[action.astype(jnp.int32)].set(rew)
    
    q_target = lax.cond(done, 
                        lambda q: q_values_done(q),
                        lambda q: q_values_next(q),
                        q_values)

    q_prediction = q_model(state, key=keys[2])
    # calculate MSE loss which has the correct gradient but has no deeper meaning
    return jnp.square(q_prediction-q_target).sum()


@eqx.filter_value_and_grad
def loss_and_grad(q_model, target_model, samples, key):
    keys = jrand.split(key, BATCHSIZE)
    obs = samples[:, :OBS_SHAPE]
    obs = obs.reshape(-1, *graph.shape)
    
    actions = samples[:, OBS_SHAPE]
    next_obs = samples[:, OBS_SHAPE+1:2*OBS_SHAPE+1]
    next_obs = next_obs.reshape(-1, *graph.shape)
    
    rewards = samples[:, 2*OBS_SHAPE+1]
    dones = samples[:, 2*OBS_SHAPE+2]
    
    loss_batch = loss_fn(q_model, target_model, obs, actions, next_obs, rewards, dones, keys)
    return loss_batch.mean()


@eqx.filter_jit
def train_fn(q_model, target_model, optim, opt_state, samples, key):
    loss, grads = loss_and_grad(q_model, target_model, samples, key)
    updates, opt_state = optim.update(grads, opt_state)
    q_model = eqx.apply_updates(q_model, updates)
    return q_model, opt_state, loss


@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, 0))
def test_fn(network, key):
    def loop_fn(carry, key):
        ret, state = carry
        subkey, key = jrand.split(key, 2)
        qvalues = network(state, key=subkey)
        mask = 1. - state.at[1, 0, :].get()
        masked_qvalues = jnp.where(mask > 0, qvalues, -jnp.inf)
        
        action = jnp.argmax(masked_qvalues).astype(jnp.int32)
        next_state, reward, done = step(state, action)
        ret += reward
        return (ret, next_state), action
    output, act_seq = lax.scan(loop_fn, (0., graph), jrand.split(key, ROLLOUT_LENGTH))
    return output[0], act_seq


@partial(jax.vmap, in_axes=(None, 0, 0))
def env_interaction_step(q_net, state, key):
    action = get_action(q_net, state, key)
    next_state, reward, done = step(state, action)
    trajectories = jnp.hstack((state.flatten(), action, next_state.flatten(), reward, done))
    return next_state, trajectories


# Exponential moving average for target network update
@eqx.filter_jit
def ema_target_net_update(target_net, q_net, tau=0.005):
    filtered_q_net = eqx.filter(q_net, eqx.is_inexact_array)
    def update_fn(target_params, q_params):
        if q_params is None:
            return target_params
        else:
            return tau * q_params + (1. - tau) * target_params
        
    return jtu.tree_map(update_fn, target_net, filtered_q_net)


def is_valid_sequence(seq):
    unique_vertices = set(seq)
    if len(unique_vertices) != ROLLOUT_LENGTH:
        return False
    else:
        return True


# Tracking different RL metrics
def plot_metrics(num_samples, ret):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Reinforcement learning metrics")
    axs[0, 0].set_title("return")
    axs[0, 0].plot(num_samples, ret)
    axs[0, 0].axhline(y=-fwd_fmas, color='r', linestyle='--')
    axs[0, 0].axhline(y=-rev_fmas, color='k', linestyle='--')
    axs[0, 0].axhline(y=-out[1], color='g', linestyle='--')
    
    # axs[0, 1].set_title("kl div")
    # axs[0, 1].plot(num_samples, kl)

    # axs[0, 2].set_title("mean entropy")
    # axs[0, 2].plot(num_samples, entropy_evo)

    # axs[1, 0].set_title("explained variance")
    # axs[1, 0].plot(num_samples, expl_var)
    
    # axs[1, 1].set_title("value fit quality")
    # axs[1, 1].plot(num_samples, value_fit_quality)
        
    fig.tight_layout()
    plt.savefig(FNAME)
    plt.close()


# Optimizer
optim = optax.adam(LR)
opt_state = optim.init(eqx.filter(q_net, eqx.is_inexact_array))

# Metrics to track
best_global_return = jnp.max(jnp.array([-fwd_fmas, -rev_fmas, -out[1]]))
best_act_seq = None
best_rets, eps_list, num_samples = [], [], []
num_total_samples = 0

# Main training loop
pbar = tqdm(range(EPISODES))
for episode in pbar:       
    reset_key, key = jrand.split(key)     
    
    # TODO put this in a single jittable function
    def rollout_fn(states, key):
        step_keys = jrand.split(key, NUM_ENVS)
        next_states, traj = env_interaction_step(q_net, states, step_keys)
        return next_states, traj
        
    keys = jrand.split(reset_key, ROLLOUT_LENGTH)
    graphs = jnp.tile(graph[jnp.newaxis, ...], (NUM_ENVS, 1, 1, 1))
    _, trajectories = lax.scan(rollout_fn, graphs, keys)
        
        
    trajectories = trajectories.reshape(-1, 2*OBS_SHAPE+3)
    trajectories = jax.device_put(trajectories, jax.devices("cpu")[0])
    
    # we cannot directly write the samples to the replay buffer after all
    # steps since the return can only be calculated at the end of the episode...
    buffer_state = jax.jit(fill_buffer, donate_argnums=0)(buffer_state, trajectories)
    
    samples = jax.jit(replay_buffer.sample)(buffer_state, key)
    samples = jax.device_put(samples.experience, jax.devices("gpu")[0])
    q_net, opt_state, loss = train_fn(q_net, target_net, optim, opt_state, samples, key)

    # epsilon is adjusted for exploration vs. exploitation     
    EPS = max(EPS*0.99, 0.001)
    
    # Test model and calculate the RL metrics
    keys = jrand.split(key, 8)
    returns, act_seqs = test_fn(q_net, keys)
    best_return = jnp.max(returns)
    num_total_samples += BATCHSIZE
    num_samples.append(num_total_samples)
    
    if best_return > best_global_return:
        best_global_return = best_return
        best_act_seq = act_seqs[jnp.argmax(best_return)]
        print(f"New best return: {best_return}")
        vertex_elimination_order = [int(i) for i in best_act_seq]
        print(f"New best action sequence: {vertex_elimination_order}")
        if not is_valid_sequence(vertex_elimination_order):
            print(len(set(vertex_elimination_order)))
            print(f"This is not a valid elimination sequence!")
    
    best_rets.append(best_return)
    eps_list.append(EPS)
    
    if episode % 10 == 0:
        plot_metrics(num_samples, best_rets)
    
    # Update target network with EMA
    target_net = ema_target_net_update(q_net, target_net)
    
    pbar.set_description(f"return: {best_return}, eps: {EPS:.3f}")
    
print("Average reward:", np.mean(best_rets))
vertex_elimination_order = [int(i) for i in best_act_seq]
print(f"Best vertex elimination sequence after {EPISODES} episodes is {vertex_elimination_order} with {best_global_return} multiplications.")


