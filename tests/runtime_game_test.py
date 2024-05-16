import os
import jax
import jax.numpy as jnp
import jax.lax as lax

from graphax.examples import RoeFlux_1d, Helmholtz

from alphagrad.vertexgame.runtime_game import RuntimeGame, _get_reward, _step

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

BATCHSIZE = 4

xs = [jnp.array([0.15, 0.15, 0.2, 0.3])]
env = RuntimeGame(1000, Helmholtz, *xs)

state = env.reset()
s0 = [state[0] for _ in range(BATCHSIZE)]
s1 = [state[1] for _ in range(BATCHSIZE)]
states= (jnp.stack(s0), jnp.stack(s1))

# state, reward, terminated = env.step(state, 1)
# state, reward, terminated = env.step(state, 4)
# state, reward, terminated = env.step(state, 2)
# state, reward, terminated = env.step(state, 3)
# state, reward, terminated = env.step(state, 0)
# print(state, reward, terminated)


from multiprocessing import Process, Manager, set_start_method

def _step(i, state, action, return_dict):
    return_dict[i] = env.step(state, action)

### Multi-processing test
def env_steps(states, actions):
    states = jax.device_put(states, jax.devices('cpu')[0])
    actions = jax.device_put(actions, jax.devices('cpu')[0])
    next_obs, next_act_seqs, rewards, dones = [], [], [], []
    _states = [(states[0][i], states[1][i]) for i in range(BATCHSIZE)]

    manager = Manager()
    out = manager.dict()
    procs = []

    # instantiating process with arguments
    for i, (state, action) in enumerate(zip(_states, actions)):
        print("starting")
        proc = Process(target=_step, args=(i, state, action, out))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        print("joining")
        proc.join()
    print(out.values())
    next_obs = [out[i][0][0] for i in range(BATCHSIZE)]
    next_act_seqs = [out[i][0][1] for i in range(BATCHSIZE)]
    rewards = [out[i][1] for i in range(BATCHSIZE)]
    dones = [out[i][2] for i in range(BATCHSIZE)]
    next_states = (jnp.stack(next_obs), jnp.stack(next_act_seqs))
    
    return next_states, jnp.array(rewards), jnp.array(dones)

if __name__ == '__main__':
    set_start_method("spawn")
    states, reward, term = env_steps(states, jnp.array([1, 1, 1, 1]))
    states, reward, term = env_steps(states, jnp.array([4, 4, 4, 4]))
    states, reward, term = env_steps(states, jnp.array([2, 2, 2, 2]))
    states, reward, term = env_steps(states, jnp.array([3, 3, 3, 3]))
    states, reward, term = env_steps(states, jnp.array([0, 0, 0, 0]))
    # states, reward, term = env_steps(states, jnp.array([5, 5, 5, 5]))
    print(reward)

