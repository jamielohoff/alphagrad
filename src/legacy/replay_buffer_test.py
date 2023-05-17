import numpy as np

from legacy.replay_buffer import AlphaGradReplayMemory

replay_memory = AlphaGradReplayMemory(100, (4, 11, 4))
edges = np.random.randint(low=0, high=10, size=(11, 15, 15))
example = np.concatenate([edges.flatten(), np.zeros((11,)), np.ones((2,))])
examples = np.stack([example for _ in range(10)])
replay_memory.push(examples)
obs, _, _, _ = replay_memory.sample(3)
print(obs[0] == edges)
print(obs[0], obs.shape)
print("###########")
print(edges, edges.shape)

