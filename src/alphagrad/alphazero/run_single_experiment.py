import os
import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.examples import RoeFlux_1d

from alphagrad.vertexgame import make_graph
from alphagrad.transformer.sequential_transformer import SequentialTransformerModel
from alphagrad.alphazero.single_graph_experiment import SingleGraphExperiment, Paths

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
print("CPUs", jax.device_count("cpu"), jax.devices("cpu"))

print(jax.device_count(), jax.devices())
print(jax.local_device_count(), jax.local_devices())

xs = [1.]*6
graph = make_graph(RoeFlux_1d, *xs)
print(graph.shape)

paths = Paths(task_file_path="./data/task_samples",
            benchmark_file_path="./data/benchmark_samples",
            chckpt_path="./checkpoints/",
            weights_path=None)

key = jrand.PRNGKey(42)
INFO = jnp.array([6, 101, 3])
model = SequentialTransformerModel(INFO, 128, 3, 8,
									ff_dim=1024,
									num_layers_policy=2,
									policy_ff_dims=[1024, 512],
									value_ff_dims=[1024, 512, 256], 
									key=key)
experiment = SingleGraphExperiment("Test", model, graph, paths, jax.devices("gpu"), jax.devices("cpu"),
                                	num_inputs=6, num_actions=101, num_outputs=6)

experiment.run()

