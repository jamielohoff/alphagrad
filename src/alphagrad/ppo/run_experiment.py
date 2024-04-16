import jax
import jax.numpy as jnp
import jax.random as jrand

from alphagrad.transformer.sequential_transformer import SequentialTransformerModel
from alphagrad.alphazero.experiment import AlphaGradExperiment, Paths


print(jax.device_count())
print(jax.local_device_count())
print(jax.devices())
print(jax.local_devices())


paths = Paths(file_path="./data/_samples",
            task_file_path="./data/task_samples",
            benchmark_file_path="./data/benchmark_samples",
            chckpt_path="./checkpoints/",
            weights_path=None)

key = jrand.PRNGKey(42)
INFO = jnp.array([6, 101, 6])
model = SequentialTransformerModel(INFO, 
									64, 
									2, 
									6,
									ff_dim=1024,
									num_layers_policy=2,
									policy_ff_dims=[1024, 512],
									value_ff_dims=[1024, 512, 128], 
									key=key)
experiment = AlphaGradExperiment("Test", model, paths, jax.devices())

experiment.run()

