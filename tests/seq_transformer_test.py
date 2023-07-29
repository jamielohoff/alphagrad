import os

import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.examples import make_Helmholtz
from alphagrad.sequential_transformer import SequentialTransformerModel

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

key = jrand.PRNGKey(123)
edges = make_Helmholtz()
model = SequentialTransformerModel([1, 6, 1], 128, 1, 6, key=key)
edges = edges.astype(jnp.float32)
print(model(edges, key))

