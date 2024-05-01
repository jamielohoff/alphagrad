import jax
import jax.numpy as jnp
import jax.random as jrand

from alphagrad.resnet.resnet import AlphaZeroModel


key = jrand.PRNGKey(0)
model = AlphaZeroModel(101, key)
xs = jrand.normal(key, (16, 5, 108, 101))
print(jax.vmap(model, in_axes=(0, None))(xs, key))

