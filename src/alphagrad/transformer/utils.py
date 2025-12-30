from typing import Callable, Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import numpy as np
import equinox as eqx

Array = jax.Array
PRNGKey = jax.random.PRNGKey

class PositionalEncoder(eqx.Module):
    """
    Positional encoder similar to the "Attention is all you need" paper.
    """
    pe: Array

    def __init__(self, in_dim: int, seq_len: int, n: int = 10_000):
        # Create matrix of [SeqLen, TokenSize] representing the positional 
        # encoding for max_len inputs
        
        pe = np.zeros((in_dim, seq_len))
        position = np.arange(0, seq_len, dtype=np.float32)[None, :]
        div_term = np.power(n, np.arange(0, in_dim, 2) / in_dim)[:, None]
        pe[0::2, :] = np.sin(position * div_term)
        pe[1::2, :] = np.cos(position * div_term)
        self.pe = jax.device_put(pe)

    def __call__(self, x: Array):
        return x + self.pe[:x.shape[0], :]
    
    
class MLP(eqx.Module):
    """
    Simple implementation of a MLP in JAX
    """
    layers: eqx.nn.Sequential
    
    def __init__(
        self, 
        in_size: int,
        out_size: int,
        layers: Sequence[int],
        activation: Callable = jnn.swish,
        final_activation: Callable = lambda x: x, 
        *,
        key: PRNGKey
    ) -> None:
        super().__init__()
        keys = jrand.split(key, len(layers))
        layer_list = [
            eqx.nn.Linear(in_size, layers[0], key=keys[0]), 
            eqx.nn.Lambda(activation)
        ]
        for i, key in enumerate(keys[1:]):
            layer_list.append(eqx.nn.Linear(layers[i], layers[i+1], key=key))
            layer_list.append(eqx.nn.Lambda(activation))
        layer_list.append(eqx.nn.Linear(layers[-1], out_size, key=keys[-1]))
        layer_list.append(eqx.nn.Lambda(final_activation))
        self.layers = layer_list
        
    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x

