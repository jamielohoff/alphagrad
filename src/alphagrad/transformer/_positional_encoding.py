import jax
import jax.numpy as jnp

import chex
import equinox as eqx

import numpy as np


class PositionalEncoder(eqx.Module):
    """
    Positional encoder similar to the "Attention is all you need" paper.
    """
    pe: chex.Array

    def __init__(self, in_dim: int, seq_len: int, n: int = 10000):
        # Create matrix of [SeqLen, TokenSize] representing the positional encoding for max_len inputs
        
        pe = np.zeros((in_dim, seq_len))
        position = np.arange(0, seq_len, dtype=np.float32)[None, :]
        div_term = np.power(n, np.arange(0, in_dim, 2) / in_dim)[:, None]
        pe[0::2, :] = np.sin(position * div_term)
        pe[1::2, :] = np.cos(position * div_term)
        self.pe = jax.device_put(pe)

    def __call__(self, xs: chex.Array):
        return xs + self.pe[:xs.shape[0], :]
    
    