from typing import Optional
import numpy as np

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx

from encoder import Encoder
from decoder import Decoder


class PositionalEncoder(eqx.Module):
    """
    Positional encoder similar to the "Attention is all you need" paper.
    """
    pe: chex.Array

    def __init__(self, in_dim: int, seq_len: int):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        
        pe = np.zeros((seq_len, in_dim))
        position = np.arange(0, seq_len, dtype=np.float32)[:,None]
        div_term = np.exp(np.arange(0, in_dim, 2) * (-jnp.log(2*seq_len) / in_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = jax.device_put(pe)

    def __call__(self, x: chex.Array):
        return x + self.pe[:x.shape[0], :]


class Transformer(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    def __init__(self, 
                num_layers: int,
                num_heads: int, 
                in_dim: int, 
                ff_dim: int, 
                dropout: float = 0.1,
                use_bias: bool = False, *,
                key: chex.PRNGKey, **kwargs) -> None:

        self.encoder = Encoder(num_layers, 
                                num_heads=num_heads, 
                                in_dim=in_dim, 
                                ff_dim=ff_dim, 
                                dropout=dropout, 
                                use_bias=use_bias,
                                key=key,
                                **kwargs)

        self.decoder = Decoder(num_layers, 
                                num_heads=num_heads, 
                                in_dim=in_dim, 
                                ff_dim=ff_dim, 
                                dropout=dropout, 
                                use_bias=use_bias,
                                key=key, 
                                **kwargs)

    def __call__(self, 
                xs: chex.Array, 
                target: chex.Array, 
                mask: Optional[chex.Array] = None, *,
                key: chex.Array):
        ekey, dkey = jrand.split(key, 2)
        return self.decoder(target, self.encoder(xs, ekey), dkey, mask=mask)

