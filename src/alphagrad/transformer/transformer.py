from typing import Optional
import numpy as np

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx

from .encoder import Encoder
from .decoder import Decoder


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

