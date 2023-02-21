from typing import Sequence, Callable, Optional

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx


class EncoderLayer(eqx.Module):
    """
    Implementation of a single Encoder layer as in "Attention is all you need",
    i.e. it consists of one Multihead attention block with residual connections
    followed by a two-layer fully connected.
    """
    attn_layer: eqx.nn.MultiheadAttention
    attn_norm: eqx.nn.LayerNorm
    attn_dropout: eqx.nn.Dropout
    
    ff_layer1: eqx.nn.Linear
    ff_activation_fn: eqx.nn.Lambda = eqx.static_field()
    ff_layer2: eqx.nn.Linear
    ff_norm: eqx.nn.LayerNorm
    ff_dropout: eqx.nn.Dropout
    
    output_activation_fn: eqx.nn.Lambda = eqx.static_field()

    def __init__(self,  
                num_heads: int,
                in_dim: int, 
                ff_dim: int, 
                dropout: float = .1,
                use_bias: bool = False,
                ff_activation_fn: Callable = jnn.relu,
                output_activation_fn: Callable = lambda x: x, *,
                key: chex.PRNGKey, 
                **kwargs) -> None:
        super().__init__()
        keys = jrand.split(key, 3)
        # Self-attention block
        self.attn_layer = eqx.nn.MultiheadAttention(num_heads, 
                                                    in_dim, 
                                                    key=keys[0],
                                                    **kwargs)
        self.attn_norm = eqx.nn.LayerNorm(in_dim)
        self.attn_dropout = eqx.nn.Dropout(p=dropout)
        
        # Feed-forward block
        ff_layer1 = eqx.nn.Linear(in_dim, 
                                    ff_dim, 
                                    use_bias=use_bias, 
                                    key=keys[1])
        self.ff_layer1 = jax.vmap(ff_layer1, in_axes=(0,))
        self.ff_activation_fn = eqx.nn.Lambda(ff_activation_fn)
        ff_layer2 = eqx.nn.Linear(ff_dim, 
                                    in_dim, 
                                    use_bias=use_bias, 
                                    key=keys[2])
        self.ff_layer2 = jax.vmap(ff_layer2, in_axes=(0,))

        self.ff_norm = eqx.nn.LayerNorm(in_dim)
        self.ff_dropout = eqx.nn.Dropout(p=dropout)
        self.output_activation_fn = eqx.nn.Lambda(output_activation_fn)

    def __call__(self, 
                xs: chex.Array, 
                mask: Optional[chex.Array] = None, *, 
                key: chex.PRNGKey) -> chex.Array:
        keys = jrand.split(key, 3)
        
        out = self.attn_norm(xs)
        out = self.attn_layer(out, out, out, mask=mask, key=keys[0])
        out = xs + self.attn_dropout(out, key=keys[1])

        ff_out = self.ff_norm(out)
        ff_out = self.ff_layer1(ff_out)
        ff_out = self.ff_activation_fn(ff_out)
        ff_out = self.ff_layer2(ff_out)
        
        ff_out = out + self.ff_dropout(ff_out, key=keys[2])
        return self.output_activation_fn(ff_out)


class Encoder(eqx.Module):
    """
    Stack of ´num_layers´ transformer encoder layers/cells. 
    """
    num_layers: int
    layers: Sequence[eqx.Module]

    def __init__(self, 
                num_layers: int, 
                num_heads: int, 
                in_dim: int,
                ff_dim: int,
                dropout: int = .1,
                use_bias: bool = False, *,
                key: chex.PRNGKey, **kwargs) -> None:
        super().__init__()
        keys = jrand.split(key, num_layers)
        self.num_layers = num_layers
        self.layers = [EncoderLayer(num_heads, in_dim, ff_dim, dropout, use_bias, key=k, **kwargs) for k in keys]

    def __call__(self, 
                xs: chex.Array, 
                mask: Optional[chex.Array] = None, *, 
                key: chex.PRNGKey):
        keys = jrand.split(key, self.num_layers)
        for k, layer in zip(keys, self.layers):
            xs = layer(xs, key=k, mask=mask)
        return xs

