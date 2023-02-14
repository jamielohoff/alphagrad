from typing import Sequence, Callable, Optional

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx


class DecoderLayer(eqx.Module):
    masked_attn_layer: eqx.nn.MultiheadAttention
    masked_attn_norm: eqx.nn.LayerNorm
    masked_attn_dropout: eqx.nn.Dropout
    
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
                key: chex.PRNGKey, **kwargs) -> None:
        super().__init__()

        keys = jrand.split(key, 4)

        self.masked_attn_layer = eqx.nn.MultiheadAttention(num_heads, 
                                                            in_dim, 
                                                            key=keys[0],
                                                            **kwargs)
        self.masked_attn_norm = eqx.nn.LayerNorm(in_dim)
        self.masked_attn_dropout = eqx.nn.Dropout(p=dropout)

        self.attn_layer = eqx.nn.MultiheadAttention(num_heads, 
                                                in_dim, 
                                                key=keys[1])
        self.attn_norm = eqx.nn.LayerNorm(in_dim)
        self.attn_dropout = eqx.nn.Dropout(p=dropout)

        ff_layer1 = eqx.nn.Linear(in_dim, 
                                    ff_dim, 
                                    use_bias=use_bias, 
                                    key=keys[2])
        self.ff_layer1 = jax.vmap(ff_layer1, in_axes=(0,))
        self.ff_activation_fn = eqx.nn.Lambda(ff_activation_fn)
        ff_layer2 = eqx.nn.Linear(ff_dim, 
                                    in_dim, 
                                    use_bias=use_bias, 
                                    key=keys[3])
        self.ff_layer2 = jax.vmap(ff_layer2, in_axes=(0,))


        self.ff_norm = eqx.nn.LayerNorm(in_dim)
        self.ff_dropout = eqx.nn.Dropout(p=dropout)
        self.output_activation_fn = eqx.nn.Lambda(output_activation_fn)

    def __call__(self, 
                target: chex.Array, 
                memory: chex.Array, 
                mask: Optional[chex.Array] = None, *, 
                key: chex.PRNGKey) -> chex.Array:
        keys = jrand.split(key, 3)
        masked_out = self.masked_attn_layer(target, target, target, mask=mask)
        masked_out = target + self.masked_attn_dropout(masked_out, key=keys[0])
        masked_out = self.masked_attn_norm(masked_out)

        out = self.attn_layer(masked_out, memory, memory)
        out = masked_out + self.attn_dropout(out, key=keys[1])
        out = self.attn_norm(out)

        ff_out = self.ff_layer1(out)
        ff_out = self.ff_activation_fn(ff_out)
        ff_out = self.ff_layer2(ff_out)
        
        ff_out = out + self.ff_dropout(ff_out, key=keys[2])
        ff_out = self.ff_norm(ff_out)
        return self.output_activation_fn(ff_out)


class Decoder(eqx.Module):
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
        self.layers = [DecoderLayer(num_heads, in_dim, ff_dim, dropout, use_bias, key=k, **kwargs) for k in keys]

    def __call__(self, 
                target: chex.Array, 
                memory: chex.Array,
                mask: Optional[chex.Array] = None, *, 
                key: chex.PRNGKey):
        keys = jrand.split(key, self.num_layers)
        for k, layer in zip(keys, self.layers):
            target = layer(target, memory, key=k, mask=mask)
        return target

