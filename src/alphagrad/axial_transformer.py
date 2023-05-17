import functools as ft
from typing import Sequence, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx

from graphax import GraphInfo
from .transformer._axial import AxialAttention


class AxialAttentionBlock(eqx.Module):
    height_attn: AxialAttention
    width_attn: AxialAttention
    
    def __init__(self,
                in_dim: int, 
                height: int,
                width: int,
                num_heads: int,
                *,
                dropout_p: float = .2,
                embedding_dim: int = None,
                key: chex.PRNGKey = None) -> None:
        if embedding_dim is None:
            embedding_dim = in_dim
            
        self.width_attn = AxialAttention(1, 
                                        width, 
                                        num_heads, 
                                        query_size=in_dim,
                                        dropout_p=dropout_p, 
                                        key=key)
        self.height_attn = AxialAttention(2, 
                                         height, 
                                        num_heads, 
                                        query_size=in_dim,
                                        dropout_p=dropout_p,
                                        key=key)
        
    def __call__(self, xs, key):
        out = self.height_attn(xs, xs, xs, key=key)
        out += self.width_attn(xs, xs, xs, key=key)
        return xs + out


class AxialTransformerModel(eqx.Module):
    input_block: eqx.nn.Conv2d
    
    torso: Sequence[AxialAttentionBlock]
    final_block: eqx.nn.AvgPool2d
    
    policy_head: eqx.nn.MLP
    value_head: eqx.nn.MLP
    
    def __init__(self,
                info: GraphInfo, 
                embedding_dim: int, 
                num_layers: int,
                num_heads: int,
                *,
                value_head_width: int = 512,
                value_head_depth: int = 2,
                num_policy_layers: int = 2,
                policy_ff_dim: int = 1024,
                key: chex.PRNGKey = None) -> None:
        super().__init__()
        keys = jrand.split(key, 5)
        
        num_i = info.num_inputs
        num_v = info.num_intermediates
        num_o = info.num_outputs
        tensor_shape = (num_i+num_v, num_v+num_o)
                
        # Defining input block
        self.input_block = eqx.nn.Conv2d(1, embedding_dim, 1, key=keys[0], use_bias=False)

                
        # Defining transformer torso 
        torso_keys = jrand.split(key, num_layers)
        self.torso = [AxialAttentionBlock(embedding_dim, *tensor_shape, num_heads, key=key) for key in torso_keys]
        
        # Not sure about this final block! Is it destroying the transformer representation?
        self.final_block = eqx.nn.AvgPool2d(tensor_shape)

        # Defining policy head
        self.policy_head = eqx.nn.MLP(embedding_dim, num_v, policy_ff_dim, num_policy_layers, key=keys[4])
        
        # Defining value head
        self.value_head = eqx.nn.MLP(embedding_dim, 1, value_head_width, value_head_depth, key=keys[4])
        
    def __call__(self, xs: chex.Array, key: chex.PRNGKey) -> chex.Array:  
        xs = self.input_block(xs[jnp.newaxis, :, :], key=key)
        keys = jrand.split(key, len(self.torso))
        for key, layer in zip(keys, self.torso):
            xs = layer(xs, key=key)
        xs = self.final_block(xs)
        xs = xs.flatten()
        value = self.value_head(xs)
        policy = self.policy_head(xs)
        return jnp.concatenate((value, policy))

