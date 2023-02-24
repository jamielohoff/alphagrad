import functools as ft
from typing import Sequence, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx

from .transformer.axial import AxialTransformerBlock
from graphax import GraphInfo


class AlphaGradModel(eqx.Module):
    embedding: eqx.nn.Conv2d
    
    torso: Sequence[AxialTransformerBlock]
    final_embedding: eqx.nn.Conv2d
    
    policy_head: eqx.nn.MLP
    value_head: eqx.nn.MLP
    
    def __init__(self,
                info: GraphInfo, 
                in_dim: int,
                embedding_dim: int, 
                kernel_size: int, 
                num_layers: int,
                num_heads: int,
                *,
                value_head_width: int = 256,
                value_head_depth: int = 2,
                num_policy_layers: int = 2,
                policy_ff_dim: int = 1024,
                key: chex.PRNGKey = None, 
                **kwargs) -> None:
        super().__init__()
        keys = jrand.split(key, 5)
        
        num_i = info.num_inputs
        num_v = info.num_intermediates
        num_o = info.num_outputs
                
        # Defining convolutional embedding
        self.embedding = eqx.nn.Conv2d(1, in_dim, kernel_size, key=keys[0])
        tensor_shape = (1, num_i+num_v-kernel_size+1, num_v+num_o-kernel_size+1)
                
        # Defining transformer torso 
        torso_keys = jrand.split(key, num_layers)
        self.torso = [AxialTransformerBlock(tensor_shape, num_heads, in_dim, embedding_dim, key=key) for key in torso_keys]
        self.final_embedding = eqx.nn.Conv2d(in_dim, 32, 1, key=keys[2])
        
        dim = tensor_shape[1]*tensor_shape[2]*32

        # Defining policy head
        self.policy_head = eqx.nn.MLP(dim, num_v, policy_ff_dim, num_policy_layers, key=keys[4])
        
        # Defining value head
        self.value_head = eqx.nn.MLP(dim, 1, value_head_width, value_head_depth, key=keys[4])
        
    def __call__(self, xs: chex.Array, key: chex.PRNGKey) -> chex.Array:  
        xs = self.embedding(xs[jnp.newaxis, :, :])

        keys = jrand.split(key, len(self.torso))
        for key, layer in zip(keys, self.torso):
            xs = layer(xs, key=key)
            
        xs = self.final_embedding(xs)
        xs = xs.flatten()
        value = self.value_head(xs)
        policy = self.policy_head(xs)
        return jnp.concatenate((value, policy))

