import functools as ft
from typing import Sequence, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx

from .transformer.encoder import Encoder
from .transformer.axial import AxialTransformerBlock
from graphax import GraphInfo


class ConvEmbedding(eqx.Module):
    conv_layer: eqx.nn.Conv2d
    linear_layer: eqx.nn.Linear
    
    def __init__(self, 
                input_shape: Tuple[int, int, int], 
                embedding_dim: int, 
                out_channels: int, 
                kernel_size: int, 
                *,
                key: chex.PRNGKey = None, 
                **kwargs) -> None:
        super().__init__()
        
        # Defining convolutional embedding and positional encoding
        num_inputs, num_intermediates, num_outputs = input_shape
        height = num_inputs + num_intermediates
        width = num_intermediates + num_outputs
        
        lin_key, conv_key = jrand.split(key, 2)
        conv_layer = eqx.nn.Conv2d(1, out_channels, kernel_size, **kwargs, key=conv_key)
        self.conv_layer = jax.vmap(conv_layer)
        
        w = width - kernel_size + 1
        h = height - kernel_size + 1
        linear_layer= eqx.nn.Linear(w*h*out_channels, embedding_dim, key=lin_key)
        self.linear_layer = jax.vmap(linear_layer)
        
    def __call__(self, x: chex.Array, key: chex.PRNGKey = None):
        x = x[:, jnp.newaxis, :, :]
        x = self.conv_layer(x)
        x = x.reshape(x.shape[0], -1)
        return self.linear_layer(x)


class AlphaGradModel(eqx.Module):
    embedding: eqx.nn.Conv2d
    
    torso: Sequence[AxialTransformerBlock]
    final_embedding: eqx.nn.Conv2d
    
    policy_tf: Encoder
    policy_linear: eqx.nn.Linear
    
    value_head: eqx.nn.MLP
    
    num_policy_heads: int
    
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
                num_policy_heads: int = 6,
                num_policy_layers: int = 2,
                policy_ff_dim: int = 1024,
                key: chex.PRNGKey = None, 
                **kwargs) -> None:
        super().__init__()
        keys = jrand.split(key, 5)
        
        num_i = info.num_inputs
        num_v = info.num_intermediates
        num_o = info.num_outputs
                
        # Defining convolutional embedding and positional encoding
        # self.embedding = ConvEmbedding(input_shape, embedding_dim, out_channels, kernel_size, key=keys[0])
        # self.positional_encoding = eqx.nn.Embedding(num_v, embedding_dim, key=keys[1])
        self.embedding = eqx.nn.Conv2d(num_v, in_dim, kernel_size, key=keys[0])
        tensor_shape = (num_v, num_i+num_v-kernel_size+1, num_v+num_o-kernel_size+1)
                
        # Defining transformer torso 
        # self.torso = Encoder(num_layers=num_layers,
        #                     num_heads=num_heads,
        #                     in_dim=embedding_dim,
        #                     ff_dim=ff_dim, 
        #                     output_activation_fn=jnn.relu,
        #                     key=keys[2])
        torso_keys = jrand.split(key, num_layers)
        self.torso = [AxialTransformerBlock(tensor_shape, num_heads, in_dim, embedding_dim, key=key) for key in torso_keys]
        
        # Defining policy head
        self.num_policy_heads = num_policy_heads
        self.final_embedding = eqx.nn.Conv2d(in_dim, num_v, 1, key=keys[2])
        
        dim = tensor_shape[1]*tensor_shape[2]
        self.policy_tf = Encoder(num_layers=num_policy_layers,
                                num_heads=num_policy_heads,
                                in_dim=dim,
                                ff_dim=policy_ff_dim,
                                key=keys[1])
        policy_linear = eqx.nn.Linear(dim, num_v, key=keys[3])
        self.policy_linear = jax.vmap(policy_linear)
        
        # Defining value head
        value_head = eqx.nn.MLP(dim, 1, value_head_width, value_head_depth, key=keys[4])
        self.value_head = jax.vmap(value_head)
        
    def __call__(self, xs: chex.Array, key: chex.PRNGKey):          
        xs = self.embedding(xs)
        keys = jrand.split(key, len(self.torso))
        for key, layer in zip(keys, self.torso):
            xs = layer(xs, key=key, mask=None)
            
        xs = self.final_embedding(xs)
        xs = xs.reshape(xs.shape[0], -1)
        value = self.value_head(xs)*100.
        
        # Mask for auto-regressive policy head     
        m = jnp.tril(jnp.ones((xs.shape[0], xs.shape[0])))
        mask = jnp.tile(m, (self.num_policy_heads, 1, 1))
        
        policy = self.policy_tf(xs, mask=mask, key=key)
        policy = self.policy_linear(policy)
        return jnp.concatenate((value, policy), axis=1)

