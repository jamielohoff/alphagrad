import functools as ft
from typing import Sequence, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx

from .transformer.encoder import Encoder
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
    embedding: ConvEmbedding
    positional_encoding: eqx.Module
    
    torso: Encoder
    policy_tf: Encoder
    policy_linear: Encoder
    value_head: eqx.nn.MLP
    
    num_policy_heads: int
    
    def __init__(self,
                info: GraphInfo, 
                embedding_dim: int, 
                out_channels: int, 
                kernel_size: int, 
                num_layers: int,
                num_heads: int,
                ff_dim: int,
                *,
                value_head_width: int = 128,
                value_head_depth: int = 2,
                num_policy_heads: int = 6,
                num_policy_layers: int = 1,
                policy_ff_dim: int = 128,
                key: chex.PRNGKey = None, 
                **kwargs) -> None:
        super().__init__()
        keys = jrand.split(key, 6)
        
        num_i = info.num_inputs
        num_v = info.num_intermediates
        num_o = info.num_outputs
        input_shape = (num_i, num_v, num_o)
                
        # Defining convolutional embedding and positional encoding
        self.embedding = ConvEmbedding(input_shape, embedding_dim, out_channels, kernel_size, key=keys[0])
        self.positional_encoding = eqx.nn.Embedding(num_v, embedding_dim, key=keys[1])
        
        
        # Defining transformer torso
        self.torso = Encoder(num_layers=num_layers,
                            num_heads=num_heads,
                            in_dim=embedding_dim,
                            ff_dim=ff_dim, 
                            output_activation_fn=jnn.relu,
                            key=keys[2])
        
        # Defining policy head
        self.num_policy_heads = num_policy_heads
        self.policy_tf = Encoder(num_layers=num_policy_layers,
                                num_heads=num_policy_heads,
                                in_dim=embedding_dim,
                                ff_dim=policy_ff_dim,
                                key=keys[3])
        policy_linear = eqx.nn.Linear(embedding_dim, num_v, key=keys[4])
        self.policy_linear = jax.vmap(policy_linear)
        
        # Defining value head
        value_head = eqx.nn.MLP(embedding_dim, 1, value_head_width, value_head_depth, key=keys[5])
        self.value_head = jax.vmap(value_head)
        
        
    def __call__(self, x: chex.Array, key: chex.PRNGKey):
        # Mask for auto-regressive model
        m = jnp.tril(jnp.ones((x.shape[0], x.shape[0])))
        mask = jnp.tile(m, (self.num_policy_heads, 1, 1))
        
        torso_key, policy_key = jrand.split(key, 2)
        embedding = self.embedding(x)
        pos_enc = self.positional_encoding(jnp.arange(0, x.shape[0]))
        out = embedding + pos_enc
        
        out = self.torso(out, mask=None, key=torso_key)
        value = self.value_head(out)
        
        policy = self.policy_tf(out, mask=mask, key=policy_key)
        policy = self.policy_linear(policy)
        return jnp.concatenate((value, policy), axis=1)

