import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx

from typing import Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx

from ..transformer._mlp import MLP


class ResidualBlock(eqx.Module):
    """Add docstring
    """
    layers: Sequence[eqx.Module]

    def __init__(self, *layers: eqx.Module) -> None:
        super().__init__()
        self.layers = layers

    def __call__(self, x: chex.Array, *, key: chex.PRNGKey = None):
        y = x
        for layer in self.layers:
            subkey, key = jrand.split(key, 2)
            y = layer(y, key=subkey)
        return x + y
    
    
class SqueezeExictationBlock(eqx.Module):
    pass


class AlphaZeroModel(eqx.Module):
    """
    TODO Add docstring
    """
    num_layers: int
    layers: Sequence[eqx.Module]
    value_head: eqx.nn.MLP
    policy_head: eqx.nn.MLP

    def __init__(self, 
                in_channels: int,
                num_classes: int, 
                key: chex.PRNGKey = None) -> None:
        super().__init__()
        self.layers = []
        key, conv_key, pkey, vkey = jrand.split(key, 4)

        self.layers.append(eqx.nn.Conv2d(in_channels, 64, 7, 2, key=conv_key))
        self.layers.append(eqx.nn.Lambda(jnn.silu))

        for _ in range(2):
            key1, key2, key = jrand.split(key, 3)
            block = ResidualBlock(eqx.nn.Conv2d(64, 64, 3, 1, 1, key=key1),
                                    eqx.nn.Lambda(jnn.silu),
                                    eqx.nn.Conv2d(64, 64, 3, 1, 1, key=key2),
                                    eqx.nn.Lambda(jnn.silu))
            self.layers.append(block)
            
        self.layers.append(eqx.nn.Conv2d(64, 128, 3, 2, key=key))    
        
        for _ in range(2):
            key1, key2, key = jrand.split(key, 3)
            block = ResidualBlock(eqx.nn.Conv2d(128, 128, 3, 1, 1, key=key1),
                                    eqx.nn.Lambda(jnn.silu),
                                    eqx.nn.Conv2d(128, 128, 3, 1, 1, key=key2),
                                    eqx.nn.Lambda(jnn.silu))
            self.layers.append(block)
            
        self.layers.append(eqx.nn.Conv2d(128, 256, 3, 2, key=key))   
            
        for _ in range(2):
            key1, key2, key = jrand.split(key, 3)
            block = ResidualBlock(eqx.nn.Conv2d(256, 256, 3, 1, 1, key=key1),
                                    eqx.nn.Lambda(jnn.silu),
                                    eqx.nn.Conv2d(256, 256, 3, 1, 1, key=key2),
                                    eqx.nn.Lambda(jnn.silu))
            self.layers.append(block)
            
        self.layers.append(eqx.nn.AvgPool2d(3, 2))   
            
        self.num_layers = len(self.layers)
        self.policy_head = MLP(6400, num_classes, [1024, 512], key=pkey)
        self.value_head = MLP(6400, 1, [1024, 256], key=vkey)

    def __call__(self, xs: chex.Array, key: chex.PRNGKey):
        xs = xs.astype(jnp.float32)
        keys = jrand.split(key, self.num_layers)
        for key, layer in zip(keys, self.layers):
            xs = layer(xs, key=key)

        xs = xs.flatten()
        policy = self.policy_head(xs)
        value = self.value_head(xs)
        # idx = jnp.argmax(value_dist).astype(jnp.int32)
        # value = jnp.array([-jnp.arange(0, 1000)[idx]])
        # _value = jnp.array([-jnp.arange(0, 1000)[idx-1]])
        # p = jnp.array([value_dist[idx]])
        # _p = jnp.array([value_dist[idx-1]])
        # value = (p * value + _p * _value)/(p + _p)
        return jnp.concatenate((value, policy))

