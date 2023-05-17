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

class ResNet34(eqx.Module):
    """Add docstring
    """
    num_layers: int
    init_conv: eqx.nn.Conv2d
    layers: Sequence[eqx.Module]
    avg_pooling: eqx.nn.AvgPool2d
    value_head: eqx.nn.MLP
    policy_head: eqx.nn.MLP

    def __init__(self, 
                num_classes: int = 15, 
                key: chex.PRNGKey = None) -> None:
        super().__init__()
        self.layers = []
        key, conv_key, pkey, vkey = jrand.split(key, 4)

        self.init_conv = eqx.nn.Conv2d(1, 64, 7, 2, key=conv_key)

        for _ in range(3):
            key1, key2, key = jrand.split(key, 3)
            block = ResidualBlock(eqx.nn.Conv2d(64, 64, 3, 1, 1, key=key1),
                                    eqx.nn.Lambda(jnn.relu),
                                    eqx.nn.Conv2d(64, 64, 3, 1, 1, key=key2),
                                    eqx.nn.Lambda(jnn.relu))
            self.layers.append(block)
            
        self.layers.append(eqx.nn.Conv2d(64, 128, 1, key=key))    
        
        for _ in range(4):
            key1, key2, key = jrand.split(key, 3)
            block = ResidualBlock(eqx.nn.Conv2d(128, 128, 3, 1, 1, key=key1),
                                    eqx.nn.Lambda(jnn.relu),
                                    eqx.nn.Conv2d(128, 128, 3, 1, 1, key=key2),
                                    eqx.nn.Lambda(jnn.relu))
            self.layers.append(block)
            
        self.layers.append(eqx.nn.Conv2d(128, 256, 1, key=key))   
            
        for _ in range(6):
            key1, key2, key = jrand.split(key, 3)
            block = ResidualBlock(eqx.nn.Conv2d(256, 256, 3, 1, 1, key=key1),
                                    eqx.nn.Lambda(jnn.relu),
                                    eqx.nn.Conv2d(256, 256, 3, 1, 1, key=key2),
                                    eqx.nn.Lambda(jnn.relu))
            self.layers.append(block)
            
        self.layers.append(eqx.nn.Conv2d(256, 512, 1, key=key))   
            
        for _ in range(3):
            key1, key2, key = jrand.split(key, 3)
            block = ResidualBlock(eqx.nn.Conv2d(512, 512, 3, 1, 1, key=key1),
                                    eqx.nn.Lambda(jnn.relu),
                                    eqx.nn.Conv2d(512, 512, 3, 1, 1, key=key2),
                                    eqx.nn.Lambda(jnn.relu))
            self.layers.append(block)
            
        self.num_layers = len(self.layers)
        self.avg_pooling = eqx.nn.AvgPool2d((10, 7), 1)
        self.policy_head = eqx.nn.MLP(512, num_classes, 1024, 2, key=pkey)
        self.value_head = eqx.nn.MLP(512, 1, 256, 2, key=vkey)

    def __call__(self, xs: chex.Array, key: chex.PRNGKey):
        xs = xs[jnp.newaxis, :, :]
        xs = self.init_conv(xs)
        keys = jrand.split(key, self.num_layers)
        for k, layer in zip(keys, self.layers):
            xs = layer(xs, key=k)
        xs = self.avg_pooling(xs)
        xs = xs.flatten()
        policy = self.policy_head(xs)
        value = self.value_head(xs)
        return jnp.concatenate((value, policy))

