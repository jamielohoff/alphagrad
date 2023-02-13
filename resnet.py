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
    layers: Sequence[chex.Array]

    def __init__(self, *layers: chex.Array) -> None:
        super().__init__()
        self.layers = layers

    def __call__(self, x: chex.Array):
        y = x
        for layer in self.layers:
            y = layer(y)
        return x + y

class ResNet(eqx.Module):
    """Add docstring
    """
    init_conv: eqx.nn.Conv2d
    layers: Sequence[eqx.Module]
    avg_pooling: eqx.nn.AvgPool2d
    fc_layer: eqx.nn.Linear

    def __init__(self, 
                num_layers: int,
                num_classes: int = 12, 
                key: chex.PRNGKey = None) -> None:
        super().__init__()
        self.layers = []

        key, conv_key, linear_key = jrand.split(key, 3)
        keys = jrand.split(key, num_layers//2)

        self.init_conv = eqx.nn.Conv2d(1, 64, 7, key=conv_key)

        for k in keys:
            key1, key2 = jrand.split(k, 2)
            block = ResidualBlock(eqx.nn.Conv2d(64, 64, 3, 1, 2, key=key1),
                                    eqx.nn.Lambda(jnn.relu),
                                    eqx.nn.Conv2d(64, 64, 3, key=key2),
                                    eqx.nn.Lambda(jnn.relu))
            self.layers.append(block)

        self.avg_pooling = eqx.nn.AvgPool2d(9, 1)
        self.fc_layer = eqx.nn.Linear(64, num_classes, key=linear_key)

    def __call__(self, x: chex.Array):
        x = self.init_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avg_pooling(x)
        x = x.flatten()
        return self.fc_layer(x)

