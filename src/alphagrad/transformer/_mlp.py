from typing import Callable, Sequence, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx

class MLP(eqx.Module):
    layers: eqx.nn.Sequential
    
    def __init__(self, 
                in_size: int,
                out_size: int,
                layers: Sequence[int],
                activation: Callable = jnn.relu,
                final_activation: Callable = lambda x: x, 
                *,
                key: chex.PRNGKey) -> None:
        super().__init__()
        keys = jrand.split(key, len(layers))
        layer_list = [eqx.nn.Linear(in_size, layers[0], key=keys[0]), eqx.nn.Lambda(activation)]
        for i, key in enumerate(keys[1:]):
            layer_list.append(eqx.nn.Linear(layers[i], layers[i+1], key=key))
            layer_list.append(eqx.nn.Lambda(activation))
        layer_list.append(eqx.nn.Linear(layers[-1], out_size, key=keys[-1]))
        layer_list.append(eqx.nn.Lambda(final_activation))
        self.layers = layer_list
        
    def __call__(self, xs: chex.Array) -> chex.Array:
        for layer in self.layers:
            xs = layer(xs)
        return xs        

