from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx
from equinox import static_field

from graphax import GraphInfo

from alphagrad.transformer import MLP
from alphagrad.transformer import Encoder
from alphagrad.transformer import PositionalEncoder


class SequentialTransformer(eqx.Module):
    num_heads: int
    pos_enc: PositionalEncoder
    encoder: Encoder
    policy_head: MLP
    value_head: MLP
    global_token: chex.Array
    global_token_mask_x: chex.Array = static_field()
    global_token_mask_y: chex.Array = static_field()
    
    def __init__(self, 
                in_dim: int,
                seq_len: int,
                num_layers: int,
                num_heads: int,
                ff_dim: int = 1024,
                policy_ff_dims: Sequence[int] = [512, 256],
                value_ff_dims: Sequence[int] = [256, 128],
                key: chex.PRNGKey = None) -> None:
        super().__init__()  
        self.num_heads = num_heads      
        e_key, p_key, v_key, t_key = jrand.split(key, 4)
        
        # Here we use seq_len + 1 because of the global class token
        self.pos_enc = PositionalEncoder(in_dim, seq_len+1)
        
        self.encoder = Encoder(num_layers=num_layers,
                                num_heads=num_heads,
                                in_dim=in_dim,
                                ff_dim=ff_dim,
                                key=e_key)
        
        self.global_token = jrand.normal(t_key, (in_dim, 1))
        self.global_token_mask_x = jnp.ones((seq_len, 1))
        self.global_token_mask_y = jnp.ones((1, seq_len+1))
        self.policy_head = MLP(in_dim, seq_len, policy_ff_dims, key=p_key)
        self.value_head = MLP(in_dim, 1, value_ff_dims, key=v_key)
        
    def __call__(self, 
                xs: chex.Array, 
                mask: chex.Array = None, 
                key: chex.PRNGKey = None) -> chex.Array:
        # Add global token to input
        xs = jnp.concatenate((self.global_token, xs), axis=-1)
        mask = jnp.concatenate((self.global_token_mask_x, mask), axis=-1)
        mask = jnp.concatenate((self.global_token_mask_y, mask), axis=-2)
        
        # Transpose inputs for equinox attention mechanism
        xs = self.pos_enc(xs).T
        mask = mask.T
        
        # Replicate mask and apply encoder
        replicated_mask = jnp.tile(mask[jnp.newaxis, :, :], (self.num_heads, 1, 1))
        xs = self.encoder(xs, mask=replicated_mask, key=key)
        
        # TODO change this part of the architecture?
        global_token_xs = xs[0]
        policy = self.policy_head(global_token_xs)
        value = self.value_head(global_token_xs)
        return jnp.concatenate((value, policy))


class SequentialTransformerModel(eqx.Module):
    embedding: eqx.nn.Conv1d
    transformer: SequentialTransformer
    
    def __init__(self, 
                info: GraphInfo,
                num_layers: int,
                num_heads: int,
                kernel_size: int = 7, 
                stride: int = 3, 
                key: chex.PRNGKey = None, 
                **kwargs) -> None:
        super().__init__()
        num_intermediates = info.num_intermediates
        self.embedding = eqx.nn.Conv1d(num_intermediates, num_intermediates, kernel_size, stride, key=key)
        in_dim = ((info.num_inputs+info.num_intermediates)-kernel_size)//stride+1
        self.transformer = SequentialTransformer(in_dim,
                                                info.num_intermediates, 
                                                num_layers, 
                                                num_heads, 
                                                key=key, 
                                                **kwargs)
    
    def __call__(self, 
                xs: chex.Array,
                mask: chex.Array, 
                key: chex.PRNGKey = None) -> chex.Array:
        embeddings = self.embedding(xs.T).T
        return self.transformer(embeddings, mask=mask, key=key)
    
