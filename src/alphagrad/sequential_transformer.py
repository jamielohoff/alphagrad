from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx

from graphax import GraphInfo, make_graph_info

from alphagrad.transformer import MLP
from alphagrad.transformer import Encoder
from alphagrad.transformer import PositionalEncoder


class SequentialTransformer(eqx.Module):
    pos_enc: PositionalEncoder
    encoder: Encoder
    policy_head: MLP
    value_head: MLP
    
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
        e_key, p_key, v_key = jrand.split(key, 3)
        
        self.pos_enc = PositionalEncoder(in_dim, seq_len)
        
        self.encoder = Encoder(num_layers=num_layers,
                                num_heads=num_heads,
                                in_dim=in_dim,
                                ff_dim=ff_dim,
                                key=e_key)
        
        size = seq_len*in_dim
        self.policy_head = MLP(size, seq_len, policy_ff_dims, key=p_key)
        self.value_head = MLP(size, 1, value_ff_dims, key=v_key)
        
    def __call__(self, xs: chex.Array, key: chex.PRNGKey = None) -> chex.Array:
        xs = self.pos_enc(xs)
        xs = self.encoder(xs, key=key)
        # TODO change this part of the architecture?
        xs = xs.flatten()
        policy = self.policy_head(xs)
        value = self.value_head(xs)
        return jnp.concatenate((policy, value))


class SequentialTransformerModel(eqx.Module):
    embedding: eqx.nn.Conv1d
    transformer: SequentialTransformer
    
    def __init__(self, 
                info: GraphInfo,
                num_layers: int,
                num_heads: int,
                kernel_size: int = 5, 
                stride: int = 1, 
                key: chex.PRNGKey = None, 
                **kwargs) -> None:
        super().__init__()
        embedding = eqx.nn.Conv1d(1, 1, kernel_size, stride, key=key)
        self.embedding = jax.vmap(embedding, in_axes=(2,))
        in_dim = (info.num_inputs+info.num_intermediates)-kernel_size+1
        self.transformer = SequentialTransformer(in_dim,
                                                info.num_intermediates, 
                                                num_layers, 
                                                num_heads, 
                                                key=key, 
                                                **kwargs)
    
    def __call__(self, xs: chex.Array, key: chex.PRNGKey = None) -> chex.Array:
        embeddings = self.embedding(xs[None, :, :]).squeeze()
        return self.transformer(embeddings, key=key)

key = jrand.PRNGKey(42)
info = make_graph_info([4, 10, 4])
tf = SequentialTransformerModel(info, 1, 2, key=key)    
edges = jnp.ones((14, 10))
print(edges)
print(tf(edges, key=key).shape)
    
