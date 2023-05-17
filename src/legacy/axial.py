from typing import Callable, Optional, Tuple, Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx


def make_mask(size, num_heads): 
    mask = jnp.tril(jnp.ones((size, size)))
    return jnp.tile(mask[jnp.newaxis, :, :],(num_heads, 1, 1))


class AxialMultiheadSelfAttention(eqx.Module):
    """Computes axial multihead attention
    axis = 0 corresponds to row-attention
    axis = 1 corresponds to column-attention

    x has to have input shape (sequence length, embedding_dim)

    Args:
        axis: _description_
    """
    axis: int
    position_embedding: eqx.nn.Embedding
    attn_layer: eqx.nn.MultiheadAttention
        
    def __init__(self, 
                axis: int, 
                tensor_shape: Sequence[int], 
                num_heads: int, 
                query_size: int, 
                *args, 
                key: chex.PRNGKey = None, 
                **kwargs) -> None:
        super().__init__()
        assert axis != -1
        embed_key, key = jrand.split(key, 2)
        
        self.axis = axis
        self.position_embedding = eqx.nn.Embedding(tensor_shape[axis], 3*query_size, key=embed_key)
        
        attn_layer = eqx.nn.MultiheadAttention(num_heads, query_size, *args, key=key, **kwargs)
        attn_fn = lambda q, k, v, mask, key: attn_layer(q, k, v, mask=mask, key=key)
        self.attn_layer = jax.vmap(attn_fn, in_axes=(axis, axis, axis, None, None))
        
    def __call__(self, 
                xs: chex.Array, 
                mask: chex.Array = None, 
                key: chex.PRNGKey = None):
        t = xs.shape[1-self.axis]
        pos_embedding = jnp.expand_dims(self.position_embedding(jnp.arange(t)), axis=self.axis)
        q_pos, k_pos, v_pos = pos_embedding.split(3, axis=-1)
        qs = xs + q_pos
        ks = xs + k_pos
        vs = xs + v_pos
        return self.attn_layer(qs, ks, vs, mask, key).reshape(xs.shape)


class AxialTransformerBlock(eqx.Module):
    """
    write something about 1d convolutions
    Inspired by axial deep lab
    """
    conv_layer1: eqx.nn.Conv2d
    conv_activation_fn1: eqx.nn.Lambda = eqx.static_field()
    conv_dropout1: eqx.nn.Dropout
    
    attn_norm: eqx.nn.LayerNorm
    row_attn_layer: AxialMultiheadSelfAttention
    col_attn_layer: AxialMultiheadSelfAttention

    conv_layer2: eqx.nn.Conv2d
    conv_activation_fn2: eqx.nn.Lambda = eqx.static_field()
    conv_dropout2: eqx.nn.Dropout

    output_activation_fn: eqx.nn.Lambda = eqx.static_field()

    def __init__(self,  
                tensor_shape: Sequence[int],
                num_heads: int,
                in_dim: int, 
                embedding_dim: int,  
                dropout: float = .1,
                use_bias: bool = False,
                conv_activation_fn: Callable = jnn.relu,
                output_activation_fn: Callable = lambda x: x, *,
                key: chex.PRNGKey = None, 
                **kwargs) -> None:
        super().__init__()
        keys = jrand.split(key, 4)
        # Conv block 1
        self.conv_layer1 = eqx.nn.Conv2d(in_dim, 
                                        embedding_dim, 
                                        kernel_size=1,
                                        use_bias=use_bias, 
                                        key=keys[0])
        self.conv_activation_fn1 = eqx.nn.Lambda(conv_activation_fn)
        self.conv_dropout1 = eqx.nn.Dropout(p=dropout)
        
        # Self-attention block
        self.attn_norm = eqx.nn.LayerNorm(embedding_dim)
        self.row_attn_layer = AxialMultiheadSelfAttention(0, 
                                                        tensor_shape, 
                                                        num_heads, 
                                                        embedding_dim, 
                                                        key=keys[1],
                                                        **kwargs)
        self.col_attn_layer = AxialMultiheadSelfAttention(1,
                                                        tensor_shape,
                                                        num_heads, 
                                                        embedding_dim, 
                                                        key=keys[2],
                                                        **kwargs)
        
        # Conv block 2
        self.conv_layer2 = eqx.nn.Conv2d(embedding_dim, 
                                        in_dim, 
                                        kernel_size=1,
                                        use_bias=use_bias, 
                                        key=keys[3])
        self.conv_activation_fn2 = eqx.nn.Lambda(conv_activation_fn)
        self.conv_dropout2 = eqx.nn.Dropout(p=dropout)

        self.output_activation_fn = eqx.nn.Lambda(output_activation_fn)

    def __call__(self, 
                xs: chex.Array, 
                mask: Optional[chex.Array] = None, *, 
                key: chex.PRNGKey = None) -> chex.Array:
        keys = jrand.split(key, 4)
        
        out = self.conv_layer1(xs)
        out = self.conv_activation_fn1(out)
        out = self.conv_dropout1(out, key=keys[0])
        
        out = jnp.moveaxis(out, 0, -1)
        out = self.attn_norm(out)
        out = self.row_attn_layer(out, key=keys[1])
        out = self.col_attn_layer(out, key=keys[2])
        out = jnp.moveaxis(out, -1, 0)
        
        out = self.conv_layer2(out)
        out = self.conv_activation_fn2(out)
        out = self.conv_dropout2(out, key=keys[3])

        out += xs
        return self.output_activation_fn(out)

