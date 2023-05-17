import math
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrand
from jaxtyping import Array

from equinox import Module, static_field
from equinox.nn import Conv2d, Dropout, Embedding

from chex import PRNGKey


def dot_product_attention_weights(query: Array,
                                key: Array,
                                query_embedding: Array,
                                key_embedding: Array) -> Array:

    query = query / math.sqrt(query.shape[-1])
    logits = jnp.einsum("sd,Sd->sS", query, key)
    logits += jnp.einsum("sd,Sd->sS", query, query_embedding)
    logits += jnp.einsum("sd,Sd->sS", key, key_embedding)
    return jax.nn.softmax(logits, axis=-1)


def dot_product_attention(query: Array,
                        key_: Array,
                        value: Array,
                        query_embedding: Array,
                        key_embedding: Array,
                        value_embedding: Array,
                        key: PRNGKey,
                        dropout: Optional[Dropout] = None,
                        *,
                        inference: Optional[bool] = None) -> Array:

    weights = dot_product_attention_weights(query, key_, query_embedding, key_embedding)
    if dropout is not None:
        weights = dropout(weights, key=key, inference=inference)
    attn = jnp.einsum("sS,Sd->sd", weights, value + value_embedding)
    return attn


class AxialAttention(Module):
    r"""
    TODO docstring
    """
    query_proj: Conv2d
    key_proj: Conv2d
    value_proj: Conv2d
    output_proj: Conv2d
    dropout: Dropout
    
    query_embedding: Embedding
    key_embedding: Embedding
    value_embedding: Embedding

    axis: int = static_field()
    input_size: int = static_field()
    num_heads: int = static_field()
    query_size: int = static_field()
    key_size: int = static_field()
    value_size: int = static_field()
    output_size: int = static_field()
    qk_size: int = static_field()
    vo_size: int = static_field()
    use_query_bias: bool = static_field()
    use_key_bias: bool = static_field()
    use_value_bias: bool = static_field()
    use_output_bias: bool = static_field()

    def __init__(
        self,
        axis: int,
        input_size: int,
        num_heads: int,
        query_size: int,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        output_size: Optional[int] = None,
        qk_size: Optional[int] = None,
        vo_size: Optional[int] = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: float = 0.,
        inference: bool = False,
        *,
        key: PRNGKey,
        **kwargs) -> None:
        r"""**Arguments:**
        - `num_heads`: Number of parallel attention heads $h$.
        - `query_size`: Number of input channels for query $Q$.
        - `key_size`: Number of input channels for key $K$. Defaults to `query_size`.
        - `value_size`: Number of input channels for value $V$. Defaults to
            `query_size`.
        - `output_size`: Number of output channels. Defaults to `query_size`.
        - `qk_size`: Number of channels to compare query and key over, per head.
            Defaults to `query_size // num_heads`.
        - `vo_size`: Number of channels to compare attention-weighted value and output
            over, per head. Defaults to `query_size // num_heads`.
        - `use_query_bias`: Whether to use a bias term in the query projections.
        - `use_key_bias`: Whether to use a bias term in the key projections.
        - `use_value_bias`: Whether to use a bias term in the value projections.
        - `use_output_bias`: Whether to use a bias term in the output projection.
        - `dropout_p`: Dropout probability on attention weights.
        - `inference`: Whether to actually apply dropout at all. If `True` then dropout
            is not applied. If `False` then dropout is applied. This may be toggled
            with [`equinox.tree_inference`][] or overridden during
            [`equinox.nn.MultiheadAttention.__call__`][].
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__(**kwargs)
        qekey, kekey, vekey, qkey, kkey, vkey, okey = jrand.split(key, 7)

        if key_size is None:
            key_size = query_size
        if value_size is None:
            value_size = query_size
        if qk_size is None:
            qk_size = query_size // num_heads
        if vo_size is None:
            vo_size = query_size // num_heads
        if output_size is None:
            output_size = query_size
            
        # Maybe optimize this?
        self.query_embedding = Embedding(input_size, qk_size, key=qekey)
        self.key_embedding = Embedding(input_size, qk_size, key=kekey)
        self.value_embedding = Embedding(input_size, vo_size, key=vekey)
            
        self.query_proj = Conv2d(query_size, 
                                num_heads*qk_size, 
                                1, 
                                use_bias=use_query_bias, 
                                key=qkey)
        self.key_proj = Conv2d(key_size, 
                               num_heads*qk_size, 
                               1, 
                               use_bias=use_key_bias, 
                               key=kkey)
        self.value_proj = Conv2d(value_size, 
                                num_heads*vo_size, 
                                1, 
                                use_bias=use_value_bias, 
                                key=vkey)
        self.output_proj = Conv2d(num_heads*vo_size, 
                                output_size, 
                                1, 
                                use_bias=use_output_bias, 
                                key=okey)
        self.dropout = Dropout(dropout_p, inference=inference)
        
        self.axis = axis
        self.input_size = input_size
        self.num_heads = num_heads
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.qk_size = qk_size
        self.vo_size = vo_size
        self.use_query_bias = use_query_bias
        self.use_key_bias = use_key_bias
        self.use_value_bias = use_value_bias
        self.use_output_bias = use_output_bias

    def __call__(self,
                query: Array,
                key_: Array,
                value: Array,
                *,
                key: Optional[PRNGKey] = None,
                inference: Optional[bool] = None) -> Array:
        """**Arguments:**
        - `query`: Query embedding. Should be a JAX array of shape
            `(query_seq_length, query_size)`.
        - `key_`: Key embedding. Should be a JAX array of shape
            `(kv_seq_length, key_size)`.
        - `value`: Value embedding. Should be a JAX array of shape
            `(kv_seq_length, value_size)`.
        - `mask`: Optional mask preventing attention to certain positions. Should be a
            JAX array of shape `(num_heads, query_seq_length, kv_seq_length)`.
        - `key`: A `jax.random.PRNGKey` used for dropout. Unused if `dropout = 0`.
            (Keyword only argument.)
        - `inference`: As [`equinox.nn.Dropout.__call__`][]. (Keyword only
            argument.)
        **Returns:**
        A JAX array of shape `(query_seq_length, output_size)`.
        """
        query_heads = self._project(self.query_proj, query, self.qk_size)
        key_heads = self._project(self.key_proj, key_, self.qk_size)
        value_heads = self._project(self.value_proj, value, self.vo_size)
        
        query_embedding = self.query_embedding(jnp.arange(self.input_size)).T
        key_embedding = self.key_embedding(jnp.arange(self.input_size)).T
        value_embedding = self.value_embedding(jnp.arange(self.input_size)).T
        
        attn_fn = partial(dot_product_attention, 
                        dropout=self.dropout, 
                        inference=inference)
        
        keys = None if key is None else jrand.split(key, query_heads.shape[0])
        
        attn_fn_head = jax.vmap(attn_fn, in_axes=(self.axis, self.axis, self.axis, None, None, None, None), out_axes=self.axis)
        attn = jax.vmap(attn_fn_head, in_axes=(0, 0, 0, None, None, None, 0), out_axes=0)(
            query_heads, 
            key_heads, 
            value_heads,
            query_embedding,
            key_embedding,
            value_embedding,
            keys)
        return self.output_proj(attn.reshape(-1, *attn.shape[2:]))
    
    def _project(self, fn, arr, size) -> Array:
        heads = fn(arr)
        return heads.reshape(self.num_heads, size, *arr.shape[1:])
      
    