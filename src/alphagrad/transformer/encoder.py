from typing import Sequence, Optional

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx

Array = jax.Array
PRNGKey = jax.random.PRNGKey


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


class SwiGLU(eqx.Module):
    """
    Implementation of SwiGLU MLP as proposed in Llama paper
    """
    gate_up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear
    
    def __init__(self, 
        hidden_size: int,
        expansion: float,
        key: PRNGKey = None,
    ):
        super().__init__()
        inner_dim = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        up_key, down_key = jrand.split(key)
        self.gate_up_proj = eqx.nn.Linear(
            hidden_size, inner_dim * 2, use_bias=False, key=up_key
        )
        self.down_proj = eqx.nn.Linear(
            inner_dim, hidden_size, use_bias=False, key=down_key
        )

    def __call__(self, x: Array) -> Array:
        out = self.gate_up_proj(x)
        gate, up = jnp.split(out, 2, axis=-1)
        return self.down_proj(jnn.silu(gate) * up)
        
class EncoderLayer(eqx.Module):
    """
    Implementation of a single Encoder layer as in "Attention is all you need",
    i.e. it consists of one Multihead attention block with residual connections
    followed by a two-layer fully connected.
    """
    attn_norm: eqx.nn.LayerNorm
    attn_layer: eqx.nn.MultiheadAttention
    mlp_norm: eqx.nn.LayerNorm
    mlp: SwiGLU
    def __init__(
        self,  
        num_heads: int,
        embd_dim: int, 
        hidden_dim: int, 
        key: PRNGKey = None, 
        **kwargs
    ) -> None:
        super().__init__()
        attn_key, mlp_key = jrand.split(key)
        
        # Self-attention block
        self.attn_norm = eqx.nn.LayerNorm(embd_dim)
        self.attn_layer = eqx.nn.MultiheadAttention(
            num_heads, embd_dim, key=attn_key, **kwargs
        )

        # Feed-forward block
        self.mlp_norm = eqx.nn.LayerNorm(embd_dim)
        self.mlp = SwiGLU(embd_dim, 4, key=mlp_key)

    def __call__(self, 
        x: Array, 
        mask: Optional[Array] = None,
        *,
        key: PRNGKey
    ) -> Array:
        keys = jrand.split(key, 3)
        
        # vmap is over sequence dimension
        y = jax.vmap(self.attn_norm)(x)
        y = self.attn_layer(y, y, y, mask=mask, key=keys[0])
        x += y

        y = jax.vmap(self.mlp_norm)(x)
        y = jax.vmap(self.mlp)(y)
        return x + y

class Encoder(eqx.Module):
    """
    Stack of ´num_layers´ transformer encoder layers/cells. 
    """
    num_layers: int
    layers: Sequence[EncoderLayer]

    def __init__(
        self, 
        num_layers: int, 
        num_heads: int, 
        embd_dim: int,
        hidden_dim: int,
        key: PRNGKey, 
        **kwargs,
    ) -> None:
        super().__init__()
        keys = jrand.split(key, num_layers)

        self.num_layers = num_layers
        self.layers = [EncoderLayer(
            num_heads, embd_dim, hidden_dim, key=key, **kwargs
        ) for key in keys]

    def __call__(self, 
        xs: Array, 
        mask: Optional[Array] = None,
        *,
        key: PRNGKey
    ) -> Array:
        for i, layer in enumerate(self.layers):
            key = jrand.fold_in(key, i)
            xs = layer(xs, mask=mask, key=key)
        return xs

