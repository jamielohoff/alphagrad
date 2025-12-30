from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx

from alphagrad.transformer import Encoder, MLP, PositionalEncoder

Array = jax.Array
PRNGKey = jax.random.PRNGKey

class GraphEmbedding(eqx.Module):
    embedding: eqx.nn.Conv2d
    projection: Array
    # output_token: Array
    
    def __init__(self, 
        graph_shape: Sequence[int],
        embd_dim: int,
        key: PRNGKey = None,
    ) -> None:
        super().__init__()
        num_i, num_vo, num_o = graph_shape
        embed_key, token_key, proj_key = jrand.split(key, 3)
        kernel_size, stride = 1, 1
        self.embedding = eqx.nn.Conv2d(
            num_vo, num_vo, (5, kernel_size), stride=(1, stride), key=embed_key
        )
        conv_size = (num_i+num_vo-kernel_size) // stride+1
        self.projection = jrand.normal(proj_key, (conv_size, embd_dim))
        # self.output_token = jrand.normal(token_key, (num_i+num_vo, 1))
    
    def __call__(self, graph: Array, key: PRNGKey = None) -> Array:
        output_mask = graph.at[2, 0, :].get()
        vertex_mask = graph.at[1, 0, :].get() - output_mask
        attn_mask = jnp.logical_or(vertex_mask.reshape(1, -1), vertex_mask.reshape(-1, 1))
        
        # output_token_mask = jnp.where(graph.at[2, 0, :].get() > 0, self.output_token, 0.)
        edges = graph.at[:, 1:, :].get() #  + output_token_mask[jnp.newaxis, :, :]
        edges = edges.astype(jnp.float32)
        
        embeddings = self.embedding(edges.transpose(2, 0, 1)).squeeze()
        embeddings = jax.vmap(jnp.matmul, in_axes=(0, None))(embeddings, self.projection)
        return embeddings.T, attn_mask.T


class SequentialTransformer(eqx.Module):
    num_heads: int
    pos_enc: PositionalEncoder
    encoder: Encoder
    policy_enc: Encoder
    policy_head: MLP
    value_head: MLP
    # global_token: Array
    # global_token_mask_x: Array = static_field()
    # global_token_mask_y: Array = static_field()
    
    def __init__(
        self, 
        embd_dim: int,
        seq_len: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int = 1024,
        num_policy_layers: int = 2,
        policy_hidden_dims: Sequence[int] = [512, 256],
        value_hidden_dims: Sequence[int] = [1024, 512],
        key: PRNGKey = None
    ) -> None:
        super().__init__()  
        self.num_heads = num_heads      
        e_key, p_key, pe_key, v_key, t_key = jrand.split(key, 5)
        
        # Here we use seq_len + 1 because of the global class token
        self.pos_enc = PositionalEncoder(embd_dim, seq_len)
        
        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            embd_dim=embd_dim,
            hidden_dim=hidden_dim,
            key=e_key
        )
        
        self.policy_enc = Encoder(
            num_layers=num_policy_layers,
            num_heads=num_heads,
            embd_dim=embd_dim,
            hidden_dim=hidden_dim,
            key=pe_key
        )
        
        # self.global_token = jrand.normal(t_key, (embd_dim, 1))
        # self.global_token_mask_x = jnp.ones((seq_len, 1))
        # self.global_token_mask_y = jnp.ones((1, seq_len+1))
        self.policy_head = MLP(embd_dim, 1, policy_hidden_dims, key=p_key)
        self.value_head = MLP(embd_dim, 1, value_hidden_dims, key=v_key)
        
        
    def __call__(self, xs: Array, mask: Array = None, key: PRNGKey = None) -> Array:
        e_key, p_key = jrand.split(key, 2)
            
        # Add global token to input
        # xs = jnp.concatenate((self.global_token, xs), axis=-1)
        # mask = jnp.concatenate((self.global_token_mask_x, mask), axis=-1)
        # mask = jnp.concatenate((self.global_token_mask_y, mask), axis=-2)
        
        # Transpose inputs for equinox attention mechanism
        xs = self.pos_enc(xs).T

        # Replicate mask and apply encoder
        if mask is not None:
            mask = mask.T #  TODO fix this weird thing here
            mask = jnp.tile(mask[jnp.newaxis, :, :], (self.num_heads, 1, 1))
            xs = self.encoder(xs, mask=mask, key=e_key)
        else: 
            xs = self.encoder(xs, mask=None, key=e_key)
        # global_token_xs = xs[0]
        values = jax.vmap(self.value_head)(xs)
        
        # similar to Global Average Pooling in ViTs
        value = jnp.mean(values)
        
        policy = jax.vmap(self.policy_head)(xs)
        return jnp.concatenate((jnp.array([value]), policy.squeeze()))


class PPOModel(eqx.Module):
    embedding: eqx.nn.Conv2d
    projection: Array
    # output_token: Array
    transformer: SequentialTransformer
    
    def __init__(
        self, 
        graph_shape: Sequence[int],
        embd_dim: int,
        num_layers: int,
        num_heads: int,
        key: PRNGKey = None,
        **kwargs
    ) -> None:
        super().__init__()
        num_i, num_vo, num_o = graph_shape
        embed_key, token_key, proj_key, tf_key = jrand.split(key, 4)
        self.embedding = eqx.nn.Conv2d(num_vo, num_vo, (5, 1), key=embed_key)
        self.projection = eqx.nn.Linear(num_i+num_vo, embd_dim, key=proj_key)
        # self.output_token = jrand.normal(token_key, (num_i+num_vo, 1))
        self.transformer = SequentialTransformer(
            embd_dim, num_vo, num_layers, num_heads, key=tf_key, **kwargs
        )
    
    def __call__(self, xs: Array, key: PRNGKey = None) -> Array:
        output_mask = xs.at[2, 0, :].get()
        vertex_mask = xs.at[1, 0, :].get() - output_mask
        attn_mask = jnp.logical_or(vertex_mask.reshape(1, -1), vertex_mask.reshape(-1, 1))
        
        # output_token_mask = jnp.where(xs.at[2, 0, :].get() > 0, self.output_token, 0.)
        edges = xs.at[:, 1:, :].get() # + output_token_mask[jnp.newaxis, :, :]
        edges = edges.astype(jnp.float32)
        
        embeddings = self.embedding(edges.transpose(2, 0, 1)).squeeze()
        # embeddings = jax.vmap(jnp.matmul, in_axes=(0, None))(embeddings, self.projection)
        embeddings = jax.vmap(self.projection, in_axes=0)(embeddings)
        return self.transformer(embeddings.T, mask=attn_mask, key=key)
    
    
class AlphaZeroModel(eqx.Module):
    embedding: eqx.nn.Conv2d
    projection: Array
    # output_token: Array
    transformer: SequentialTransformer
    
    def __init__(
        self, 
        graph_shape: Sequence[int],
        embd_dim: int,
        num_layers: int,
        num_heads: int,
        key: PRNGKey = None,
        **kwargs
    ) -> None:
        super().__init__()
        num_i, num_vo, num_o = graph_shape
        embed_key, token_key, proj_key, tf_key = jrand.split(key, 4)
        kernel_size, stride = 3, 2
        self.embedding = eqx.nn.Conv2d(
            num_vo, num_vo, (5, kernel_size), stride=(1, stride), key=embed_key
        )
        conv_size = (num_i+num_vo-kernel_size) // stride+1
        self.projection = jrand.normal(proj_key, (conv_size, embd_dim))
        # self.output_token = jrand.normal(token_key, (num_i+num_vo, 1))
        self.transformer = SequentialTransformer(
            embd_dim, num_vo, num_layers, num_heads, key=tf_key, **kwargs
        )
    
    def __call__(self, xs: Array, key: PRNGKey = None) -> Array:
        output_mask = xs.at[2, 0, :].get()
        vertex_mask = xs.at[1, 0, :].get() - output_mask
        attn_mask = jnp.logical_or(vertex_mask.reshape(1, -1), vertex_mask.reshape(-1, 1))
        
        # output_token_mask = jnp.where(xs.at[2, 0, :].get() > 0, self.output_token, 0.)
        edges = xs.at[:, 1:, :].get() #  + output_token_mask[jnp.newaxis, :, :]
        edges = edges.astype(jnp.float32)
        
        embeddings = self.embedding(edges.transpose(2, 0, 1)).squeeze()
        embeddings = jax.vmap(jnp.matmul, in_axes=(0, None))(embeddings, self.projection)
        return self.transformer(embeddings.T, mask=attn_mask, key=key)
    

class PolicyNet(eqx.Module):
    num_heads: int
    embedding: GraphEmbedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    head: MLP
    
    def __init__(
        self, 
        graph_shape: Sequence[int],
        embd_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int = 1024,
        mlp_dims: Sequence[int] = [512, 256],
        key: PRNGKey = None
    ) -> None:
        super().__init__()     
        num_i, num_vo, num_o = graph_shape
        self.num_heads = num_heads
        encoder_key, embed_key, key = jrand.split(key, 3)
        self.embedding = GraphEmbedding(graph_shape, embd_dim, key=embed_key)
        
        # Here we use seq_len + 1 because of the global class token
        self.pos_enc = PositionalEncoder(embd_dim, num_vo)
        
        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            embd_dim=embd_dim,
            hidden_dim=hidden_dim,
            key=encoder_key
        )

        self.head = MLP(embd_dim, 1, mlp_dims, key=key)
        
    def __call__(self, graph: Array, key: PRNGKey = None) -> Array:  
        # Embed the input graph
        embeddings, mask = self.embedding(graph)
        
        # Transpose inputs for equinox attention mechanism
        embeddings = self.pos_enc(embeddings).T
        
        # Replicate mask and apply encoder
        mask = jnp.tile(mask[jnp.newaxis, :, :], (self.num_heads, 1, 1))
        xs = self.encoder(embeddings, mask=mask, key=key)
        
        policy = jax.vmap(self.head)(xs)
        return policy.squeeze()


class ValueNet(eqx.Module):
    num_heads: int
    embedding: GraphEmbedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    head: MLP
    # global_token: Array
    # global_token_mask_x: Array = static_field()
    # global_token_mask_y: Array = static_field()
    
    def __init__(
        self, 
        graph_shape: Sequence[int],
        embd_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int = 1024,
        mlp_dims: Sequence[int] = [1024, 512],
        key: PRNGKey = None
    ) -> None:
        super().__init__()    
        num_i, num_vo, num_o = graph_shape  
        self.num_heads = num_heads 
        embedding_key, encoder_key, token_key, key = jrand.split(key, 4)
        self.embedding = GraphEmbedding(graph_shape, embd_dim, key=embedding_key)
        
        # Here we use seq_len + 1 because of the global class token
        self.pos_enc = PositionalEncoder(embd_dim, num_vo)
        
        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            embd_dim=embd_dim,
            hidden_dim=hidden_dim,
            key=encoder_key
        )
        
        # self.global_token = jrand.normal(token_key, (embd_dim, 1))
        # self.global_token_mask_x = jnp.ones((num_vo, 1))
        # self.global_token_mask_y = jnp.ones((1, num_vo+1))
        self.head = MLP(embd_dim, 1, mlp_dims, key=key)
        
    def __call__(self, graph: Array, key: PRNGKey = None) -> Array:
        # Embed the input graph
        embeddings, mask = self.embedding(graph)
        
        # Add global token to input
        # embeddings = jnp.concatenate((self.global_token, embeddings), axis=-1)
        # mask = jnp.concatenate((self.global_token_mask_x, mask), axis=-1)
        # mask = jnp.concatenate((self.global_token_mask_y, mask), axis=-2)
        
        # Transpose inputs for equinox attention mechanism
        embeddings = self.pos_enc(embeddings).T
        
        # Replicate mask and apply encoder
        mask = jnp.tile(mask[jnp.newaxis, :, :], (self.num_heads, 1, 1))
        xs = self.encoder(embeddings, mask=mask, key=key)
        values = jax.vmap(self.head)(xs)
        
        return jnp.mean(values)

