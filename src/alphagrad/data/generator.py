
from typing import Sequence, Tuple
from tqdm import tqdm

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex

from graphax import GraphInfo, VertexGameState, clean, embed
from graphax.examples import make_random
from graphax.transforms import safe_preeliminations_gpu, compress_graph


def to_jnp(info: GraphInfo) -> chex.Array:
    return jnp.array([*info])


def get_masks(_info: GraphInfo, info: GraphInfo) -> Tuple[chex.Array, chex.Array]:
    d = info.num_intermediates - _info.num_intermediates
    
    # make state mask
    verts = jnp.arange(_info.num_intermediates+1, info.num_intermediates+1)[::-1]
    zeros = jnp.zeros(_info.num_intermediates)
    state_mask = jnp.concatenate((verts, zeros))
    
    # make attention mask
    in_dim = info.num_inputs + info.num_intermediates
    attn_ones = jnp.zeros((in_dim, _info.num_intermediates))
    attn_zeros = jnp.ones((in_dim, d))
    attn_mask = jnp.concatenate((attn_ones, attn_zeros))
    
    return state_mask, attn_mask


class VertexGameGenerator:
    """
    TODO add documentation
    """
    game_idxs: chex.Array
    info_repository: Sequence[GraphInfo]
    edge_repository: Sequence[chex.Array]
    state_repository: Sequence[chex.Array]
    attn_mask_repository: Sequence[chex.Array]
    
    def __init__(self, 
                num_games: int, 
                info: GraphInfo, 
                key: chex.PRNGKey = None) -> None:
        """initializes a fixed repository of possible vertex games

        Args:
            num_games (int): _description_
            info (chex.Array): _description_
            key (chex.PRNGKey, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.game_idxs = jnp.arange(num_games, dtype=jnp.int32)       
        self.info_repository = []
        self.edge_repository = []
        self.state_repository = []
        self.attn_mask_repository = []
        
        keys = jrand.split(key, num_games)
        for key in tqdm(keys):
            fraction = jrand.uniform(key, minval=.1, maxval=.5)
            edges, info = make_random(key, info, fraction=fraction)
            edges, info = clean(edges, info)
            edges, _info = safe_preeliminations_gpu(edges, info)
            edges, _info = compress_graph(edges, _info)
            
            state_mask, attn_mask = get_masks(_info, info)
            self.state_repository.append(state_mask)
            self.attn_mask_repository.append(attn_mask)
            
            edges, info = embed(edges, _info, info)
            self.info_repository.append(info)
            self.edge_repository.append(edges)
            
    def __call__(self, 
                batchsize: int, 
                num_devices: int = 1,
                key: chex.PRNGKey = None) -> VertexGameState:
        """Samples from the repository of possible games

        Args:
            x (_type_): _description_

        Returns:
            Any: _description_
        """
        idxs = jrand.choice(key, self.game_idxs, shape=(batchsize,))
        ts = jnp.zeros(batchsize)
        infos = jnp.stack([to_jnp(self.info_repository[idx]) for idx in idxs])
        edges = jnp.stack([self.edge_repository[idx] for idx in idxs])
        vertices = jnp.stack([self.state_repository[idx] for idx in idxs])
        attn_masks = jnp.stack([self.attn_mask_repository[idx] for idx in idxs])
        
        if num_devices > 1:
            per_device_batchsize = batchsize // num_devices
            ts = ts.reshape(num_devices, per_device_batchsize)
            infos = infos.reshape(num_devices, per_device_batchsize, *infos.shape[1:])
            edges = edges.reshape(num_devices, per_device_batchsize, *edges.shape[1:])
            vertices = vertices.reshape(num_devices, per_device_batchsize, *vertices.shape[1:])
            attn_masks = attn_masks.reshape(num_devices, per_device_batchsize, *attn_masks.shape[1:])
            
        return VertexGameState(t=ts,
                               info=infos,
                               edges=edges,
                               vertices=vertices), attn_masks

