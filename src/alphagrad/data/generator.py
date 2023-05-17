
from typing import Sequence
from tqdm import tqdm

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex

from graphax import GraphInfo, VertexGameState, clean
from graphax.examples import make_random


def to_jnp(info: GraphInfo) -> chex.Array:
    return jnp.array([*info])


class VertexGameGenerator:
    """
    TODO add documentation
    """
    game_idxs: chex.Array
    info_repository: Sequence[GraphInfo]
    edge_repository: Sequence[chex.Array]
    
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
        
        keys = jrand.split(key, num_games)
        for key in tqdm(keys):
            fraction = jrand.uniform(key, minval=.1, maxval=.9)
            edges, info = make_random(key, info, fraction=fraction)
            edges, info = clean(edges, info)
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
        vertices = jnp.zeros((batchsize, self.info_repository[0][1]))
        
        if num_devices > 1:
            per_device_batchsize = batchsize // num_devices
            ts = ts.reshape(num_devices, per_device_batchsize)
            infos = infos.reshape(num_devices, per_device_batchsize, *infos.shape[1:])
            edges = edges.reshape(num_devices, per_device_batchsize, *edges.shape[1:])
            vertices = vertices.reshape(num_devices, per_device_batchsize, *vertices.shape[1:])
        return VertexGameState(t=ts,
                               info=infos,
                               edges=edges,
                               vertices=vertices)

