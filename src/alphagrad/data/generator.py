from typing import Callable, Sequence, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex

from graphax import GraphInfo, VertexGameState
from graphax.examples import construct_random


def to_jnp(info: GraphInfo) -> chex.Array:
    return jnp.array([*info])


def expand(edges: chex.Array, info: GraphInfo) -> chex.Array:
    padding = ((0, info.num_intermediates-1), (0, 0), (0, 0))
    return jnp.pad(edges[jnp.newaxis, :, :], 
                            pad_width=padding, 
                            mode="constant", 
                            constant_values=0)

from graphax.examples import construct_Helmholtz
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
        for key in keys:
            fraction = jrand.uniform(key)
            edges, info = construct_Helmholtz()# construct_random(key, info, fraction=fraction)
            self.info_repository.append(info)
            self.edge_repository.append(edges)

    # TODO maybe implement as iterable?
    def __call__(self, 
                batchsize: int, 
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
        edges = jnp.stack([expand(self.edge_repository[idx], self.info_repository[idx]) for idx in idxs])
        vertices = jnp.zeros((batchsize, self.info_repository[0][1]))
        return VertexGameState(t=ts,
                               info=infos,
                               edges=edges,
                               vertices=vertices)
