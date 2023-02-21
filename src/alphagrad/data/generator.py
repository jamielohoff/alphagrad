from typing import Callable, Sequence, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex

from graphax.graph import GraphState
from graphax.examples.random import construct_random_graph


class VertexGameGenerator:
    """
    TODO add documentation
    """
    game_idxs: chex.Array
    info_repository: Sequence[chex.Array]
    edge_repository: Sequence[chex.Array]
    
    def __init__(self, 
                num_games: int, 
                info: chex.Array, 
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
        num_x, num_v, num_y, _, _ = info
        
        self.info_repository = []
        self.edge_repository = []
        
        keys = jrand.split(key, num_games)
        for key in keys:
            fraction = jrand.uniform(key)
            gs = construct_random_graph(num_x, num_v, num_y, key, fraction=fraction)
            self.info_repository.append(gs.info)
            self.edge_repository.append(gs.edges)

    # TODO maybe implement as iterable?
    def __call__(self, batchsize: int, key: chex.PRNGKey = None) -> GraphState:
        """Samples from the repository of possible games

        Args:
            x (_type_): _description_

        Returns:
            Any: _description_
        """
        idxs = jrand.choice(key, self.game_idxs, shape=(batchsize,))
        info = jnp.stack([self.info_repository[idx] for idx in idxs])
        edges = jnp.stack([self.edge_repository[idx] for idx in idxs])
        state = jnp.zeros((batchsize, self.info_repository[0][1]))
        return GraphState(info=info, edges=edges, state=state)

key = jrand.PRNGKey(1337)
gen = VertexGameGenerator(16, jnp.array([4, 11, 4, 20, 0]), key=key)  
print(gen(8, key).edges) 
print(gen(8, key).info)

