import functools as ft
from typing import Sequence

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex

from graphax import GraphInfo, VertexGameState, embed, make_graph_info
from graphax.vertex_game import make_vertex_game_state
from graphax.examples import (make_LIF, 
                              make_adaptive_LIF, 
                              make_Helmholtz, 
                              make_lighthouse, 
                              make_hole,
                              make_scalar_assignment_tree,
                              make_random)


def make_benchmark_games(key: chex.PRNGKey, 
                        info: GraphInfo) -> Sequence[VertexGameState]:
    keys = jrand.split(key, 7)
    names = ["lighthouse",
            "LIF",
            "adaptive LIF",
            "Helmholtz",
            "hole",
            "scalar assignment tree",
            "random",
            "random funnel"]
        
    # has to have an even amount of examples!
    edges, _info = make_lighthouse()
    edges, _ = embed(keys[0], edges, _info, info)
    lighthouse_game = make_vertex_game_state(edges, info)
    
    edges, _info = make_LIF()
    edges, _ = embed(keys[1], edges, _info, info)
    lif_game = make_vertex_game_state(edges, info)
    
    edges, _info = make_adaptive_LIF()
    edges, _ = embed(keys[2], edges, _info, info)
    adalif_game = make_vertex_game_state(edges, info)
    
    edges, _info = make_Helmholtz()
    edges, _ = embed(keys[3], edges, _info, info)
    helmholtz_game = make_vertex_game_state(edges, info)
    
#     edges, _info = make_free_energy()
#     edges, _ = embed(keys[4], edges, _info, info)
#     free_energy_game = make_vertex_game_state(edges, info)
    
    edges, _info = make_hole()
    edges, _ = embed(keys[5], edges, _info, info)
    hole_game = make_vertex_game_state(edges, info)
    
    edges, _info = make_scalar_assignment_tree()
    edges, _ = embed(keys[6], edges, _info, info)
    scalar_assignment_tree_game = make_vertex_game_state(edges, info)
    
    edges, _ = make_random(keys[7], info, fraction=.25)
    random_game = make_vertex_game_state(edges, info)
    
    _info = make_graph_info([10, 15, 1])
    edges, _ = make_random(keys[8], _info, fraction=.25)
    edges, _ = embed(keys[9], edges, _info, info)
    random_funnel_game = make_vertex_game_state(edges, info)
    
    return [lighthouse_game,
            lif_game, 
            adalif_game,
            helmholtz_game,
            hole_game,
            scalar_assignment_tree_game,
            random_game,
            random_funnel_game], names
    
    