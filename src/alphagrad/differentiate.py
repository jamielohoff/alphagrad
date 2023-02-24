import jax
import jax.numpy as jnp

import chex
import mctx
import equinox as eqx

from graphax import VertexGameState
from graphax.vertex_game import make_vertex_game_state

def batch_vertex_game_states(games):
	batchsize = len(games)
	ts = jnp.zeros(batchsize)
	infos = jnp.stack([jnp.array(game.info) for game in games])
	edges = jnp.stack([game.edges for game in games])
	vertices = jnp.zeros((batchsize, games[0].info.num_intermediates))
	return VertexGameState(t=ts,
							info=infos,
							edges=edges,
							vertices=vertices)


@eqx.filter_jit
def differentiate(network, env_interaction_fn, key, *games):
	batchsize = len(games)
	batch_games = batch_vertex_game_states(games)
	init_carry = (batch_games, jnp.zeros(batchsize), key)
	output = env_interaction_fn(network, batchsize, init_carry)
	return output[:, -1, -2]

