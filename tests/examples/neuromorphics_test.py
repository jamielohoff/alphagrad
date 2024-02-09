import jax
import jax.numpy as jnp

from graphax.examples.neuromorphic import ADALIF_SNN
from alphagrad.vertexgame import (minimal_markowitz, forward, reverse, 
                                cross_country, make_graph)

x = None
edges = make_graph(ADALIF_SNN, *x)
print(edges)

order = minimal_markowitz(edges)
print(forward(edges)[1])
print(reverse(edges)[1])
print(cross_country(order, edges)[1])

