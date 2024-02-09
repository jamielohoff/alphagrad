import jax
import jax.numpy as jnp

from graphax.examples import Helmholtz
from alphagrad.vertexgame import sparsify, densify, make_graph

edges = make_graph(Helmholtz, jnp.ones(4))


header, sparse_edges = sparsify(edges)
print(sparse_edges)

dense_edges = densify(header, sparse_edges)
print(dense_edges)

