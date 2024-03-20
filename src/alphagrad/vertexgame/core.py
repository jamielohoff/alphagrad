""" 
GPU-friendly edge and vertex elimination procedures for Cross-Country Elimination 
that are totally JIT-compilable. For an in-depth discussion of Cross-Country 
Elimination and the methods described here see the book 
`Evaluating Derivatives` by Griewank et al., 2008,
https://doi.org/10.1137/1.9780898717761

DO NOT TOUCH!
"""

from functools import partial
from typing import Sequence, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from chex import Array


"""
Documentation of the sparsity types:
--------------------------------------

Every entry in the 3-dimensional tensor has the following meaning:
(sparsity type, Jacobian shape 1st input component == 1st component, 
                Jacobain shape 2nd input component == 2nd component,
                Jacobian shape 1st output component == 3rd component,
                Jacobian shape 2nd output component == 4th component)
Thus the current implementation can only deal with scalars vectors and matrices
and related operations. 
It is basically a adjecency matrix of the computational graph, with the 3rd 
dimension indicating the sparsity type and shape of the Jacobians associated
with the respective edge.

NOTE: No support for higher-order tensors yet!

Sparsity types explanation:
0: No edge between vertices
1: Dense Jacobian, i.e. no Kronecker symbols
-1: For `copy` operation that keep sparsity
8: 
-8: unused

Diagonal matrix sparsity types:
2: (1, 3)
3: (2, 4)
4: (1, 4)
5: (2, 3)
6: (1, 3) and (2, 4)
7: (1, 4) and (2, 3)

Pure Kronecker symbol sparsity types:
-2: K(1, 3)
-3: K(2, 4)
-4: K(1, 4)
-5: K(2, 3)
-6: K(1, 3) and K(2, 4)
-7: K(1, 4) and K(2, 3)

Mix between pure Kronecker and diagonal matrix sparsity meaning:
8: K(1, 3) and (2, 4)
9: K(1, 4) and (2, 3)
-8: (1, 3) and K(2, 4)
-9: (1, 4) and K(2, 3)
==> The negative sign on sparsity entry is similar to a conjugation operation

To signify replicating dimensions, we just set the value of the respective
thing to negative it's current value.
Example: (2, 3, 4, 3, -5) has a replicating dimension in 2nd output dimension
"""


# Row idx is incoming edge, col idx is outgoing edge
#  Contraction map of the indices
CONTRACTION_MAP =  jnp.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              
                              [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              
                              [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              
                              [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]]])    


# Row idx is incoming edge, col idx is outgoing edge
# Gives the resulting sparsity type if two hyperdimensional Jacobians
# are multiplied with each other
MUL_SPARSITY_MAP = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1],
                              [0, 1, 2, 1, 4, 1, 2, 4, 2],
                              [0, 1, 1, 3, 1, 5, 3, 5, 3],
                              [0, 1, 1, 4, 1, 2, 4, 2, 4],
                              [0, 1, 5, 1, 3, 1, 5, 3, 5],
                              [0, 1, 2, 3, 4, 5, 6, 7, 6],
                              [0, 1, 5, 4, 3, 2, 7, 6, 7],
                              [0, 1, 2, 3, 4, 5, 6, 7, 8]])


MUL_SPARSITY_MAP_LEFT =  jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [0, 1, 2, 1, 4, 1, 2, 4, 2],
                                    [0, 1, 1, 3, 1, 5, 3, 5, 3]])

MUL_SPARSITY_MAP_RIGHT = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [0, 1, 2, 1, 4, 1, 2, 4, 2],
                                    [0, 1, 1, 3, 1, 5, 3, 5, 3]])

MUL_SPARSITY_MAP_BOTH = jnp.array([[9, 10, 6,  7],
                                   [7, 6, 10, 12],
                                   [6, 7, 11, 12],
                                   [12, 11, 7, 6]])


# NOTE also impacts the factors of in_edges
# TODO implement this!

# Row idx is incoming edge, col idx is outgoing edge
# Gives the resulting sparsity type if two hyperdimensional Jacobians
# are added to each other
ADD_SPARSITY_MAP = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [2, 1, 2, 1, 1, 1, 2, 1, 2],
                              [3, 1, 1, 3, 1, 1, 3, 1, 3],
                              [4, 1, 1, 1, 4, 1, 1, 4, 4],
                              [5, 1, 1, 1, 1, 5, 1, 5, 5],
                              [6, 1, 2, 3, 1, 1, 6, 1, 6],
                              [7, 1, 1, 1, 4, 5, 1, 7, 7],
                              [8, 1, 2, 3, 4, 5, 6, 7, 8]])


# Add sparsity replication algorithm:
# We devised a simple solution where we just multiply the respective size of the
# dimension of the Jacobian by negative on if the dimension is replicating
# TODO implement this!


# Impact of pure Kronecker symbols on the resulting add sparsity type:
# nothing changes compared to the ADD_SPARSITY_MAP except for the cases where
# both sparsity types indicate a pure Kronecker symbol, then the resulting
# sparsity type is also a pure Kronecker symbol
# maybe use modulo operator to implement this?
# TODO implement this!


# Contraction map has to be implemented by hand?
# NOTE need separate Contraction map for pure Kronecker case since there it also
# impacts the factors of in_edges!


# We should be able to reuse the entire MUL_SPARSITY_MAP for the pure Kronecker
# case, since the resulting sparsity type is the same as for the diagonal case
# It just needs some additional if-else conditionals to capture the corner cases
# when both Jacobians have a pure Kronecker symbol as sparsity type
# TODO implement this!


Edge = Tuple[int, int]


def get_shape(edges: Array):
    num_v = edges.shape[2]
    num_i = edges.shape[1] - num_v - 1
    return num_i, num_v


def get_output_mask(edges: Array):
    return edges[2, 0, :]


def get_vertex_mask(edges: Array):
    return edges[1, 0, :]


def get_elimination_order(edges: Array):
    return edges[3, 0, :]


def make_empty_edges(info: Array) -> Array:
    """
    Creates an empty matrix fo represent the connectivity of the computational graph.
    """
    num_i = info[0]
    num_v = info[1]
    return jnp.zeros((5, num_i+num_v+1, num_v), dtype=jnp.int32)


@partial(jax.vmap, in_axes=(0, 0))
def sparsity_where(in_edge, out_edge):
    # takes care of the corner cases where there already exists an edge with a 
    # different sparsity type
    i = in_edge.astype(jnp.int32)
    j = out_edge.astype(jnp.int32)
    new_sparsity_type = ADD_SPARSITY_MAP[i, j]
    return lax.select(jnp.logical_and(i < 0, j < 0), -new_sparsity_type, new_sparsity_type)


@partial(jax.vmap, in_axes=(1, None))
def sparsity_fmas_map(in_edge, out_edge):
    """
    TODO add documentation here!
    """
    # Get the sparsity type of the ingoing and outgoing edge
    i = in_edge[0].astype(jnp.int32)
    j = out_edge[0].astype(jnp.int32)

    new_sparsity_type = MUL_SPARSITY_MAP[i, j]
    # Take care of the corner cases where we have pure Kronecker symbols
    new_sparsity_type = lax.select(jnp.logical_and(i < 0, j < 0), 
                                    -new_sparsity_type, 
                                    new_sparsity_type)
    
    contraction_map = CONTRACTION_MAP[:, i, j]
    
    # jax.debug.print("map {map}", map=contraction_map)
    masked_factors = lax.cond(jnp.sum(contraction_map) > 0,
                                lambda a: jnp.where(contraction_map > 0, a, 1),
                                lambda a: jnp.zeros(4, dtype=jnp.int32),
                                out_edge[1:])
    # jax.debug.print("{in_edge} {out_edge}", in_edge=in_edge[1:3], out_edge=masked_factors)
    fmas = jnp.prod(in_edge[1:3])*jnp.prod(masked_factors)
    return new_sparsity_type, fmas


# TODO exchange in_edges and out_edges
def get_fmas_of_jacprod(all_edges, fmas, in_edges, out_edges, nonzero, vertex, num_i):
    # Define aliases
    in_edges_primals = in_edges[3:, :]
    in_edges_outs = in_edges[1:3, :]
    
    out_edges_primals = out_edges[3:, :]
    out_edges_outs = out_edges[1:3, :]
        
    # Calculate fmas
    # Select only the edges that are connected to the vertex through code below
    new_sparsity, _fmas = sparsity_fmas_map(in_edges, out_edges[:, vertex+num_i-1])
    # jax.debug.print("{ins}", ins=in_edges)
    # jax.debug.print("{out}", out=out_edges[:, vertex+num_i-1])
    # jax.debug.print("{fmas}", fmas=_fmas)
    
    # Calculate resulting sparsity type
    new_sparsity = sparsity_where(out_edges[0, :], new_sparsity)
    new_sparsity = jnp.broadcast_to(new_sparsity, (1, *new_sparsity.shape))
    fmas = jnp.sum(_fmas)
    # In shape new edges
    new_edges_ins = jnp.where(in_edges_primals[1] > 0, in_edges_primals, out_edges_primals)
    
    # Out shape new edges
    new_edges_outs = jnp.broadcast_to(out_edges_outs[:, vertex+num_i-1, jnp.newaxis], out_edges_outs.shape)
    new_edges_outs = jnp.where(in_edges_outs[1] > 0, new_edges_outs, out_edges_outs)
    new_edges = jnp.concatenate((new_sparsity, new_edges_outs, new_edges_ins), axis=0)
    # jax.debug.print("new col {col}", col=new_edges[0, :])
        
    # Set the Jacobian adjacency matrix
    all_edges = lax.dynamic_update_index_in_dim(all_edges, new_edges, nonzero, -1)
            
    return all_edges, fmas


def vertex_eliminate(vertex: int, graph: Array) -> Tuple[Array, float]:
    """
    Fully JIT-compilable function that implements the vertex-elimination procedure. 
    Vertex elimination means that we front-eliminate all incoming edges and 
    back-eliminate all outgoing edges of a given vertex. However, the implementation
    here does not make use of the function above to be more efficient.

    Arguments:
        vertex (int): Vertex we want to eliminate.
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    # jax.debug.print("{graph}", graph=graph[0, :, :])
    num_i, num_v = get_shape(graph)
    edges = graph[:, 1:, :]
    in_edges = edges[:, :, vertex-1]
    def update_edges_fn(carry, nonzero):
        edges, fmas = carry
        # Get the index of the operation and the 
        out_edges = edges[:, :, nonzero]
        # Calculate the fma operations and the new shapes of the Jacobians for 
        # the respective and update the vertex
        edges, _fmas = lax.cond(nonzero > -1, 
                                lambda e, f, ie, oe, nz, v: get_fmas_of_jacprod(e, f, ie, oe, nz, v, num_i), 
                                lambda e, f, ie, oe, nz, v: (e, 0), 
                                edges, fmas, in_edges, out_edges, nonzero, vertex)
        fmas += _fmas        
        carry = (edges, fmas)
        return carry, None
    
    nonzeros = jnp.nonzero(edges[0, num_i+vertex-1, :], size=num_v, fill_value=-1)[0].T
        
    output, _ = lax.scan(update_edges_fn, (edges, 0), nonzeros)
    new_edges, fmas = output
    # Delete old edges
    new_edges = new_edges.at[:, num_i+vertex-1, :].set(0)
    new_edges = new_edges.at[:, :, vertex-1].set(0)

    graph = graph.at[1, 0, vertex-1].set(1)
    graph = graph.at[:, 1:, :].set(new_edges)
    # jax.debug.print("{vertex} {fmas}", vertex=vertex, fmas=fmas)
    return graph, fmas


def cross_country(order: Sequence[int], edges: Array) -> Array:
    """
    Fully JIT-compilable function that implements cross-country elimination 
    according to the given order.

    Arguments:
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    def cc_fn(carry, vertex):
        _edges, fmas = carry
        not_masked = jnp.logical_not(_edges.at[1, 0, vertex-1].get() > 0)
                
        _edges, _fmas = lax.cond(not_masked,
                                lambda e: vertex_eliminate(vertex, e),
                                lambda e: (e, 0),
                               _edges)
        fmas += _fmas
        carry = (_edges, fmas)
        return carry, _fmas
    vertices = jnp.array(order)
    output, fmas = lax.scan(cc_fn, (edges, 0), vertices)
    return output, fmas


def forward(edges: Array):
    """
    Fully JIT-compilable function that implements forward-mode AD by 
    eliminating the vertices in sequential order 1,2,3,...,n-1,n and ignores
    the ones that are given by vertex_mask, because these are typically the 
    output vertices.

    Arguments:
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_i, num_vo = get_shape(edges)
    order = jnp.arange(1, num_vo+1)
    output, _ = cross_country(order, edges)
    return output


def reverse(edges: Array):
    """
    Fully JIT-compilable function that implements reverse-mode AD by 
    eliminating the vertices in sequential order 1,2,3,...,n-1,n and ignores
    the ones that are given by vertex_mask, because these are typically the 
    output vertices.

    Arguments:
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_i, num_vo = get_shape(edges)
    order = jnp.arange(1, num_vo+1)[::-1]
    output, _ = cross_country(order, edges)
    return output

