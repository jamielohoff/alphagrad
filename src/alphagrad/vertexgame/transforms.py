"""File that contains various types of transformations that leave the 
extended adjacency matrix invariant under cross country elimination.

Most of the transformations either extend or reduce the repesentation size.
"""

from functools import partial
from typing import Sequence

import jax
import jax.lax as lax
import jax.numpy as jnp

from .core import vertex_eliminate, get_shape

Array = jax.Array
PRNGKey = jax.random.PRNGKey


def connectivity_checker(edges: Array) -> Array:    
    """
    Function that checks if graph is properly connected.
    
    Args:
        edges (Array): Extended adjacency matrix.

    Returns:
        bool: Whether all nodes in a graph are properly connected.
    """   
    num_i, num_vo = get_shape(edges)
    in_sum = jnp.sum(edges[0, 1:, :], axis=1)
    out_sum = jnp.sum(edges[0, 1:, :], axis=0)
    ins_connected = jnp.not_equal(in_sum, 0)[num_i:]
    outs_connected = jnp.not_equal(out_sum, 0)
    output_mask = edges.at[2, 0, :].get()
    is_connected = jnp.logical_xor(ins_connected, outs_connected)
    return jnp.logical_or(jnp.logical_not(is_connected), output_mask)


def compress(edges: Array) -> Array:
    """
    Function that removes all zero rows and cols from the computational graph
    representation, i.e. the extended adjacency matrix.
    
    NOTE: This changes the shape of the edges array and the number of 
    intermediate variables and is thus not jittable!
    
    Args:
        edges (Array): Extended adjacency matrix.

    Returns:
        Sequence[int]: Compressed adjacency matrix.
    """
    num_i, num_v, num_o = edges.at[0, 0, 0:3].get()
            
    i, num_removed_vertices = 1, 0
    for _ in range(1, num_v+1):            
        s1 = jnp.sum(edges.at[:, i+num_i, :].get()) == 0
        s2 = jnp.sum(edges.at[:, 1:, i-1].get()) == 0
        if s1 and s2:         
            edges = jnp.delete(edges, i+num_i, axis=1)
            edges = jnp.delete(edges, i-1, axis=2)
            num_removed_vertices += 1
        else:
            i += 1

    num_v = edges.shape[2]
    num_i = edges.shape[1] - num_v - 1
    num_o = jnp.sum(edges.at[2, 0, :].get())
    shape = jnp.array([num_i, num_v-num_o, num_o])
    edges = edges.at[0, 0, 0:3].set(shape)
    return edges


def clean(edges: Array) -> Array:
    """
    Removes all unconnected interior nodes.
    
    Args:
        edges (Array): Extended adjacency matrix.

    Returns:
        Sequence[int]: Extended adjacency matrix with new shape.
    """
    num_i, num_vo = get_shape(edges)
    row_shape = num_i+num_vo
        
    conn = connectivity_checker(edges)
    is_clean = jnp.all(conn)
    while not is_clean:
        idxs = jnp.nonzero(jnp.logical_not(conn))[0]
        def clean_edges_fn(_edges, idx):
            _edges = _edges.at[:, num_i + idx + 1, :].set(jnp.zeros((5, num_vo)))
            _edges = _edges.at[:, 1:, idx].set(jnp.zeros((5, row_shape)))
            return _edges, None
        edges, _ = lax.scan(clean_edges_fn, edges, idxs)
        conn = connectivity_checker(edges)
        is_clean = jnp.all(conn)

    return edges


def embed(edges: Array, new_shape: Sequence[int]) -> Array:
    """
    Embeds a smaller graph into a larger graph frame by appending empty rows.
    
    NOTE: Changes size of the tensor to new shape and thus not jittable!
    
    Args:
        edges (Array): Extended adjacency matrix.
        new_shape(Sequence[int]): New shape of the extended adjacency matrix. 
        Has to be a sequence of length 3.

    Returns:
        Sequence[int]: Extended adjacency matrix with new shape.
    """
    assert len(new_shape) == 3, "`new_shape` has to have dimension 3!"
    num_i, num_vo = get_shape(edges)
    new_num_i, new_num_vo, new_num_o = new_shape
    
    i_diff = new_num_i - num_i
    vo_diff = new_num_vo - num_vo

    if i_diff == 0 and vo_diff == 0:
        return edges
    elif i_diff < 0 or vo_diff < 0:
        raise ValueError(f"Graph of shape {edges.shape} to large to be embedded in shape{new_shape}!")
    
    le, re = jnp.split(edges, (num_i+1,), axis=1)
    edges = jnp.concatenate([le, jnp.zeros((5, i_diff, num_vo), dtype=jnp.int32), re], axis=1)
        
    edges = jnp.pad(edges, ((0, 0), (0, vo_diff), (0, vo_diff)), mode="constant", constant_values=0)
    edges = edges.at[1, 0, num_vo:].set(1)
        
    # Update edge state size to new size
    edges = edges.at[0, 0, :].set(0)
    edges = edges.at[0, 0, 0:3].set(jnp.array([new_num_i, new_num_vo-new_num_o, new_num_o]))
    return edges


def safe_preeliminations(edges: Array, return_order: bool = False) -> Array:
    """
    Eliminates only vertices with minimal Markowitz degree 0 and 1.
    
    Args:
        edges (Array): Extended adjacency matrix.

    Returns:
        Sequence[int]: Extended adjacency matrix with vertices with Markowitz
        degree 0 or 1 removed.
    """
    num_i, num_vo = get_shape(edges)
    
    def loop_fn(_edges, _):
        degree, mMd_vertex = get_minimal_markowitz(_edges)
        _edges, vertex = lax.cond(
            degree < 2,
            lambda e: (vertex_eliminate(mMd_vertex, e)[0], mMd_vertex),
            lambda e: (e, -1), 
            _edges
        )
        return _edges, vertex
    
    it = jnp.arange(1, num_vo+1)
    edges, preelim_order = lax.scan(loop_fn, edges, it)
    
    if return_order:
        return edges, [int(p) for p in preelim_order if p > 0]
    return edges


### Tools for minimal Markowitz
@partial(jax.jit, static_argnums=1)
def minimal_markowitz(edges: Array, num_v: int) -> Sequence[int]:    
    """Function that executes the minimal Markowitz elemination order.

    Args:
        edges (Array): Extended adjacency matrix.
        num_v (int): Number of intermediate vertices.

    Returns:
        Sequence[int]: Elimination order derived from minimal Markowitz degree.
    """
    def loop_fn(_edges, _):
        degree, mMd_vertex = get_minimal_markowitz(_edges)
        _edges, _ = vertex_eliminate(mMd_vertex, _edges)
        return _edges, mMd_vertex

    it = jnp.arange(1, num_v+1)
    _, idxs = lax.scan(loop_fn, edges, it)

    return [i.astype(jnp.int32) for i in idxs]


def get_minimal_markowitz(edges: Array, degrees: bool = False) -> Sequence[int]:
    """
    Function that calculates the elimination order of a computational graph
    with regard to the minimal Markowitz degree.
    
    Args:
        edges (Array): Extended adjacency matrix.
        degrees (bool): Whether to print the Markowitz degrees of all nodes.

    Returns:
        Sequence[int]: Elimination order derived from minimal Markowitz degree.
    """    
    def loop_fn(carry, vertex):
        is_eliminated_vertex = edges.at[1, 0, vertex-1].get() == 1
        markowitz_degree = lax.cond(
            is_eliminated_vertex,
            lambda v, e: -1, 
            lambda v, e: compute_markowitz_degree(v, e), 
            vertex, edges
        )
        return carry, markowitz_degree
    
    vertices = jnp.arange(1, edges.shape[-1]+1)
    _, markowitz_degrees = lax.scan(loop_fn, (), vertices)
    if degrees:
        print(f"Markowitz degrees: {markowitz_degrees}")
    idx = jnp.sum(edges.at[1, 0, :].get())
    mMd_vertex = jnp.argsort(markowitz_degrees)[idx]+1
    return markowitz_degrees[mMd_vertex-1], mMd_vertex


def compute_markowitz_degree(vertex: int, edges: Array) -> int:
    """Function that computes the Markowitz degree of a given vertex in the 
    computational graph.

    Args:
        vertex (int): The vertex in question.
        edges (Array): Extended ajdacency matrix representing the computational
        graph.

    Returns:
        int: Markowitz degree of said vertex.
    """
    num_i, num_vo = get_shape(edges)
    in_edge_slice = edges.at[:, vertex+num_i, :].get()
    out_edge_slice = edges.at[:, 1:, vertex-1].get()      

    in_edge_count = count_in_edges(in_edge_slice)
    out_edge_count = count_out_edges(out_edge_slice)
    return in_edge_count * out_edge_count


def count_in_edges(edge_slice: Array) -> int:
    """Counts the number of input edges of a specific vertex. This is achieved
    by extacting a horizontal slice from the extended adjacency matrix at the
    position of the vertex.
    
    Args:
        edge_slice (Array): Input slice of the extended ajdacency matrix 
        representing for a specific vertex.

    Returns:
        int: Number of input edges.
    """
    def loop_fn(num_edges, slice):
        sparsity_type = slice.at[0].get()
        _num_edges = lax.cond(
            sparsity_type == 1,
            lambda s: jnp.prod(s.at[3:].get()),
            lambda s: matrix_parallel(s),
            slice
        )
        num_edges += _num_edges
        return num_edges, None

    num_edges, _ = lax.scan(loop_fn, 0, edge_slice.T)
    
    return num_edges


def count_out_edges(edge_slice: Array) -> int:
    """Counts the number of output edges of a specific vertex. This is achieved 
    by extacting a vertical slice from the extended adjacency matrix at the
    position of the vertex.
    
    Args:
        edge_slice (Array): Output slice of the extended ajdacency matrix 
        representing for a specific vertex.

    Returns:
        int: Number of output edges.
    """
    def loop_fn(num_edges, slice):
        sparsity_type = slice.at[0].get()
        _num_edges = lax.cond(
            sparsity_type == 1,
            lambda s: jnp.prod(s.at[1:3].get()),
            lambda s: matrix_parallel(s),
            slice
        )
        num_edges += _num_edges
        return num_edges, None

    num_edges, _ = lax.scan(loop_fn, 0, edge_slice.T)
    
    return num_edges


def matrix_parallel(slice: Array) -> int:
    sparsity_type = slice.at[0].get()
    mp = jnp.logical_or(sparsity_type == 6, sparsity_type == 7)
    _num_edges = lax.cond(
        jnp.logical_or(mp, sparsity_type == 8),
        lambda s: 1,
        lambda s: vector_parallel(s),
        slice
    )
    return _num_edges


SPARSITY_MAP = jnp.array([2, 1, 1, 2])

def vector_parallel(slice: Array):
    sparsity_type = slice.at[0].get()
    idxs = SPARSITY_MAP[sparsity_type-2]
    return slice.at[idxs].get()


### Symmetry transformations for the extended adjacency matrix
def swap_rows(i: int, j: int, edges: Array) -> Array:
    """Swaps two rows in the extended adjacency matrix.

    Args:
        i (int): i-th row
        j (int): j-th row
        edges (Array): Extended adjacency matrix.

    Returns:
        Array: Adjacency matrix with i-th and j-th row swapped.
    """
    val1 = edges.at[i, :].get()
    val2 = edges.at[j, :].get()
    edges = edges.at[i, :].set(val2)
    edges = edges.at[j, :].set(val1)
    return edges


def swap_cols(i: int, j: int, edges: Array) -> Array:
    """Swaps two columns in the extended adjacency matrix.

    Args:
        i (int): i-th column
        j (int): j-th column
        edges (Array): Extended adjacency matrix.

    Returns:
        Array: Adjacency matrix with i-th and j-th column swapped.
    """
    val1 = edges.at[:, i].get()
    val2 = edges.at[:, j].get()
    edges = edges.at[:, i].set(val2)
    edges = edges.at[:, j].set(val1)
    return edges


def swap_inputs(i: int, j: int, edges: Array) -> Array:
    num_i, num_vo = get_shape(edges)
    return swap_rows(i+num_i-1, j+num_i-1, edges)


def swap_outputs(i: int, j: int, edges: Array) -> Array:
    return swap_cols(edges, i-1, j-1)


def _swap_intermediates(i: int, j: int, edges: Array) -> Array:
    num_i, num_vo = get_shape(edges)
    edges = swap_rows(i+num_i-1, j+num_i-1, edges)
    return swap_cols(i-1, j-1, edges)


def swap_intermediates(i: int, j: int, edges: Array) -> Array:
    """
    Symmetry operation of the computational graph that interchanges
    two vertices while preserving the computational graph.

    Args:
        i (int): i-th intermediate variable.
        j (int): j-th intermediate variable.
        edges (Array): Extended adjacency matrix.

    Returns:
        Array: Adjacency matrix with i-th and j-th intermediate swapped.
    """

    i, j = lax.cond(i < j, 
                    lambda m, n: (m, n),
                    lambda m, n: (n, m),
                    i, j)
    
    num_i, num_vo = get_shape(edges)
    _i = i+num_vo-1
    _j = j+num_vo-1
    s1 = edges.at[:, _i].get()
    s2 = edges.at[:, _j].get()
    sum1 = jnp.sum(s1[_i+1:])
    sum2 = jnp.sum(s2[_i+1:])
        
    edges = lax.cond(
        sum1 + sum2 == 0,
        lambda x: _swap_intermediates(i, j, x),
        lambda x: x,
        edges
    )
    
    return edges

