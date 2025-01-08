from typing import Sequence, Tuple
import time

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import jraph
from jraph import GraphsTuple

from chex import Array

from alphagrad.vertexgame import make_graph
from alphagrad.vertexgame.core import ADD_SPARSITY_MAP, MUL_SPARSITY_MAP, CONTRACTION_MAP


# jax.config.update("jax_disable_jit", True)


# The size of these buffers are the main bottleneck of the algorithm.
IN_VAL_BUFFER_SIZE = 10
OUT_VAL_BUFFER_SIZE = 10


def sparse_mul(in_jac: Array, out_jac: Array) -> Tuple[float, Array]:
    """
    Function that computes the shape of the resulting sparse multiplication of 
    the Jacobian of the incoming edge and the Jacobian of the outgoing edge.
    It also computes the number of necessary multiplications to do so.

    Args:
        in_jac (Array): Sparsity type and Jacobian shape of the incoming edge.
        out_jac (Array): Sparsity type and Jacobian shape of the outgoing edge.

    Returns:
        Tuple: Tuple containing the sparsity type and Jacobian shape of the 
                resulting edge as well as the number of multiplications.
    """
    # Get the sparsity type of the incoming and outgoing edge and compute the
    # sparsity type of the resulting edge
    in_sparsity = in_jac[0].astype(jnp.int32)
    out_sparsity = out_jac[0].astype(jnp.int32)
    res_sparsity = jnp.array([MUL_SPARSITY_MAP[in_sparsity, out_sparsity]])
    
    # Check how the contraction between incoming and outgoing Jacobian is done
    # and compute the resulting number of multiplications
    contraction_map = CONTRACTION_MAP[:, in_sparsity, out_sparsity]
    factors = jnp.concatenate((out_jac[1:3], jnp.abs(out_jac[3:]), in_jac[3:]))
    masked_factors = lax.cond(jnp.sum(contraction_map) > 0,
                                lambda a: jnp.where(contraction_map > 0, a, 1),
                                lambda a: jnp.zeros_like(a), 
                                factors)

    fmas = jnp.prod(masked_factors)
    return fmas, jnp.concatenate([res_sparsity, in_jac[1:3], out_jac[3:]])


def sparse_add(in_jac: Array, out_jac: Array) -> Array:
    """
    Function that computes the shape of the resulting sparse addition of the
    Jacobian of the incoming edge and the Jacobian of the outgoing edge.

    Args:
        in_jac (Array): Sparse type and Jacobian shape of the incoming edge.
        out_jac (Array): Sparse type and Jacobian shape of the outgoing edge.

    Returns:
        Array: Sparse type and Jacobian shape of the resulting edge.
    """
    in_sparsity = in_jac[0].astype(jnp.int32)
    out_sparsity = out_jac[0].astype(jnp.int32)
    res_sparsity = jnp.array([ADD_SPARSITY_MAP[in_sparsity, out_sparsity]])
    return jnp.concatenate([res_sparsity, in_jac[1:]])
    


def del_and_copy_edge(n: int, 
                        i: int, 
                        pos_buf: Array, 
                        jacs_buf: Array, 
                        edge_conn: Array, 
                        edge_vals: Array) -> Tuple:
    """
    Function that deletes the respective edge at position `i` from `edge_conn`
    and `edge_vals` and copies the edge into the buffers `pos_buf` and `vals_buf`.
    Furthermore, it deletes the the edge from `edge_conn` and the value of the 
    edge from `edge_vals`.

    Args:
        n (int): Global counter variable that track where we are in the sparse
                representation of our graph.
        i (Array): Global counter variable that tracks the index of the buffer.
        pos_buf (Array): Buffer that stores the connectivity of the edge in question.
        vals_buf (Array): Buffer that stores the value of the edge in question.
        edge_conn (Array): Connectivity of the graph. Essentially contains
                            senders and receivers of the graph.
        edge_vals (Array): Edge values of the graph.

    Returns:
        Tuple: A tuple containing the updated counter variables, buffers and
                state of the computational graph.
    """
    # Fill the position and value buffers with the edge we want to delete
    pos_buf = pos_buf.at[i, :].set(edge_conn[n])
    jacs_buf = jacs_buf.at[i, :].set(edge_vals[n])
    
    # Delete the edge from the graph representation
    edge_conn = edge_conn.at[n].set(-1)
    edge_vals = edge_vals.at[n].set(0)
    
    return (i+1, pos_buf, jacs_buf, edge_conn, edge_vals)


def cond(condition, true_fn, false_fn, *xs):
    if condition:
        return true_fn(*xs)
    else:
        return false_fn(*xs)
    
    
def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)
        

def get_edges(vertex: int, edge_conn: Array, edge_vals: Array) -> Tuple:    
    """
    Function that iterates through the sparse representation of the computational
    graph and looks for edges connected to a specific vertex.
    For every edge connected to the vertex, the connectivity and the value
    of the respective edge are written to buffers depending on whether the edge
    is ingoing (`in_` prefix) or outgoing (`out_`prefix).

    Args:
        vertex (int): Computational graph vertex we want to eliminate according
                    to the vertex elimination scheme
        edge_conn (Array): Connectivity of the graph. Essentially contains
                            senders and receivers of the graph.
        edge_vals (Array): Edge values of the graph.
    Returns:
        Tuple: A tuple containing the updated buffers and state of the graph.
    """
    
    # Define identity function for lax.cond
    def id(*xs):
        return xs[1:]
    
    def loop_fn(carry, _):
        # n tracks where we are in the sparse graph representation
        # i, j track the current indices of in the ingoing and outgoing 
        # representations of the buffer
        n, i, j, in_pos, in_vals, out_pos, out_vals, edge_conn, edge_vals = carry

        # Get the current edge
        edge = edge_conn[n]
        
        # Edge is ingoing edge of the vertex
        out = lax.cond(edge[1] == vertex, del_and_copy_edge, id, 
                        n, i, in_pos, in_vals, edge_conn, edge_vals)
        i, in_pos, in_vals, edge_conn, edge_vals = out
        
        # Edge is outgoing edge of the vertex
        out = lax.cond(edge[0] == vertex, del_and_copy_edge, id, 
                        n, j, out_pos, out_vals, edge_conn, edge_vals)
        j, out_pos, out_vals, edge_conn, edge_vals = out
                        
        carry = (n+1, i, j, in_pos, in_vals, out_pos, out_vals, edge_conn, edge_vals)
        return carry, 0
    
    # Loop running over all the edges in the sparse representation of the graph
    carry_init = (0, 0, 0, -jnp.ones((IN_VAL_BUFFER_SIZE, 2)), 
                            jnp.zeros((IN_VAL_BUFFER_SIZE, 5)), 
                            -jnp.ones((OUT_VAL_BUFFER_SIZE, 2)), 
                            jnp.zeros((OUT_VAL_BUFFER_SIZE, 5)), edge_conn, edge_vals)
    output, _ = lax.scan(loop_fn, carry_init, None, length=edge_conn.shape[0])
    
    return output[1:]


def add_edge(edge: Array, 
            in_jac: Array, 
            out_jac: Array, 
            n: int, 
            k: int, 
            edge_conn: Array, 
            edge_vals: Array, 
            free_idxs: Array,
            n_ops: int) -> Tuple:
    """
    Function that adds an edge to the computational graph. If the edge already
    exists, the current value is added to the product of the ingoing and outgoing
    edge. It uses the `free_idxs` buffer to keep track of where in the 
    computational graph representation we can add new edges after the edges 
    connected to the vertex in question have been removed by the `get_edges` function.

    Args:
        edge (Array): The edge that we want to add to the computational graph.
                        It consists of the sender and receiver of the edge.
        in_jac (Array): Sparsity type and Jacobian shape of the ingoing edge.
        out_jac (Array): Sparsity type and Jacobian shape of the outgoing edge.
        n (int): Counter variable that tracks where we are in the `edge_combos`
                buffer of the `make_new_edges` function.
        k (int): Counter variable that tracks where we are in the `free_idxs` buffer.
        edge_conn (Array): Connectivity of the graph. Essentially contains
                            senders and receivers of the graph.
        edge_vals (Array): Edge values of the graph.
        free_idxs (Array): Buffer that keeps track of where we can add new edges
                            in the graph representation.
        n_ops (int): Counter variable that tracks the number of multiplications
                    incurred by the vertex elimination.

    Returns:
        Tuple: Returns a tuple containing the updated counter variables, buffers
                and state of the computational graph.
    """
    # Check if the edge we want to create aleady exists
    edge_exists = jnp.all(edge_conn == edge, axis=-1)
    existing_edge_idx = jnp.argwhere(edge_exists, size=1, fill_value=-1)[0][0]

    # Depending on whether the edge exists or not, we either add the value of the
    # edge to the existing edge or create a new edge
    # TODO add more documentation
    ops, jac = sparse_mul(in_jac, out_jac)
    k, idx, jac = lax.cond(existing_edge_idx > -1,
                            lambda k: (k, existing_edge_idx, 
                                        sparse_add(jac, edge_vals[existing_edge_idx])),
                            lambda k: (k+1, free_idxs[k].astype(jnp.int32), jac), k)

    # Add the edge to the graph representation
    edge_vals = edge_vals.at[idx].set(jac)
    edge_conn = edge_conn.at[idx].set(edge)
    return n+1, k, edge_conn, edge_vals, n_ops+ops


def make_new_edges(edge_combos: Array, 
                    in_vals: Array, 
                    out_vals: Array, 
                    edge_conn: Array, 
                    edge_vals: Array, 
                    free_idxs: Array) -> Tuple:
    """
    Function that creates new edges in the computational graph. It uses the 
    combination of ingoing and outgoing edges to create new edges stored in
    the `edge_combos` variable. 

    Args:
        edge_combos (Array): 
        in_vals (Array): Values of the ingoing edges.
        out_vals (Array): Values fo the outgoing edges.
        edge_conn (Array): Connectivity of the graph. Essentially contains senders
                            and receivers of the graph.
        edge_vals (Array): Edge values of the graph.
        free_idxs (Array): Buffer that keeps track of where we can add new edges.

    Returns:
        Tuple: Returns a tuple containing the updated counter variables and
                graph representation as well as the number of multiplications
                that the vertex elimination incurred.
    """
    # Define identity function for lax.cond
    def id(edge, in_val, out_val, n, k, edge_conn, edge_vals, free_idxs, n_ops):
        return (n+1, k, edge_conn, edge_vals, n_ops)
    
    def loop_fn(carry, edge):
        n, k, edge_conn, edge_vals, n_ops = carry
        # Check whether the edge is valid by checking if the sender and receiver
        # are not -1.
        is_valid_edge = jnp.logical_and(edge[0]>=0, edge[1]>=0)
        
        # Add the edge to the graph representation
        # TODO use the correct number of `n` for a mixed size of senders and receivers
        in_val = in_vals[n % IN_VAL_BUFFER_SIZE]
        out_val = out_vals[n // OUT_VAL_BUFFER_SIZE]
        carry = lax.cond(is_valid_edge, add_edge, id,
                        edge, in_val, out_val, n, k, edge_conn, edge_vals, free_idxs, n_ops)

        return carry, None
        
    # Loop running over all the edges in the `edge_combos` buffer
    carry_init = (0, 0, edge_conn, edge_vals, 0.)
    output, _ = lax.scan(loop_fn, carry_init, edge_combos)
    return output[1:]


def vertex_eliminate(vertex: int, graph: GraphsTuple) -> GraphsTuple:
    """
    Function that implements vertex elimination in a sparse, jittable fashion.
    TODO add more documentation

    Args:
        graph (GraphsTuple): Graph representation of the computational graph.
        vertex (int): Vertex that we want to eliminate.

    Returns:
        GraphsTuple: The resulting graph after the vertex elimination.
    """
    # Divide the graph representation into its components
    # edge_conn contains the senders and receivers of the graph, i.e. the connectivity
    # of the vertices with each other
    # edge_vals contains the values of the edges
    edge_conn = jnp.stack([graph.senders, graph.receivers]).T
    edge_vals = graph.edges
        
    # Get the edges connected to the vertex
    # i, j are the used number of places in the buffers, i.e. the number of ingoing
    # and outgoing edges
    i, j, in_pos, in_vals, out_pos, out_vals, edge_conn, edge_vals = get_edges(vertex, edge_conn, edge_vals)
    
    jax.debug.print("vertex: {v}, in edges: {i}, out edges: {j}", v=vertex-5, i=i, j=j)

    # Calculate the new edges and where in the graph representation we can add them
    is_zero = jnp.all(edge_vals == 0, axis=-1)
    free_idxs = jnp.argwhere(is_zero, size=10).flatten()    
    edge_combos = jnp.stack(jnp.meshgrid(in_pos[:, 0], out_pos[:, 1]))
    edge_combos = edge_combos.reshape(2, IN_VAL_BUFFER_SIZE*OUT_VAL_BUFFER_SIZE).T

    # Add the new edges to the graph representation
    # k is the number of newly created edges
    output = make_new_edges(edge_combos, in_vals, out_vals, edge_conn, edge_vals, free_idxs)
    k, edge_conn, edge_vals, n_ops = output
    
    # Build everything into a new graph
    senders = edge_conn[:, 0]
    receivers = edge_conn[:, 1]
    
    nodes = graph.nodes.at[vertex].set(1)
    graph = GraphsTuple(nodes=nodes,
                        edges=edge_vals,
                        senders=senders,
                        receivers=receivers,
                        n_node=graph.n_node,
                        n_edge=graph.n_edge-i-j+k, # TODO this does not give the correct value yet
                        globals=graph.globals+n_ops)
    return graph


def cross_country(order: Sequence[int], graph: GraphsTuple) -> GraphsTuple:
    """
    Function that implements the cross-country AD using the 
    vertex elimination algorithm.
    TODO add more documentation

    Args:
        order (Sequence[int]): The order in which the vertices are eliminated.
        graph (GraphsTuple): Graph representation of the computational graph.

    Returns:
        GraphsTuple: The resulting graph after vertex elimination.
    """
    # Eliminate the vertices only if they are not masked
    def loop_fn(carry, i):
        graph, d = carry
        graph = lax.cond(graph.nodes[i] == 0,
                        lambda g: vertex_eliminate(i, g),
                        lambda g: g, graph)
        
        out = jnp.array([i-5, graph.globals[0][0] - d])
        carry = (graph, graph.globals[0][0])
        return carry, out
    
    # Looping over all the vertices in the order specified
    graph, out = lax.scan(loop_fn, (graph, 0), order)
    out = jnp.stack(out).T
    return graph[0], out
    
    
def forward(graph: GraphsTuple) -> GraphsTuple:
    """
    Function that implements forward-mode AD on the computational graph.

    Args:
        graph (GraphsTuple): Graph representation of the computational graph.

    Returns:
        GraphsTuple: The resulting graph after executing forward-mode AD.
    """
    order = jnp.arange(0, len(graph.nodes))
    graph = cross_country(order, graph)
    return graph


def reverse(graph: GraphsTuple) -> GraphsTuple:
    """
    Function that implements reverse-mode AD on the computational graph.

    Args:
        graph (GraphsTuple): Graph representation of the computational graph.

    Returns:
        GraphsTuple: The resulting graph after executing reverse-mode AD.
    """
    order = jnp.arange(0, len(graph.nodes))[::-1]
    graph, out = cross_country(order, graph)
    return graph, out


def embed(num_nodes: int, num_edge: int, graph: GraphsTuple) -> GraphsTuple:
    """
    Function that embeds a computational graph into a larger computational graph
    with `num_nodes` and `num_edge` nodes and edges respectively.

    Args:
        num_nodes (int): Number of nodes of the new computational graph.
        num_edge (int): Number of edges of the new computational graph.
        graph (GraphsTuple): Graph representation of the computational graph.

    Returns:
        GraphsTuple: The resulting graph after embedding.
    """
    node_padding = num_nodes - graph.n_node
    edge_padding = num_edge - graph.n_edge
    
    # Add padding to the nodes
    nodes = jnp.concatenate([graph.nodes, jnp.ones(node_padding)])
    edges = jnp.concatenate([graph.edges, jnp.zeros((edge_padding, 5))])
    
    senders = jnp.concatenate([graph.senders, -jnp.ones(edge_padding)])
    receivers = jnp.concatenate([graph.receivers, -jnp.ones(edge_padding)])
    
    graph = GraphsTuple(nodes=nodes,
                        edges=edges,
                        senders=senders,
                        receivers=receivers,
                        n_node=num_nodes,
                        n_edge=num_edge,
                        globals=graph.globals)
    return graph


from graphax import jacve
from graphax.sparse.utils import count_muls_jaxpr
from graphax.examples import (RobotArm_6DOF, RoeFlux_1d, f, Perceptron, 
                            Simple, Lighthouse, Hole, Helmholtz)


F = RobotArm_6DOF
xs = [jnp.zeros((1,))]*6


# def NeuralNetwork(x, W1, b1, W2, b2, y):
#     y1 = W1 @ x
#     z1 = y1 + b1
#     a1 = jnp.tanh(z1)
    
#     y2 = W2 @ a1
#     z2 = y2 + b2
#     a2 = jnp.tanh(z2)
#     d = a2 - y
#     return .5*jnp.sum(d**2)

# F = NeuralNetwork
# key = jrand.PRNGKey(42)
# x = jnp.ones(4)
# y = jrand.normal(key, (4,))

# w1key, b1key, key = jrand.split(key, 3)
# W1 = jrand.normal(w1key, (8, 4))
# b1 = jrand.normal(b1key, (8,))

# w2key, b2key, key = jrand.split(key, 3)
# W2 = jrand.normal(w2key, (4, 8))
# b2 = jrand.normal(b2key, (4,))

# xs = [x, W1, b1, W2, b2, y]

# F = f

# a = jnp.ones(4)
# b = jnp.ones((2, 3))
# c = jnp.ones((4, 4))
# d = jnp.ones((3, 3))
# e = jnp.ones((4, 1))
# xs = [a, b, c, d, e]

# F = Helmholtz
# xs = [jnp.ones((4,))]

# F = RoeFlux_1d
# xs = [jnp.ones((1,))]*6


args = range(len(xs))
dense_graph = make_graph(F, *xs)
print(dense_graph.shape)
jaxpr = jax.make_jaxpr(jacve(F, order="rev", argnums=args, count_ops=True))(*xs)
deriv_jaxpr = jax.make_jaxpr(jacve(F, order="rev", argnums=args, count_ops=True))(*xs)

jacobian, aux = jax.jit(jacve(F, order="rev", argnums=args, count_ops=True, ))(*xs)
print("num muls:", aux["num_muls"], "num_adds:", aux["num_adds"])
print(count_muls_jaxpr(deriv_jaxpr) - count_muls_jaxpr(jaxpr))


nonzeros = jnp.nonzero(dense_graph[1, 1:])

nodes = jnp.where(dense_graph[1, 0, :] > 0, 1, 0)
num_input_nodes = dense_graph.shape[1] - dense_graph.shape[2] - 1

input_nodes = jnp.ones(num_input_nodes)
nodes = jnp.concatenate([input_nodes, nodes])


edges = []
for i, j in zip(nonzeros[0], nonzeros[1]):
    edge = dense_graph[:, i+1, j]
    edges.append(edge)
edges = jnp.stack(edges)

sparse_graph = GraphsTuple(nodes=nodes,
                            edges=edges,
                            senders=nonzeros[0],
                            receivers=num_input_nodes+nonzeros[1],
                            n_node=len(nodes),
                            n_edge=len(nonzeros[0]),
                            globals=jnp.array([[0.]]))


from alphagrad.vertexgame import reverse as old_reverse
from alphagrad.vertexgame.transforms import embed as old_embed

embedded_graph = sparse_graph
# print("sparse_graph", sparse_graph)

# TODO the sparse version of vertex elimination does not yield the correct number
# of multiplications and additions
start = time.time()
out_graph, out = jax.jit(reverse)(embedded_graph)
end = time.time()
print("jraph reverse time jit", end-start, out_graph.globals)

start = time.time()
out_graph, out = jax.jit(reverse)(embedded_graph)
end = time.time()
print("jraph reverse time", end-start, out_graph.globals)
# print("out_graph", out_graph)

print([(int(i), int(j)) for i, j in zip(out[0], out[1])])

# key = jrand.PRNGKey(123)
# print("embedding takes time")
# dense_graph = old_embed(key, dense_graph, [20, 150, 20])
start = time.time()
out_graph, nops = jax.jit(old_reverse)(dense_graph)
end = time.time()
print("alphagrad time jit", end-start, nops)

start = time.time()
out_graph, nops = jax.jit(old_reverse)(dense_graph)
end = time.time()
print("alphagrad time", end-start, nops)


import equinox as eqx
import jax.nn as jnn
from typing import Callable, Sequence


def add_self_edges_fn(receivers: jnp.ndarray, senders: jnp.ndarray,
                      total_num_nodes: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Adds self edges. Assumes self edges are not in the graph yet."""
    receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)), axis=0)
    senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)), axis=0)
    return receivers, senders


class EdgeGATLayer(eqx.Module):
    edge_update: eqx.nn.Linear
    node_update: eqx.nn.Linear
    attn_logits: eqx.nn.Linear
    
    update_edge_fn: Callable
    update_node_fn: Callable
    attn_logit_fn: Callable
    attn_reduce_fn: Callable
    edge_gat_fn: Callable
    
    def __init__(self, edge_feature_shapes, node_feature_shapes, key):
        edge_key, node_key, attn_key = jrand.split(key, 3)
        
        in_edge_size = edge_feature_shapes[0] + 2*node_feature_shapes[0]
        out_edge_size = edge_feature_shapes[1]
        self.edge_update = eqx.nn.Linear(in_edge_size, out_edge_size, key=edge_key)
        
        in_node_size = 2*edge_feature_shapes[1] + node_feature_shapes[0]
        out_node_size = node_feature_shapes[1]
        self.node_update = eqx.nn.Linear(in_node_size, out_node_size, key=node_key)
        
        in_attn_size = edge_feature_shapes[1] + 2*node_feature_shapes[0]
        self.attn_logits = eqx.nn.Linear(in_attn_size, 1, key=attn_key)
        
        self.edge_gat_fn = jraph.GraphNetGAT(
            update_edge_fn=self.update_edge_fn, 
            update_node_fn=self.update_node_fn,
            attention_logit_fn=self.attn_logit_fn,
            attention_reduce_fn=self.attn_reduce_fn,
        )
        
    def update_edge_fn(self, edges, sent_attrs, recv_attrs, glob_attrs):
        xs = jnp.concatenate((sent_attrs, recv_attrs, edges), axis=1)
        out = jax.vmap(self.edge_update)(xs)
        return jnn.leaky_relu(out)
    
    # This is where the edge-to-node aggregation happens
    def update_node_fn(self, nodes, sent_attrs, recv_attrs, glob_attrs):
        xs = jnp.concatenate((sent_attrs, recv_attrs, nodes), axis=1)
        out = jax.vmap(self.node_update)(xs)
        return jnn.leaky_relu(out)
        
    def attn_logit_fn(self, edges, sent_attrs, recv_attrs, glob_attrs):
        xs = jnp.concatenate((sent_attrs, recv_attrs, edges), axis=1)
        out = jax.vmap(self.attn_logits)(xs)
        return jnn.leaky_relu(out)
    
    def attn_reduce_fn(self, edges, weights):
        weighted_edges = edges*weights
        return weighted_edges
    
    def __call__(self, graph: jraph.GraphsTuple):
        return self.edge_gat_fn(graph)


class EdgeGATNetwork(eqx.Module):
    sparsity_embedding: eqx.nn.Embedding
    op_type_embedding: eqx.nn.Embedding
    
    gat_layers: Sequence[EdgeGATLayer]
    output_layer: eqx.nn.Linear
    
    def __init__(self, sparsity_embedding_size: int, op_type_embedding_size: int, 
                edge_feature_shapes, node_feature_shapes, key):
        assert len(edge_feature_shapes) == len(node_feature_shapes), "The number"\
            "of `edge_feature_shapes` has to be equal the number of `node_feature_shapes`"
        embed_key1, embed_key2, out_key = jrand.split(key, 3)
        self.sparsity_embedding = eqx.nn.Embedding(22, sparsity_embedding_size, key=embed_key1)
        self.op_type_embedding = eqx.nn.Embedding(10, op_type_embedding_size, key=embed_key2)
        self.output_layer = eqx.nn.Linear(node_feature_shapes[-1], 1, key=out_key)
        
        keys = jrand.split(key, len(edge_feature_shapes))
        self.gat_layers = []
        self.gat_layers.append(EdgeGATLayer(
                                (sparsity_embedding_size+4, edge_feature_shapes[0]), 
                                (op_type_embedding_size, node_feature_shapes[0]),
                                keys[0]))
        
        for i, ikey in enumerate(keys[1:]):
            self.gat_layers.append(
                EdgeGATLayer(
                    (edge_feature_shapes[i], edge_feature_shapes[i+1]), 
                    (node_feature_shapes[i], node_feature_shapes[i+1]),
                    ikey
                )
            )
            
    def apply_sparsity_embedding(self, graph):
        sparsity_embeddings = jax.vmap(self.sparsity_embedding)(11+graph.edges[:, 0])
        embed_edges = jnp.concatenate([sparsity_embeddings, edges[:, 1:]], axis=-1)
        return graph._replace(edges=embed_edges)
    
    def apply_op_type_embedding(self, graph):
        embed_nodes = jax.vmap(self.op_type_embedding)(graph.nodes[:, 0])
        return graph._replace(nodes=embed_nodes)
    
    def __call__(self, graph: jraph.GraphsTuple):
        mask = graph.nodes
        graph = self.apply_sparsity_embedding(graph)
        graph = self.apply_op_type_embedding(graph)
        
        for gat_layer in self.gat_layers:
            graph = gat_layer(graph)
            
        # TODO implement value function as well using the global attr?
        nodes = jax.vmap(self.output_layer)(graph.nodes)
        nodes = jnp.squeeze(nodes)
        return jnp.where(jnp.squeeze(mask) > 0., -jnp.inf, nodes)


key = jrand.PRNGKey(42)
edge_gat_net = EdgeGATNetwork(4, 4, (32, 32, 32), (16, 16, 16), key)


sparse_graph = sparse_graph._replace(
    nodes=jnp.astype(sparse_graph.nodes[:, jnp.newaxis], jnp.int32),
    edges=jnp.astype(sparse_graph.edges, jnp.int32),
    senders=jnp.astype(sparse_graph.senders, jnp.int32),
    receivers=jnp.astype(sparse_graph.receivers, jnp.int32),
)

logits = edge_gat_net(sparse_graph)
print(jnn.softmax(logits))

