import jax
import jax.numpy as jnp
import jax.random as jrand

from .vertexgame import (make_graph, get_graph_shape, forward, 
                        reverse, minimal_markowitz, cross_country)
from graphax.examples import RoeFlux_1d, RoeFlux_3d, RobotArm_6DOF, f, g
# TODO add Hessian and ViT examples


def make_benchmark_scores(graph):
    _, fwd_fmas = forward(graph)
    _, rev_fmas = reverse(graph)
    
    mM_order = minimal_markowitz(graph, int(graph.at[0, 0, 1].get()))
    out, _ = cross_country(mM_order, graph)
    mM_fmas = out[1]
    
    scores = [fwd_fmas, rev_fmas, mM_fmas]
    return mM_order, scores


def make_RoeFlux_1d():
    xs = [.01, .02, .02, .01, .03, .03]
    graph = make_graph(RoeFlux_1d, xs)
    graph_shape = get_graph_shape(graph)
    return make_graph(RoeFlux_1d, xs), graph_shape, RoeFlux_1d


def make_RoeFlux_3d():
    ul0 = jnp.array([.1])
    ul = jnp.array([.1, .2, .3])
    ul4 = jnp.array([.5])
    ur0 = jnp.array([.2])
    ur = jnp.array([.2, .2, .4])
    ur4 = jnp.array([.6])
    xs = (ul0, ul, ul4, ur0, ur, ur4)
    return make_graph(RoeFlux_3d, xs), RoeFlux_3d


def make_RobotArm_6DOF():
    xs = [.02]*6
    return make_graph(RobotArm_6DOF, xs), RobotArm_6DOF


def make_f():
    key = jrand.PRNGKey(250197)
    a = jrand.uniform(key, (4,))
    b = jrand.uniform(key, (2, 3))
    c = jrand.uniform(key, (4, 4))
    d = jrand.uniform(key, (4, 1))
    xs = [a, b, c, d]
    return make_graph(f, xs), f


def make_g():
    xs = [.15]*15
    return make_graph(g, xs), g

