import jax
import jax.numpy as jnp
import jax.random as jrand

from .vertexgame import (make_graph, get_graph_shape, forward, 
                        reverse, minimal_markowitz, cross_country)
from graphax.examples import RoeFlux_1d, RoeFlux_3d, RobotArm_6DOF, f, g, Helmholtz
# TODO add Hessian and ViT examples


def make_benchmark_scores(graph):
    _, fwd_fmas = jax.jit(forward)(graph)
    _, rev_fmas = jax.jit(reverse)(graph)
    
    mM_order = jax.jit(minimal_markowitz, static_argnums=1)(graph, int(graph.at[0, 0, 1].get()))
    _, mM_fmas = jax.jit(cross_country)(mM_order, graph)
    
    scores = [fwd_fmas, rev_fmas, mM_fmas]
    return mM_order, scores


def make_fn(fn, *xs):
    graph = make_graph(fn, *xs)
    graph_shape = get_graph_shape(graph)
    return graph, graph_shape, fn


def make_Helmholtz():
    xs = [jnp.array([.05, .15, .15, 0.2])]
    return make_fn(Helmholtz, *xs)


def make_RoeFlux_1d():
    xs = [.01, .02, .02, .01, .03, .03]
    return make_fn(RoeFlux_1d, *xs)


def make_RoeFlux_3d():
    ul0 = jnp.array([.1])
    ul = jnp.array([.1, .2, .3])
    ul4 = jnp.array([.5])
    ur0 = jnp.array([.2])
    ur = jnp.array([.2, .2, .4])
    ur4 = jnp.array([.6])
    xs = (ul0, ul, ul4, ur0, ur, ur4)
    return make_fn(RoeFlux_3d, *xs)


def make_RobotArm_6DOF():
    xs = [.02]*6
    return make_fn(RobotArm_6DOF, *xs)


def make_f():
    key = jrand.PRNGKey(250197)
    a = jrand.uniform(key, (4,))
    b = jrand.uniform(key, (2, 3))
    c = jrand.uniform(key, (4, 4))
    d = jrand.uniform(key, (4, 1))
    xs = [a, b, c, d]
    return make_fn(f, *xs)


def make_g():
    xs = [.15]*15
    return make_fn(g, *xs)

