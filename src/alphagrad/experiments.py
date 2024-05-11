import jax
import jax.numpy as jnp
import jax.random as jrand

from .vertexgame import (make_graph, get_graph_shape, forward, 
                        reverse, minimal_markowitz, cross_country)
from graphax.examples import (RoeFlux_1d, RoeFlux_3d, RobotArm_6DOF, f, g, Helmholtz,
                                Perceptron, HumanHeartDipole, PropaneCombustion, Encoder,
                                BlackScholes_Jacobian)

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


def make_HumanHeartDipole():
    xs = [.15]*8
    return make_fn(HumanHeartDipole, *xs)


def make_PropaneCombustion():
    xs = [.15]*11
    return make_fn(PropaneCombustion, *xs)


def make_RoeFlux_1d():
    xs = [.01, .02, .02, .01, .03, .03]
    return make_fn(RoeFlux_1d, *xs)


def make_RoeFlux_3d():
    batchsize = 1
    ul0 = jnp.array([.1])
    ul = jnp.array([.1, .2, .3])
    ul4 = jnp.array([.5])
    ur0 = jnp.array([.2])
    ur = jnp.array([.2, .2, .4])
    ur4 = jnp.array([.6])
    xs = (ul0, ul, ul4, ur0, ur, ur4)
    xs = [jnp.tile(x[jnp.newaxis, ...], (batchsize, 1)) for x in xs]
    return make_fn(jax.vmap(RoeFlux_3d), *xs)


def make_RobotArm_6DOF():
    xs = [.02]*6
    return make_fn(RobotArm_6DOF, *xs)


def make_f():
    key = jrand.PRNGKey(250197)
    a = jrand.uniform(key, (4,))
    b = jrand.uniform(key, (2, 3))
    c = jrand.uniform(key, (4, 4))
    d = jrand.uniform(key, (4, 1))
    xs = (a, b, c, d)
    return make_fn(f, *xs)


def make_g():
    batchsize = 1
    xs = [jnp.array([.15])]*15
    xs = [jnp.tile(x[jnp.newaxis, ...], (batchsize, 1)) for x in xs]
    return make_fn(jax.vmap(g), *xs)


def make_Perceptron():
    key = jrand.PRNGKey(1234)

    x = jnp.ones(4)
    y = jrand.normal(key, (4,))

    w1key, b1key, key = jrand.split(key, 3)
    W1 = jrand.normal(w1key, (8, 4))
    b1 = jrand.normal(b1key, (8,))

    w2key, b2key, key = jrand.split(key, 3)
    W2 = jrand.normal(w2key, (4, 8))
    b2 = jrand.normal(b2key, (4,))

    xs = (x, y, W1, b1, W2, b2, 0., 1.)
    return make_fn(Perceptron, *xs)


def make_Encoder():
    key = jrand.PRNGKey(250197)
    x = jnp.ones((4, 4))
    y = jrand.normal(key, (2, 4))

    wq1key, wk1key, wv1key, key = jrand.split(key, 4)
    WQ1 = jrand.normal(wq1key, (4, 4))
    WK1 = jrand.normal(wk1key, (4, 4))
    WV1 = jrand.normal(wv1key, (4, 4))

    wq2key, wk2key, wv2key, key = jrand.split(key, 4)
    WQ2 = jrand.normal(wq2key, (4, 4))
    WK2 = jrand.normal(wk2key, (4, 4))
    WV2 = jrand.normal(wv2key, (4, 4))

    w1key, w2key, b1key, b2key = jrand.split(key, 4)
    W1 = jrand.normal(w1key, (4, 4))
    b1 = jrand.normal(b1key, (4,))

    W2 = jrand.normal(w2key, (2, 4))
    b2 = jrand.normal(b2key, (2, 1))
    
    xs = (x, y, WQ1, WQ2, WK1, WK2, WV1, WV2, W1, W2, b1, b2, 0., 1., 0., 1.)
    return make_fn(Encoder, *xs)


def make_BlackScholes_Jacobian():
    xs = (1., 1., 1., 1., 1.)
    return make_fn(BlackScholes_Jacobian, *xs)

