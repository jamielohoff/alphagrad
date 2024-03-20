import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax import jacve
from graphax.examples import Simple, Helmholtz, Perceptron, f, Encoder
from alphagrad.vertexgame import vertex_eliminate, forward, reverse, make_graph


# # Test on simple example
# graph = make_simple()
# graph, fmas = jax.jit(forward)(graph)
# print(fmas, "8")

# graph = make_simple()
# graph, fmas = jax.jit(reverse)(graph)
# print(fmas, "6")


# # Test on Helmholtz example
# graph = make_Helmholtz()

# # Optimal elimination procedure
# graph, fmas = jax.jit(vertex_eliminate)(2, graph)
# print(fmas, "1")

# graph, _fmas = jax.jit(vertex_eliminate)(5, graph)
# fmas += _fmas
# print(_fmas, "4")

# graph, _fmas = jax.jit(vertex_eliminate)(4, graph)
# fmas += _fmas
# print(_fmas, "8")

# graph, _fmas = jax.jit(vertex_eliminate)(3, graph)
# fmas += _fmas
# print(_fmas, "4")

# graph, _fmas = jax.jit(vertex_eliminate)(1, graph)
# fmas += _fmas
# print(_fmas, "16")
# print("Result:")
# print(fmas, "33")

# graph = make_Helmholtz()
# graph, fmas = jax.jit(forward)(graph)
# print(fmas, "56")

# graph = make_Helmholtz()
# graph, fmas = jax.jit(reverse)(graph)
# print(fmas, "36")

# # Test on neural network
# key = jrand.PRNGKey(1234)

# x = jnp.ones(4)
# y = jrand.normal(key, (4,))

# w1key, b1key, key = jrand.split(key, 3)
# W1 = jrand.normal(w1key, (8, 4))
# b1 = jrand.normal(b1key, (8,))

# w2key, b2key, key = jrand.split(key, 3)
# W2 = jrand.normal(w2key, (4, 8))
# b2 = jrand.normal(b2key, (4,))

# xs = (x, y, W1, b1, W2, b2, 0., 1.)
# print(jax.make_jaxpr(Perceptron)(*xs))
# graph = make_graph(Perceptron, *xs)
# output = forward(graph)
# print(output[1])

# veres, aux = jacve(Perceptron, order="fwd", count_ops=True)(*xs)
# print("fwd num muls", aux["num_muls"])

# total_fmas = 0
# for i in range(1, 10):
#     graph, fmas = vertex_eliminate(i, graph)
#     print(i, fmas)
#     total_fmas += fmas
# print(total_fmas)

# # Test Encoder
# key = jrand.PRNGKey(250197)
# x = jnp.ones((4, 4))
# y = jnp.ones((2, 4))

# WQ1 = jnp.ones((4, 4))
# WK1 = jnp.ones((4, 4))
# WV1 = jnp.ones((4, 4))

# WQ2 = jnp.ones((4, 4))
# WK2 = jnp.ones((4, 4))
# WV2 = jnp.ones((4, 4))

# W1 = jnp.ones((4, 4))
# b1 = jnp.ones(4)

# W2 = jnp.ones((2, 4))
# b2 = jnp.ones((2, 1))
    
# xs = (x, y, WQ1, WQ2, WK1, WK2, WV1, WV2, W1, W2, b1, b2, 0., 1., 0., 1.)
# argnums = list(range(len(xs)))

# print(jax.make_jaxpr(Encoder)(*xs))

# graph = make_graph(Encoder, *xs)

key = jrand.PRNGKey(250197)
x = jnp.ones((4, 4))

WQ1 = jnp.ones((4, 4))
WK1 = jnp.ones((4, 4))
    
xs = (x, WQ1, WK1)
argnums = list(range(len(xs)))

def test_fn(x, Wq, Wk):
    q = Wq @ x
    k = Wk @ x
    a = q.T @ k
    return a

graph = make_graph(test_fn, *xs)

print(jax.make_jaxpr(test_fn)(*xs))

deriv_fn = jax.jit(jacve(test_fn, order="fwd", argnums=argnums, count_ops=True))
veres, aux = deriv_fn(*xs)
print("rev num_muls", aux["num_muls"])

output = forward(graph)
print("reverse", output[1])

