import jax
import jax.numpy as jnp

import graphax as gx
from graphax.examples.easy import (Simple, Helmholtz, Hole, BlackScholes, 
                                   CloudSchemes_step, Lighthouse)
from graphax.examples. randoms import f
from graphax.examples.minpack import HumanHeartDipole, PropaneCombustion
from graphax.examples.roe import RoeFlux_1d, RoeFlux_3d
from graphax.examples.differential_kinematics import RobotArm_6DOF
from alphagrad.vertexgame import make_graph


edges = make_graph(BlackScholes, jnp.ones(4), jnp.ones(4), jnp.ones(4), jnp.ones(4), jnp.ones(4))
print(edges)
print(gx.get_shape(edges))
_, ops = gx.forward(edges)
print(ops)
_, ops = gx.reverse(edges)
print(ops)

order = gx.minimal_markowitz(edges)
output, ops = gx.cross_country(order, edges)
_, ops = output
print(ops)


# edges, info = make_LIF()
# edges, info = safe_preeliminations_gpu(edges, info)
# edges, info = compress_graph(edges, info)

# print(edges, info)
# _, ops = forward_gpu(edges, info)
# print(ops)
# _, ops = reverse_gpu(edges, info)
# print(ops)


# edges, info = make_hessian()
# edges, info = safe_preeliminations_gpu(edges, info)
# edges, info = compress_graph(edges, info)

# print(edges, info)
# _, ops = forward_gpu(edges, info)
# print(ops)
# _, ops = reverse_gpu(edges, info)
# print(ops)

