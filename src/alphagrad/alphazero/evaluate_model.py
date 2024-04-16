import os
import argparse
import time

from torch.utils.data import DataLoader

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu


parser = argparse.ArgumentParser()

parser.add_argument("--gpu", type=str, 
                    default="0", help="GPU identifier.")

parser.add_argument("--seed", type=int,
                    default=42, help="Random seed.")

parser.add_argument("--num_simulations", type=int, 
                    default=100, help="Number of simulations.")

parser.add_argument("--num_inputs", type=int, 
                    default=20, help="Number input variables.")

parser.add_argument("--num_actions", type=int, 
                    default=105, help="Number of actions.")

parser.add_argument("--num_outputs", type=int, 
                    default=20, help="Number of output variables.")

parser.add_argument("--load_model", type=str,
                    default=None, help="Path to the model weights that have to be loaded.")

parser.add_argument("--num_layers", type=int,
                    default=6, help="Number of transformer blocks.")

parser.add_argument("--num_layers_policy", type=int,
                    default=2, help="Number of transformer blocks.")

parser.add_argument("--num_heads", type=int,
                    default=8, help="Number of attention heads.")

parser.add_argument("--embedding_dim", type=int,
                    default=128, help="Dimension of the token embeddings.")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import equinox as eqx

from graphax import jacve, tree_allclose
from graphax.examples.differential_kinematics import RobotArm_6DOF
from graphax.examples.roe import RoeFlux_1d

from alphagrad import forward, reverse, cross_country, make_graph
from alphagrad.vertexgame.transforms import (minimal_markowitz, 
                                            safe_preeliminations,
                                            compress,
                                            embed)
from alphagrad.vertexgame import step

from alphagrad.utils import get_masked_logits
from alphagrad.alphazero.environment_interaction import (make_recurrent_fn,
							                    make_environment_interaction)
from alphagrad.transformer.sequential_transformer import SequentialTransformerModel
from alphagrad.evaluate import evaluate


key = jrand.PRNGKey(args.seed)
NUM_INPUTS = args.num_inputs
NUM_INTERMEDIATES = args.num_actions
NUM_OUTPUTS = args.num_outputs
INFO = [NUM_INPUTS, NUM_INTERMEDIATES, NUM_OUTPUTS]

F = RobotArm_6DOF
xs = jrand.normal(key, (6,)).tolist()


nn_key, env_key, key = jrand.split(key, 3)
MODEL = SequentialTransformerModel(INFO, 
									args.embedding_dim, 
									args.num_layers, 
									args.num_heads,
									ff_dim=1024,
									num_layers_policy=args.num_layers_policy,
									policy_ff_dims=[1024, 512],
									value_ff_dims=[512, 256, 64], 
									key=nn_key)


MODEL = eqx.tree_deserialise_leaves(args.load_model, MODEL)
print("Loading model complete.")

batched_step = jax.vmap(step)
batched_get_masked_logits = jax.vmap(get_masked_logits, in_axes=(0, 0))

recurrent_fn = make_recurrent_fn(MODEL,
                                batched_step, 
                                batched_get_masked_logits)

env_interaction = make_environment_interaction(INFO, 
                                            	args.num_simulations,
                                                recurrent_fn,
                                                batched_step,
												temperature=0)

edges = make_graph(F, 1., 1., 1., 1., 1., 1.)
output_vertices = jnp.nonzero(edges.at[2, 0, :].get())[0]
output_vertices = [int(i)+1 for i in output_vertices]
edges, preelim_order = safe_preeliminations(edges, return_preeliminated=True)
edges = compress(edges)
edges = embed(key, edges, INFO)
print("Preprocessing graph complete.")

reward, alphagrad_order = eqx.filter_jit(evaluate)(MODEL, env_interaction, env_key, edges[jnp.newaxis, :])
print("Evaluation complete.")

preelim_order = [int(i) for i in preelim_order if i > 0]
num_inputs, num_intermediates, num_outputs = edges.at[0, 0, 0:3].get()
alphagrad_order = alphagrad_order[0, :num_intermediates-num_outputs]
sorted_order = sorted(alphagrad_order)
alphagrad_order = [sorted_order.index(int(i))+1 for i in alphagrad_order]


order = []
order.extend(preelim_order)
for vertex in alphagrad_order:
    while vertex in order or vertex in output_vertices:
        vertex += 1
    order.append(vertex)
print("Elimination order", order)


edges = make_graph(F, 1., 1., 1., 1., 1., 1.)
_, nops = jax.jit(cross_country)(order, edges)
print("CCE ops", nops)

_, nops = jax.jit(forward)(edges)
print("Forward ops", nops)

_, nops = jax.jit(reverse)(edges)
print("Reverse ops", nops)

mM_order = minimal_markowitz(edges)
mM_order = [int(i) for i in mM_order]
_, nops = jax.jit(cross_country)(mM_order, edges)
print("minimal Markowitz ops", nops)


steps = 1000

rev_fn = jax.jit(jacve(F, "rev", (0, 1, 2, 3, 4, 5)))
st = time.time()
for _ in range(steps):
    res = rev_fn(1., 1., 1., 1., 1., 1.)
print("Rev time", time.time() - st)

cce_fn = jax.jit(jacve(F, order, (0, 1, 2, 3, 4, 5)))
st = time.time()
for _ in range(steps):
    veres = cce_fn(1., 1., 1., 1., 1., 1.)
print("CCE time", time.time() - st)

mM_fn = jax.jit(jacve(F, mM_order, (0, 1, 2, 3, 4, 5)))
st = time.time()
for _ in range(steps):
    res = mM_fn(1., 1., 1., 1., 1., 1.)
print("mM time", time.time() - st)

jax_rev_fn = jax.jit(jax.jacrev(F, (0, 1, 2, 3, 4, 5)))
st = time.time()
for _ in range(steps):
    revres = jax_rev_fn(1., 1., 1., 1., 1., 1.)
print("Jax Rev time", time.time() - st)

print(tree_allclose(veres, revres))

