from tqdm import tqdm
from torch.utils.data import DataLoader

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp

from graphax import jacve, tree_allclose

from . import forward, reverse
from . import make_graph
from .transforms import clean, compress
from .dataset import GraphDataset


jaxpr = ""
fn = lambda x: x


def get_fwd_rev_ops(edges):
    _, fwd_ops = forward(edges)
    _, rev_ops = reverse(edges)
    return fwd_ops, rev_ops


def check_graphax_integrity(path):
    graph_dataset = GraphDataset(path, include_code=True)
    dataloader = DataLoader(graph_dataset, 
                            batch_size=32, 
                            shuffle=False, 
                            num_workers=8, 
                            drop_last=False)

    fwd_successes,  rev_successes = [], []
    for codes, edges in tqdm(dataloader):
        for c, e in zip(codes, edges):
            # # Calculate ops for edges
            # e = jnp.array(e)
            # # edges_fwd_ops, edges_rev_ops = get_fwd_rev_ops(e)
            # _, ops = forward(e)
            
            # Calculate ops for jacve
            c = c.decode("utf-8")
            c += "\n"
            c += "global fn\n"
            c += "fn = f"
            exec(c, globals())
            xs = [jnp.ones(invar.aval.shape)*0.01 for invar in jaxpr.jaxpr.invars]
            argnums = [i for i in range(len(xs))]
            
            e = make_graph(jaxpr, *xs)
            e = clean(e)
            e = compress(e)
            # _, ops = forward(e)
            edges_fwd_ops, edges_rev_ops = get_fwd_rev_ops(e)
                
            grad_fn = jax.jit(jacve(fn, order="fwd", argnums=argnums, count_ops=True))
            fwd_grad, fwd_ops = grad_fn(*xs)
                    
            grad_fn = jax.jit(jacve(fn, order="rev", argnums=argnums, count_ops=True))
            rev_grad, rev_ops = grad_fn(*xs)
            
            if len(xs) < len(jaxpr.jaxpr.outvars):
                jax_grad = jax.jit(jax.jacfwd(fn, argnums=argnums))(*xs)
                print(tree_allclose(fwd_grad, jax_grad, equal_nan=True))
            else:
                jax_grad = jax.jit(jax.jacrev(fn, argnums=argnums))(*xs)
                print(tree_allclose(rev_grad, jax_grad, equal_nan=True))
            
            print(edges_rev_ops, int(rev_ops[0]), edges_rev_ops == int(rev_ops[0]))
            # fwd_successes.append(int(edges_fwd_ops == int(fwd_ops[0])))
            rev_successes.append(int(edges_rev_ops == int(rev_ops[0])))
            # print(f"Success rate fwd = {sum(fwd_successes)/len(fwd_successes)*100.:.2f}%")
            print(f"Success rate rev = {sum(rev_successes)/len(rev_successes)*100.:.2f}%")
        
