from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jrand

from chex import PRNGKey

from graphax.examples import (f, RoeFlux_1d,
                                RobotArm_6DOF,
                                Encoder,
                                EncoderDecoder,
                                Lighthouse)


from ..utils import create, write, sparsify
from ..interpreter import make_graph
from ..transforms import safe_preeliminations, compress, embed

def make_task_dataset(key: PRNGKey, fname: str, info: Sequence[int] =[20, 105, 20]) -> None:
    """
    Creates a benchmark dataset that

    Args:
        info (GraphInfo): _description_

    Returns:
        _type_: _description_
    """
    keys = jrand.split(key, 8)
    samples = []
    create(fname, 8, info)

    # We use the field that is usually reserved for source code to store the names
    
    # Number of FMAs after safe preeliminations
    # fwd: 16, rev: 11, cc: 13
    print("Making Lighthouse...")
    edges = make_graph(Lighthouse, 1., 1., 1., 1.)
    edges = safe_preeliminations(edges)
    edges = compress(edges)
    edges = embed(keys[0], edges, info)
    header, sparse_edges = sparsify(edges)
    samples.append(("Lighthouse", header, sparse_edges))
    
    # Number of FMAs after safe preeliminations
    # fwd: 19, rev: 13, cc: 15
    print("Making f...")
    a = jnp.ones(4)
    b = jnp.ones((2, 3))
    c = jnp.ones((4, 4))
    d = jnp.ones((3, 3))
    e = jnp.ones((4, 1))

    edges = make_graph(f, a, b, c, d, e)
    edges = safe_preeliminations(edges)
    edges = compress(edges)
    edges = embed(keys[1], edges, info)
    header, sparse_edges = sparsify(edges)
    samples.append(("f", header, sparse_edges))
            
    # Number of FMAs after safe preeliminations
    # fwd: 69824, rev: n/v, cc: 23552
    print("Making transformer encoder...")
    x = jnp.ones((4, 4))
    y = jnp.ones((2, 4))
    
    WQ1 = jnp.ones((4, 4))
    WK1 = jnp.ones((4, 4))
    WV1 = jnp.ones((4, 4))
    
    WQ2 = jnp.ones((4, 4))
    WK2 = jnp.ones((4, 4))
    WV2 = jnp.ones((4, 4))

    W1 = jnp.ones((4, 4))
    b1 = jnp.ones(4)

    W2 = jnp.ones((2, 4))
    b2 = jnp.ones((2, 1))
    
    gamma = jnp.ones(2)
    beta = jnp.zeros(2)

    xs = (x, y, WQ1, WQ2, WK1, WK2, WV1, WV2, W1, W2, b1, b2, gamma, beta)
    edges = make_graph(Encoder, *xs)
    edges = safe_preeliminations(edges)
    edges = compress(edges)
    edges = embed(keys[4], edges, info)
    header, sparse_edges = sparsify(edges)
    samples.append(("Transformer Encoder", header, sparse_edges))
    
    # Number of FMAs after safe preeliminations
    # fwd: 81968, rev: 15584, cc: 32400
    print("Making transformer encoder-decoder...")
    x = jnp.ones((4, 4))
    y = jnp.ones((2, 4))
    
    WQ1 = jnp.ones((4, 4))
    WK1 = jnp.ones((4, 4))
    WV1 = jnp.ones((4, 4))
    
    WQ2 = jnp.ones((4, 4))
    WK2 = jnp.ones((4, 4))
    WV2 = jnp.ones((4, 4))
    
    WQ3 = jnp.ones((4, 4))
    WK3 = jnp.ones((4, 4))
    WV3 = jnp.ones((4, 4))

    W1 = jnp.ones((4, 4))
    b1 = jnp.ones(4)

    W2 = jnp.ones((2, 4))
    b2 = jnp.ones((2, 1))
    
    gamma = jnp.ones(3)
    beta = jnp.zeros(3) 

    xs = (x, y, WQ1, WQ2, WQ3, WK1, WK2, WK3, WV1, WV2, WV3, W1, W2, b1, b2, gamma, beta)
    edges = make_graph(EncoderDecoder, *xs)
    edges = safe_preeliminations(edges)
    edges = compress(edges)
    edges = embed(keys[5], edges, info)
    header, sparse_edges = sparsify(edges)
    samples.append(("Transformer Encoder-Decoder", header, sparse_edges))
    
    # Number of FMAs after safe preeliminations
    # fwd: 384, rev: 217, cc: 277
    print("Making Roe Flux...")
    xs = [1.]*6
    edges = make_graph(RoeFlux_1d, *xs)
    edges = safe_preeliminations(edges)
    edges = compress(edges)
    edges = embed(keys[6], edges, info)
    header, sparse_edges = sparsify(edges)
    samples.append(("1D Roe Flux", header, sparse_edges))
    
    #  Number of FMAs after safe preeliminations
    # fwd: 329, rev: 177, cc: 181
    print("Making 6DOF...")
    xs = [1.]*6
    edges = make_graph(RobotArm_6DOF, *xs)
    edges = safe_preeliminations(edges)
    edges = compress(edges)
    edges = embed(keys[7], edges, info)
    header, sparse_edges = sparsify(edges)
    samples.append(("Differential Kinematics 6DOF Robot", header, sparse_edges))
    
    write(fname, samples)    

    