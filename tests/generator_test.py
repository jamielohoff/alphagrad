import jax
import jax.random as jrand

from graphax.core import GraphInfo
from alphagrad.data.generator import VertexGameGenerator

key = jrand.PRNGKey(1337)
info = GraphInfo(num_inputs=4,
                num_intermediates=11,
                num_outputs=4,
                num_edges=0)
gen = VertexGameGenerator(16, info, key=key)  
print(gen(8, key).edges.shape) 
print(gen(8, key).info)

