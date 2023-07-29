import jax
import jax.random as jrand

from graphax.core import GraphInfo
from alphagrad.data.generator import VertexGameGenerator

key = jrand.PRNGKey(1337)
info = []
gen = VertexGameGenerator(16, info, key=key)  
print(gen(8, key).edges.shape) 
print(gen(8, key).info)

