import os
import jax
import jax.numpy as jnp

from graphax.perf import plot_performance, plot_performance_jax_only
from graphax.examples import RoeFlux_1d

os.environ["XLA_FLAGS"] = "--xla_backend_optimization_level=3"

# 320 mults
order = [8, 82, 27, 66, 7, 78, 76, 13, 48, 42, 68, 86, 95, 4, 59, 28, 77, 54, 1, 
         94, 5, 58, 72, 93, 75, 31, 53, 33, 57, 90, 44, 25, 89, 88, 84, 96, 74, 
         92, 83, 91, 45, 51, 81, 80, 11, 10, 85, 43, 22, 73, 19, 71, 6, 18, 17, 
         79, 47, 50, 52, 21, 37, 38, 55, 49, 69, 35, 65, 29, 64, 16, 9, 60, 15, 
         61, 23, 87, 70, 67, 24, 46, 63, 39, 2, 62, 3, 41, 40, 32, 26, 34, 56, 
         30, 14, 98, 36, 12, 20, 100] 

mM_order = [4, 5, 8, 9, 16, 17, 25, 27, 31, 33, 38, 43, 44, 45, 69, 84, 1, 2,
            10, 13, 18, 21, 26, 28, 32, 34, 37, 39, 42, 47, 50, 53, 57, 59, 
            62, 64, 66, 67, 68, 71, 73, 75, 76, 77, 80, 81, 83, 85, 86, 87, 
            91, 92, 95, 11, 14, 19, 22, 51, 54, 58, 60, 63, 65, 72, 79, 88, 
            90, 93, 96, 3, 6, 7, 15, 29, 40, 56, 61, 74, 78, 82, 48, 89, 94, 
            23, 35, 46, 24, 70, 41, 98, 100, 12, 20, 30, 49, 52, 55, 36]


shape = (512,)
xs = [.01, .015, .015, .01, .03, .03]
xs = [jnp.ones(shape)*x for x in xs]
# xs = jax.device_put(xs, jax.devices("cpu")[0])
plot_performance_jax_only(RoeFlux_1d, xs, "./RoeFlux_1d.png", samplesize=1000)

