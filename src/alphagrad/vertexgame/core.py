""" 
GPU-friendly edge and vertex elimination procedures for Cross-Country Elimination 
that are totally JIT-compilable. For an in-depth discussion of Cross-Country 
Elimination and the methods described here see the book 
`Evaluating Derivatives` by Griewank et al., 2008,
https://doi.org/10.1137/1.9780898717761

DO NOT TOUCH!
"""

from functools import partial
from typing import Sequence, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from chex import Array


"""
Documentation of the sparsity types:
--------------------------------------

Every entry in the 3-dimensional tensor has the following meaning:
(sparsity type, Jacobian shape 1st input component == 1st component, 
                Jacobain shape 2nd input component == 2nd component,
                Jacobian shape 1st output component == 3rd component,
                Jacobian shape 2nd output component == 4th component)
Sorting:
(sparsity type, out_dim1, out_dim2, primal_dim1, primal_dim2)
Thus the current implementation can only deal with scalars vectors and matrices
and related operations. 
It is basically a adjecency matrix of the computational graph, with the 3rd 
dimension indicating the sparsity type and shape of the Jacobians associated
with the respective edge.

NOTE: No support for higher-order tensors yet!

Sparsity types explanation:
--- Mixed types ---
11: TBD
10: K(1, 3) and K(2, 4), basically a constant multiple of the Kronecker symbol
9: K(1, 4) and (2, 3)
8: K(1, 3) and (2, 4)
--- Diagonal parts ---
7: (1, 4) and (2, 3)
6: (1, 3) and (2, 4)
5: (2, 3)
4: (1, 4)
3: (2, 4)
2: (1, 3)
--- Base types ---
1: Dense Jacobian, i.e. no Kronecker symbols or diagonal matrices
0: No edge between vertices
-1: `copy` operation that keep sparsity, i.e. no Kronecker symbolsor diagonal matrices
--- Pure Kronecker symbols ---
-2: K(1, 3)
-3: K(2, 4)
-4: K(1, 4)
-5: K(2, 3)
-6: K(1, 3) and K(2, 4)
-7: K(1, 4) and K(2, 3)
--- "Conjugates" of 8,9 ---
-8: (1, 3) and K(2, 4)
-9: (1, 4) and K(2, 3)
-10: K(1, 4) and K(2, 3), basically a constant multiple of the Kronecker symbol

(1, 3) means diagonal between 1st and 3rd index
K(1, 3) means Kronecker symbol between 1st and 3rd index

==> The negative sign on sparsity entry is similar to a conjugation operation


To signify replicating dimensions, we just set the value of the respective
thing to negative it's current value.
Example: (2, 3, 4, 3, -5) has a replicating dimension in 2nd primal dimension
"""

NUM_SPARSITY_TYPES = 21
OFFSET = (NUM_SPARSITY_TYPES - 1) // 2
# Row idx is incoming edge, col idx is outgoing edge
#  Contraction map of the indices
# 1 means active, 0 means masked away
#                    out_edge -10 -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9  10 11  in_edge
CONTRACTION_MAP =  jnp.array([[[0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # -10
                               [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],  # -9
                               [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],  # -8
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -7
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -6
                               [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],  # -5
                               [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],  # -4
                               [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],  # -3
                               [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],  # -2
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -1
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 1
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 2
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 3
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 4
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 5
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 6
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 7
                               [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],  # 8
                               [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],  # 9
                               [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # 10
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], # 11
                              
#                    out_edge -10 -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9  10 11  in_edge
                              [[0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # -10
                               [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # -9
                               [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # -8
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -7
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -6
                               [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # -5
                               [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # -4
                               [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # -3
                               [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # -2
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -1
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 1
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 2
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 3
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 4
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 5
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 6
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 7
                               [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 8
                               [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 9
                               [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 10
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], # 11
                              
#                    out_edge -10 -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9  10 11  in_edge
                              [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -10
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -9
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -8
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -7
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -6
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # -5
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -4
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # -3
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -2
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -1
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 1
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 3
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 5
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], # 11
                              
#                    out_edge -10 -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9  10 11  in_edge
                              [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -10
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -9
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -8
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -7
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -6
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -5
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # -4
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -3
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # -2
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -1
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 1
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 2
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 4
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], # 11
                              
#                    out_edge -10 -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9  10 11  in_edge
                              [[0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # -10
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # -9
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # -8
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -7
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -6
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # -5
                               [1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],  # -4
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # -3
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # -2
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -1
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 1
                               [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 2
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 3
                               [1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],  # 4
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 5
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 6
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 7
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 8
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 9
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 10
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], # 11
                              
#                    out_edge -10 -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9  10 11  in_edge
                              [[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # -10
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # -9
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # -8
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -7
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -6
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # -5
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # -4
                               [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # -3
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # -2
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -1
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 1
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 2
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # 3
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 4
                               [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 5
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 6
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 7
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 8
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 9
                               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 10
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]) # 11   


#                     out_edge -10  -9  -8  -7  -6  -5  -4  -3  -2  -1   0   1   2   3   4   5   6   7   8   9  10  11 in_edge
MUL_SPARSITY_MAP = jnp.array([[ 10, -8, -9, -6, -7, -3, -2, -5, -4,-10,  0,  1,  4,  5,  2,  3,  7,  6,  9,  8,-10,  1],  # -10
                              [  8,  6,  7,  8, -9,  3, -2, -5,  4, -9,  0,  1,  5,  4,  3,  2,  7,  6, -9,  6, -9,  1],  # -9
                              [  9, -9, -8,  9, -8,  5, -4, -3,  2, -8,  0,  1,  2,  3,  4,  5,  6,  7,  6,  7, -8,  1],  # -8
                              [ 10, -8, -9, -6, -7, -3, -2, -5, -4, -7,  0,  1,  4,  5,  2,  3,  7,  6,  9,  8,-10,  1],  # -7
                              [-10, -9, -8, -7, -6, -5, -4, -3, -2, -6,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  1],  # -6
                              [ -2,  2, -5, -2, -5,  1, -2, -5,  1, -5,  0,  1,  1,  5,  2,  1,  5,  2,  5, -2, -5,  1],  # -5
                              [ -3, -3,  4, -3, -4, -3,  1,  1, -4, -4,  0,  1,  4,  1,  1,  3,  4,  3, -4,  3, -4,  1],  # -4
                              [ -4,  4, -3, -4, -3,  1, -4, -3,  1, -3,  0,  1,  1,  3,  4,  1,  3,  4,  3, -4, -3,  1],  # -3
                              [ -5, -5,  2, -5, -2, -5,  1,  1, -2, -2,  0,  1,  2,  1,  1,  5,  2,  5, -2,  5, -2,  1],  # -2
                              [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],  # -1
                              [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  #  0
                              [  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  #  1
                              [  5,  5,  2,  5,  2,  5,  4,  1,  2,  2,  0,  1,  2,  1,  1,  5,  2,  5,  2,  5,  2,  1],  #  2
                              [  4,  4,  3,  4,  3,  1,  1,  3,  1,  3,  0,  1,  1,  3,  4,  1,  3,  4,  3,  4,  3,  1],  #  3
                              [  3,  3,  4,  3,  4,  3,  1,  1,  4,  4,  0,  1,  4,  1,  1,  3,  4,  3,  4,  3,  4,  1],  #  4
                              [  2,  2,  5,  2,  5,  1,  2,  5,  1,  5,  0,  1,  1,  5,  2,  1,  5,  2,  5,  2,  5,  1],  #  5
                              [  7,  7,  6,  7,  6,  5,  4,  3,  2,  6,  0,  1,  2,  3,  4,  5,  6,  7,  6,  7,  6,  1],  #  6
                              [  6,  6,  7,  6,  7,  3,  2,  5,  4,  7,  0,  1,  4,  5,  2,  3,  7,  6,  7,  6,  7,  1],  #  7
                              [ -9,  7,  6, -9,  8, -5, -4,  3, -2,  8,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  8,  1],  #  8
                              [ -8, -9,  9, -8,  9, -3,  2,  5, -4,  9,  0,  1,  5,  4,  3,  2,  7,  6,  7,  6,  9,  1],  #  9
                              [-10, -9, -8, -7, -6, -5, -4, -3, -2, 10,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  1],  #  10
                              [  1,  1,  1,  1,  1,  1,  1,  1,  1, 11,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 11]]) #  11


# Row idx is incoming edge, col idx is outgoing edge
# Gives the resulting sparsity type if two hyperdimensional Jacobians
# are added to each other
# NOTE: this guy is almost perfectly symmetric, we could also just store the upper triangle
#                     out_edge -10  -9  -8  -7  -6  -5  -4  -3  -2  -1   0   1   2   3   4   5   6   7   8   9   10  11   in_edge
ADD_SPARSITY_MAP = jnp.array([[-10,  7,  1,  7,  1,  5,  4,  1,  1,-10,-10,  1,  1,  1,  4,  5,  1,  7,  1,  7,  1,  1],  # -10
                              [  7, -9,  1, -9,  1, -5,  4,  1,  1, -9, -9,  1,  1,  1,  4,  5,  1,  7,  1,  7,  1,  1],  # -9
                              [  1,  1, -8,  1, -8,  1,  1, -3,  2, -8, -8,  1,  2,  3,  1,  1,  6,  1,  6,  1,  6,  1],  # -8
                              [  7, -9,  1, -7,  1, -5, -4,  1,  1, -7, -7,  1,  1,  1,  4,  5,  1,  7,  1,  9,  1,  1],  # -7
                              [  1,  1, -8,  1, -6,  1,  1, -3, -2, -6, -6,  1,  2,  3,  1,  1,  6,  1,  8,  1,  6,  1],  # -6
                              [  5, -5,  1, -5,  1, -5,  1,  1,  1, -5, -5,  1,  1,  1,  1,  5,  1,  5,  1,  5,  1,  1],  # -5
                              [  4,  4,  1, -4,  1,  1, -4,  1,  1, -4, -4,  1,  1,  1,  4,  1,  1,  4,  1, -4,  1,  1],  # -4
                              [  1,  1, -3,  1, -3,  1,  1, -3,  1, -3, -3,  1,  1,  3,  1,  1,  3,  1,  3,  1,  3,  1],  # -3
                              [  1,  1,  2,  1, -2,  1,  1,  1, -2, -2, -2,  1,  2,  1,  1,  1,  2,  1, -2,  1,  2,  1],  # -2
                              [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -1,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],  # -1
                              [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],  #  0
                              [  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  #  1
                              [  1,  1,  2,  1,  2,  1,  1,  1,  2,  2,  2,  1,  2,  1,  1,  1,  2,  1,  2,  1,  2,  1],  #  2
                              [  1,  1,  3,  1,  3,  1,  1,  3,  1,  3,  3,  1,  1,  3,  1,  1,  3,  1,  3,  1,  3,  1],  #  3
                              [  4,  4,  1,  4,  1,  1,  4,  1,  1,  4,  4,  1,  1,  1,  4,  1,  1,  4,  1,  4,  1,  1],  #  4
                              [  5,  5,  1,  5,  1,  5,  1,  1,  1,  5,  5,  1,  1,  1,  1,  5,  1,  5,  1,  5,  1,  1],  #  5
                              [  1,  1,  6,  1,  6,  1,  1,  3,  2,  6,  6,  1,  2,  3,  1,  1,  6,  1,  6,  1,  6,  1],  #  6
                              [  7,  7,  1,  7,  1,  5,  4,  1,  1,  7,  7,  1,  1,  1,  4,  5,  1,  7,  1,  7,  1,  1],  #  7
                              [  1,  1,  6,  1,  8,  1,  1,  3, -2,  8,  8,  1,  2,  3,  1,  1,  6,  1,  8,  1,  1,  1],  #  8
                              [  7,  7,  1,  9,  1,  5, -4,  1,  1,  9,  9,  1,  1,  1,  4,  5,  1,  7,  1,  9,  1,  1],  #  9
                              [  1,  1,  6,  1, 10,  1,  1,  3,  2, 10, 10,  1,  2,  3,  1,  1,  6,  1,  6,  1, 10,  1],  #  10
                              [  1,  1,  1,  1,  1,  1,  1,  1,  1, 11, 11,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 11]]) #  11


Edge = Tuple[int, int]


def get_shape(edges: Array):
    num_v = edges.shape[2]
    num_i = edges.shape[1] - num_v - 1
    return num_i, num_v


def get_graph_shape(graph: Array) -> Array:
    num_i = graph.at[0, 0, 0].get()
    num_vo = graph.at[0, 0, 1].get() + graph.at[0, 0, 2].get()
    num_o = graph.at[0, 0, 2].get()
    return jnp.array([num_i, num_vo, num_o])


def get_output_mask(edges: Array):
    return edges[2, 0, :]


def get_vertex_mask(edges: Array):
    return edges[1, 0, :]


def get_elimination_order(edges: Array):
    return edges[3, 0, :]


def make_empty_edges(info: Array) -> Array:
    """
    Creates an empty matrix fo represent the connectivity of the computational graph.
    """
    num_i = info[0]
    num_v = info[1]
    return jnp.zeros((5, num_i+num_v+1, num_v), dtype=jnp.int32)


@partial(jax.vmap, in_axes=(0, 0))
def sparsity_where(in_edge, out_edge):
    # Takes care of the corner cases where there already exists an edge with a 
    # different sparsity type
    i = in_edge.astype(jnp.int32) + OFFSET
    j = out_edge.astype(jnp.int32) + OFFSET
    return ADD_SPARSITY_MAP[i, j]


@partial(jax.vmap, in_axes=(1, None))
def sparsity_fmas_map(in_edge, out_edge):
    """
    TODO add documentation here!
    """
    # Get the sparsity type of the ingoing and outgoing edge
    i = in_edge[0].astype(jnp.int32) + OFFSET
    j = out_edge[0].astype(jnp.int32) + OFFSET

    new_sparsity_type = MUL_SPARSITY_MAP[i, j]                                                              
    contraction_map = CONTRACTION_MAP[:, i, j]

    factors = jnp.concatenate((out_edge[1:3], jnp.abs(out_edge[3:]), in_edge[3:]))
    masked_factors = lax.cond(jnp.sum(contraction_map) > 0,
                                lambda a: jnp.where(contraction_map > 0, a, 1), # 1 enables factors, 0 disables factors
                                lambda a: jnp.zeros_like(a, dtype=jnp.int32),
                                factors)
    # Here we deal with replicating dimensions
    masked_factors = jnp.where(masked_factors >= 0, masked_factors, 1)
    
    fmas = jnp.prod(masked_factors)
    fmas = lax.select(jnp.logical_and(jnp.abs(i) == 10+OFFSET, jnp.abs(j) == 10+OFFSET), 1, fmas)
    # jax.debug.print("{out_edge} : {in_edge} : {nst} : {fmas}", out_edge=out_edge, in_edge=in_edge, nst=new_sparsity_type, fmas=fmas)
    return new_sparsity_type, fmas


def get_fmas_jacprod(all_edges, fmas, in_edges, out_edges, nonzero, vertex, num_i):
    # Define aliases
    in_edges_primals = in_edges[3:, :]
    in_edges_outs = in_edges[1:3, :]
    
    out_edges_primals = out_edges[3:, :]
    out_edges_outs = out_edges[1:3, :]
        
    # Calculate fmas
    # Select only the edges that are connected to the vertex through code below
    new_sparsity, _fmas = sparsity_fmas_map(in_edges, out_edges[:, vertex+num_i-1])
    fmas = jnp.sum(_fmas)
    
    # Calculate resulting sparsity type
    new_sparsity = sparsity_where(out_edges[0, :], new_sparsity)
    new_sparsity = jnp.broadcast_to(new_sparsity, (1, *new_sparsity.shape))

    # In shape new edges
    new_edges_ins = jnp.where(in_edges_primals[1] != 0, in_edges_primals, out_edges_primals)
    
    # Out shape new edges
    new_edges_outs = jnp.broadcast_to(out_edges_outs[:, vertex+num_i-1, jnp.newaxis], out_edges_outs.shape)
    new_edges_outs = jnp.where(in_edges_outs[1] != 0, new_edges_outs, out_edges_outs)
    
    # Assemble new edges
    new_edges = jnp.concatenate((new_sparsity, new_edges_outs, new_edges_ins), axis=0)
        
    # Set the Jacobian adjacency matrix
    all_edges = lax.dynamic_update_index_in_dim(all_edges, new_edges, nonzero, -1)
            
    return all_edges, fmas


def vertex_eliminate(vertex: int, graph: Array) -> Tuple[Array, float]:
    """
    Fully JIT-compilable function that implements the vertex-elimination procedure. 
    Vertex elimination means that we front-eliminate all incoming edges and 
    back-eliminate all outgoing edges of a given vertex. However, the implementation
    here does not make use of the function above to be more efficient.

    Arguments:
        vertex (int): Vertex we want to eliminate.
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_i, num_v = get_shape(graph)
    edges = graph[:, 1:, :]
    in_edges = edges[:, :, vertex-1]
    def update_edges_fn(carry, nonzero):
        edges, fmas = carry
        # Get the index of the operation and the 
        out_edges = edges[:, :, nonzero]
        # Calculate the fma operations and the new shapes of the Jacobians for 
        # the respective and update the vertex
        edges, _fmas = lax.cond(nonzero > -10, 
                                lambda e, f, ie, oe, nz, v: get_fmas_jacprod(e, f, ie, oe, nz, v, num_i), 
                                lambda e, f, ie, oe, nz, v: (e, 0), 
                                edges, fmas, in_edges, out_edges, nonzero, vertex)
        fmas += _fmas        
        carry = (edges, fmas)
        return carry, None
    
    nonzeros = jnp.nonzero(edges[0, num_i+vertex-1, :], size=num_v, fill_value=-10)[0].T
    output, _ = lax.scan(update_edges_fn, (edges, 0), nonzeros)
    new_edges, fmas = output
    
    # Delete old edges
    new_edges = new_edges.at[:, num_i+vertex-1, :].set(0)
    new_edges = new_edges.at[:, :, vertex-1].set(0)

    graph = graph.at[1, 0, vertex-1].set(1)
    graph = graph.at[:, 1:, :].set(new_edges)
    return graph, fmas


def cross_country(order: Sequence[int], edges: Array) -> Array:
    """
    Fully JIT-compilable function that implements cross-country elimination 
    according to the given order.

    Arguments:
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    def cc_fn(carry, vertex):
        _edges, fmas = carry
        not_masked = jnp.logical_not(_edges.at[1, 0, vertex-1].get() > 0)
                
        _edges, _fmas = lax.cond(not_masked,
                                lambda e: vertex_eliminate(vertex, e),
                                lambda e: (e, 0),
                               _edges)
        fmas += _fmas
        # jax.debug.print("{v}:{fmas}", v=vertex, fmas=_fmas)
        carry = (_edges, fmas)
        return carry, _fmas
    vertices = jnp.array(order)
    output, _ = lax.scan(cc_fn, (edges, 0), vertices)
    return output


def forward(edges: Array):
    """
    Fully JIT-compilable function that implements forward-mode AD by 
    eliminating the vertices in sequential order 1,2,3,...,n-1,n and ignores
    the ones that are given by vertex_mask, because these are typically the 
    output vertices.

    Arguments:
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_i, num_vo = get_shape(edges)
    order = jnp.arange(1, num_vo+1)
    return cross_country(order, edges)


def reverse(edges: Array):
    """
    Fully JIT-compilable function that implements reverse-mode AD by 
    eliminating the vertices in sequential order 1,2,3,...,n-1,n and ignores
    the ones that are given by vertex_mask, because these are typically the 
    output vertices.

    Arguments:
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_i, num_vo = get_shape(edges)
    order = jnp.arange(1, num_vo+1)[::-1]
    return cross_country(order, edges)

