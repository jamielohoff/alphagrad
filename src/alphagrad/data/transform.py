"""
This file contains methods that provide symmetry operations that can be
safely used on computational graphs for data augumentation.

mainly operates on numpy arrays
TODO add documentation
"""
from typing import Sequence
import numpy as np

from graphax.core import GraphInfo
from graphax.examples import construct_Helmholtz


edges, info = construct_Helmholtz()
edges = np.stack([np.array(edges).flatten() for _ in range(info.num_intermediates)])
batch = np.stack([edges for _ in range(8)])


def swap_rows(edges: np.ndarray, i: int, j: int) -> np.ndarray:
    edges[:, [i, j]] = edges[:, [j, i]]
    return edges


def swap_cols(edges: np.ndarray, i: int, j: int) -> np.ndarray:
    edges[:, :, [i, j]] = edges[:, :, [j, i]]
    return edges


def swap_intermediates(edges: np.ndarray, i: int, j: int, info:GraphInfo) -> np.ndarray:
    num_i = info.num_inputs
    edges = swap_rows(edges, i+num_i-1, j+num_i-1)
    return swap_cols(edges, i-1, j-1)


def swap_inputs(edges: np.ndarray, i: int, j: int, info: GraphInfo) -> np.ndarray:
    num_i = info.num_inputs
    return swap_rows(edges, i+num_i-1, j+num_i-1)


def swap_outputs(edges: np.ndarray, i: int, j: int, info: GraphInfo) -> np.ndarray:
    return swap_cols(edges, i-1, j-1)

