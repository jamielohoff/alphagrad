"""A collection of sharding utils for simple FSDP sharding
"""

from typing import Sequence, Union
from functools import reduce

import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

Array = jax.Array
PRNGKey = jax.Array


Axes = Union[int, Sequence[int]]
Shape = Axes
AxesNames = Union[int, Sequence[int]]


def repackage(obj):
    if not isinstance(obj, (tuple, list)):
        return (obj,)
    return obj
    

def prod(nums: Sequence[int]) -> int:
	return reduce(lambda x, y: x*y, nums)


def get_factorizations(n: int, num_factors: int) -> Sequence[Sequence[int]]:
    """Produces a list of lists where every list contains a set of `num_factors`
    integers that, when multiplied, equals `n`. Essentially factor decomposition.
    
    Example:
    For n = 8 and num_factors = 2 this function returns [[2, 4]].

    Args:
        n (int): A given positive integer which we want to decompose.
        num_factors (int): The number of factors that we want to represent `n` with.

    Returns:
        Sequence[Sequence[int]]: _description_
    """
    assert n > 0, "`n` has to be greater than 0!"
    assert num_factors > 2, "`num_factors` has to be greater than 2!"
    # actually a leetcode problem
    # returns a list of tuples `factors` that all divide a given number
    # no duplicates allowed
    # only store them in descending order

    # top-down recursion problem 
    if num_factors > 1:
        factors = []
        for i in range(2, n):
            if n % i == 0:
                fc = get_factorizations(-(-n // i), i, num_factors-1)
                factors.extend([(*f, i) for f in fc if prod((i, *f)) == n])
        return factors
    else:
        # return all possible numbers that divide `n`
        return [[j] for j in range(2, n+1) if n % j == 0]


def get_sharding_dims(
    shape: Sequence[int], 
    axes: Sequence[int], 
    num_devices: int
) -> Sequence[int]:
    """Returns

    Args:
        shape (Sequence[int]): _description_
        axes (Sequence[int]): _description_
        num_devices (int): _description_

    Returns:
        Sequence[int]: _description_
    """
    # order the axes from largest to smallest using argsort
    shape_axes = [(ax, shape[ax]) for ax in axes]
    shape_axes = sorted(shape_axes, key=lambda x: x[1], reverse=True)

    # find all decompositions of `num_devices` with exactly len(axes) factors
    if len(axes) > 1:
        sharding_candidates = get_factorizations(num_devices, len(axes))
    else:
        return (num_devices,)
    
    # then search for an eligible sharding that divides all axes
    mapping = []

    for candidate in sharding_candidates:
        divides = True
        for (ax, s), c in zip(shape_axes, candidate):
            divides *= (s % c == 0)
            mapping.append((ax, c))
        if divides:
            # select the first valid sharding that has been found
            break

    if not mapping:
        # raise error when no valid mapping could be found
        raise ValueError(f"No valid sharding could be found for shape {shape}," 
                         f"sharded along axes {axes} with number of devices"
                         f"{num_devices}")

    mapping = sorted(mapping, key=lambda x: x[0])
    return tuple([m[1] for m in mapping])


def get_sharding(
    shape: Shape,
    axes: Axes,
    axes_names: AxesNames,
    num_devices: int
) -> NamedSharding:
    """A function for generating a valid sharding for a given tensor
    
    use jax.device_put(tensor, named_sharding) to apply the sharding
    
    this sharding puts the largest factor first,

    Args:
        shape (Shape): Shape of the array we intend to shard. Can be an integer
        or a list/tuple of integers.
        axis (Axes): The axes along which we want to shard the tensor. 
        Can either be an integer or a list/tuple of integers.
        axis_names (AxesNames): Names of the respective axes for
        easier access later. Can be a single string or a list of strings.
        num_devices (int): The number of devices we want to shard the tensor across.

    Returns:
        NamedSharding: A valid NamedSharding object which can be used to shard 
        the tensor.
    """
    shape = repackage(shape)
    axes = repackage(axes)
    axes_names = repackage(axes_names)
    
    sharding = get_sharding_dims(shape, axes, num_devices)
    devices = mesh_utils.create_device_mesh(sharding)
    mesh = Mesh(devices, axis_names=axes_names)

    spec = PartitionSpec(*axes_names)
    return NamedSharding(mesh, spec)

