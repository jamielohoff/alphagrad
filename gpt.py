import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import chex
import equinox as eqx

BATCHSIZE = 32
BLOCKSIZE = 8 # maximum context size

# open and load dataset
with open("tiny-shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)

# since we tokenize at character level, we need to create
# a mapping from characters to integers, i.e. a look-up table
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[char] for char in s] 
decode = lambda l: "".join([itos[i] for i in l])

data = jnp.array(encode(text), dtype=jnp.int32)

n = int(.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str = "train", key: chex.PRNGKey = None):
    data = train_data if split == "train" else val_data
    ix = jrand.randint(key, (BATCHSIZE,), minval=0, maxval=len(data)-BLOCKSIZE)
    x = jnp.stack([data[i:i+BLOCKSIZE] for i in ix])
    y = jnp.stack([data[i+1:i+BLOCKSIZE+1] for i in ix])
    return x, y




