import os
import argparse
from functools import partial
import tqdm

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import chex
import optax
import equinox as eqx

from encoder import Encoder

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="TinyGPT_test", help="Name of the experiment.")

parser.add_argument("--gpu", type=str, 
                    default="0", help="GPU identifier.")

parser.add_argument("--seed", type=int,
                    default=1337, help="Random seed.")

parser.add_argument("--epochs", type=int, 
                    default=5000, help="Number of runs on random data.")

parser.add_argument("--batchsize", type=int, 
                    default=32, help="Learning batchsize.")

parser.add_argument("--blocksize", type=int, 
                    default=8, help="Maximum context size.")

parser.add_argument("--lr", type=float, 
                    default=1e-3, help="Learning rate.")

parser.add_argument("--heads", type=int, 
                    default=6, help="Learning rate.")

parser.add_argument("--depth", type=int, 
                    default=3, help="Learning rate.")

parser.add_argument("--embedding_size", type=int, 
                    default=128, help="Learning rate.")
 
args = parser.parse_args()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

BATCHSIZE = args.batchsize
BLOCKSIZE = args.blocksize # maximum context size
LR = args.lr
EPOCHS = args.epochs
HEADS = args.heads
EMBEDDING_SIZE = args.embedding_size
DEPTH = args.depth
MASK = jnp.tile(jnp.tril(jnp.ones((BLOCKSIZE, BLOCKSIZE)))[jnp.newaxis, :, :],(HEADS,1,1))


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


class TransformerModel(eqx.Module):
    tokenizer: eqx.nn.Embedding
    pos_enc: eqx.Module
    tf: eqx.Module
    linear: eqx.Module

    def __init__(self, 
                vocab_size,
                block_size, 
                num_layers, 
                num_heads, 
                in_dim, 
                ff_dim, 
                key) -> None:
        super().__init__()
        token_key, pos_key, lin_key, key = jrand.split(key, 4)
        self.tokenizer = eqx.nn.Embedding(vocab_size, in_dim, key=token_key)

        self.pos_enc = eqx.nn.Embedding(block_size, in_dim, key=pos_key)

        self.tf = Encoder(num_layers=num_layers,
                            num_heads=num_heads,
                            in_dim=in_dim,
                            ff_dim=ff_dim, 
                            key=key)

        linear = eqx.nn.Linear(in_features=in_dim, out_features=vocab_size, key=lin_key)
        self.linear = jax.vmap(linear)

    def __call__(self, idx, key, mask):
        t = idx.shape[-1]
        token = self.tokenizer(idx)
        pos = self.pos_enc(jnp.arange(0, t))

        x = token + pos
        x = self.tf(x, key=key, mask=mask)
        return self.linear(x)


def generate(model, idx, max_new_tokens=256, key=None):
    model = eqx.tree_inference(model, True)
    for _ in range(max_new_tokens):
        subkey, key = jrand.split(key, 2)
        context = idx[-BLOCKSIZE:]
        logits = model(context, key, None)
        logits = logits[-1, :]
        idx_next = jrand.categorical(subkey, logits, shape=(1,))
        idx = jnp.concatenate((idx, idx_next), axis=0)
    return idx


@partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
def ce_loss(model, x, target, key, mask):
    prediction = model(x, key, mask)
    return optax.softmax_cross_entropy_with_integer_labels(prediction, target)


@eqx.filter_value_and_grad
def compute_grads(model, xs, targets, keys, mask):
    return ce_loss(model, xs, targets, keys, mask).mean()


@eqx.filter_jit
def update(model, opt_state, xs, targets, keys, mask):
    loss, grads = compute_grads(model, xs, targets, keys, mask)
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optim.update(grads, opt_state, params=params)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


key = jrand.PRNGKey(args.seed)
model = TransformerModel(VOCAB_SIZE, BLOCKSIZE, DEPTH, HEADS, EMBEDDING_SIZE, 4*EMBEDDING_SIZE, key)
optim = optax.adamw(args.lr)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

pbar = tqdm.tqdm(range(EPOCHS))
for epoch in pbar:
    batch_key, model_key, key = jrand.split(key, 3)
    x, y = get_batch(key=batch_key)
    keys = jrand.split(key, BATCHSIZE)
    loss, model, opt_state = update(model, opt_state, x, y, keys, MASK)
    pbar.set_description(f"loss: {loss}")

print(decode(generate(model, jnp.array([0]), key=key).tolist()))

 