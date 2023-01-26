import os 
import time
import functools as ft
from typing import Tuple

import numpy as np
from tqdm import tqdm

import chex
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx
import optax

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from transformer import PositionalEncoder
from encoder import Encoder

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)


def generate_square_subsequent_mask(num_heads: int, sz: int) -> chex.Array:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    mask = np.tril(np.ones((sz, sz)), k=1)[jnp.newaxis, :, :]
    return jnp.repeat(mask, num_heads, axis=0).astype(bool)


train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])


def data_process(raw_text_iter) -> chex.Array:
    """Converts raw text into a flat array."""
    data = [np.array(vocab(tokenizer(item)), dtype=np.int64) for item in raw_text_iter]
    return np.concatenate(tuple(filter(lambda t: np.size(t) > 0, data)))


# train_iter was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)


def batchify(data: chex.Array, bsz: int) -> chex.Array:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.shape[0] // bsz
    data = data[:seq_len * bsz]
    data = np.ascontiguousarray(data.reshape(bsz, seq_len).T)
    return data


batch_size = 32
eval_batch_size = 8
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)


T = 35
def get_batch(source: chex.Array, i: int) -> Tuple[chex.Array, chex.Array]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(T, len(source) - 1 - i)
    data = source[i:i+seq_len].T
    target = source[i+1:i+1+seq_len].T
    return data, target


# definition of the hyper parameters
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability


@ft.partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
def calc_loss(model, mask, batch, target, key):
    predictions = model(batch, mask=mask, key=key)
    print(predictions.shape)
    predictions = predictions.reshape(-1, ntokens)
    return optax.softmax_cross_entropy_with_integer_labels(predictions, target)#/predictions.shape[0]


@eqx.filter_value_and_grad
def loss_and_grads(model, mask, batch, target, key):
    keys = jrand.split(key, batch_size)
    loss_batch = calc_loss(model, mask, batch, target, keys)
    return loss_batch.mean()


@eqx.filter_jit
def update_fn(model, mask, optim, opt_state, batch, target, key):
    loss, grads = loss_and_grads(model, mask, batch, target, key)

    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


key = jrand.PRNGKey(42)


class SimpleEncoderModel(eqx.Module):
    in_dim: int
    pos_encoder: PositionalEncoder
    embedding: eqx.nn.Embedding
    encoder: Encoder
    linear: eqx.nn.Linear
    
    def __init__(self, 
                num_layers,  
                num_heads, 
                in_dim, 
                ff_dim, 
                dropout, *,
                key) -> None:
        super().__init__()
        self.in_dim = in_dim
        keys = jrand.split(key, 3)
        self.pos_encoder = PositionalEncoder(in_dim)
        self.embedding = eqx.nn.Embedding(ntokens, in_dim, key=keys[0])
        self.encoder = Encoder(num_layers, 
                                num_heads,
                                in_dim, 
                                ff_dim, 
                                dropout=dropout,
                                key=keys[1])
        linear = eqx.nn.Linear(in_dim, ntokens, key=keys[2])
        self.linear = jax.vmap(linear, in_axes=(0,))
        
    def __call__(self, xs: chex.Array, key: chex.PRNGKey, mask=None):
        xs = self.embedding(xs) * jnp.sqrt(self.in_dim)
        xs = self.pos_encoder(xs)
        xs = self.encoder(xs, key=key, mask=mask)
        return self.linear(xs)


model = SimpleEncoderModel(2, in_dim=emsize, num_heads=nhead, ff_dim=d_hid, dropout=.1, key=key)

lr = 5.0
epochs = 1000
schedule = optax.exponential_decay(init_value=lr, 
                                    transition_steps=train_data.shape[0] - 1,
                                    decay_rate=.95, 
                                    staircase=True)
optim = optax.chain(optax.clip_by_global_norm(.5), 
                    optax.adam(learning_rate=schedule))
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


for epoch in tqdm(range(epochs)):
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(nhead, T)
    num_batches = len(train_data) // T
    for batch, i in enumerate(range(0, train_data.shape[0] - 1, T)):
        key, model_key = jrand.split(key, 2)
        data, targets = get_batch(train_data, i)
        seq_len = data.shape[1]
        if seq_len != T:  # only on last batch
            src_mask = src_mask[:, :seq_len, :seq_len]
            
        model, opt_state, loss = update_fn(model,
                                            src_mask,
                                            optim,
                                            opt_state, 
                                            data,
                                            targets,
                                            model_key)
        total_loss += loss
        if batch % log_interval == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = jnp.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

