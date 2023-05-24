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

import torchvision.transforms as tf
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from alphagrad.transformer._encoder import Encoder

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="CIFAR10_ViT_test", help="Name of the experiment.")

parser.add_argument("--gpu", type=str, 
                    default="0", help="GPU identifier.")

parser.add_argument("--seed", type=int,
                    default=1337, help="Random seed.")

parser.add_argument("--epochs", type=int, 
                    default=50, help="Number of runs on random data.")

parser.add_argument("--batchsize", type=int, 
                    default=128, help="Learning batchsize.")

parser.add_argument("--lr", type=float, 
                    default=1e-3, help="Learning rate.")

parser.add_argument("--heads", type=int, 
                    default=6, help="Number of attention heads.")

parser.add_argument("--depth", type=int, 
                    default=6, help="Number of attention blocks.")

parser.add_argument("--embedding_size", type=int, 
                    default=128, help="size of the embedding.")
 
args = parser.parse_args()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

PATH = "./cifar10"
IMAGESIZE = 32*32
PATCH_DIM = 8*8
CHANNELS = 3
NUM_PATCHES = IMAGESIZE // PATCH_DIM
NUM_CLASSES = 10
PATCHSIZE = PATCH_DIM * CHANNELS
BATCHSIZE = args.batchsize
BATCHSIZE_TEST = 4*args.batchsize
LR = args.lr
EPOCHS = args.epochs
HEADS = args.heads
EMBEDDING_SIZE = args.embedding_size
DEPTH = args.depth

transform = tf.Compose([tf.ToTensor(), 
                        tf.Normalize((0.1307,), (0.3081,)),
                        tf.RandomHorizontalFlip(.5)])

train_set = CIFAR10(PATH, 
                train=True, 
                download=True,
                transform=transform)

train_loader = DataLoader(train_set, 
                        batch_size=BATCHSIZE, 
                        shuffle=True, 
                        num_workers=8,
                        drop_last=True)

test_set = CIFAR10(PATH, 
                train=False, 
                download=True,
                transform=transform)

test_loader = DataLoader(test_set, 
                        batch_size=BATCHSIZE_TEST, 
                        drop_last=True,
                        shuffle=True)


# TODO clarify which positional encoding is the best
class TransformerModel(eqx.Module):
    patch_embedding: eqx.nn.Linear
    pos_enc: eqx.Module
    tf: eqx.Module
    linear: eqx.nn.Linear

    def __init__(self, 
                patch_dim,
                num_classes,
                num_patches, 
                num_layers, 
                num_heads, 
                embed_dim, 
                ff_dim, 
                key) -> None:
        super().__init__()
        embed_key, pos_key, lin_key, key = jrand.split(key, 4)

        patch_embedding = eqx.nn.Linear(patch_dim, embed_dim, key=embed_key)
        self.patch_embedding = jax.vmap(patch_embedding)

        self.pos_enc = eqx.nn.Embedding(num_patches, embed_dim, key=pos_key)

        self.tf = Encoder(num_layers=num_layers,
                            num_heads=num_heads,
                            in_dim=embed_dim,
                            ff_dim=ff_dim, 
                            key=key)

        self.linear = eqx.nn.Linear(num_patches*embed_dim, num_classes, key=lin_key)

    def __call__(self, x, key):
        t = x.shape[0]
        embedding = self.patch_embedding(x)
        pos = self.pos_enc(jnp.arange(0, t))
        x = embedding + pos
        x = self.tf(x, key=key, mask=None)
        x = x.reshape(-1)
        return self.linear(x)


@partial(jax.vmap, in_axes=(None, 0, 0, 0))
def ce_loss(model, x, target, key):
    prediction = model(x, key)
    return optax.softmax_cross_entropy_with_integer_labels(prediction, target)


@eqx.filter_value_and_grad
def compute_grads(model, xs, targets, keys):
    return ce_loss(model, xs, targets, keys).mean()


@eqx.filter_jit
def update(model, opt_state, xs, targets, keys):
    loss, grads = compute_grads(model, xs, targets, keys)
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optim.update(grads, opt_state, params=params)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def evaluate(model, key, loader):
    model = eqx.tree_inference(model, True)
    vmap_model = jax.vmap(model)
    accs = []
    for inputs, targets in loader:
        batchsize = len(targets)
        subkey, key = jrand.split(key, 2)
        keys = jrand.split(subkey, batchsize)
        
        x = inputs.numpy()
        x = x.transpose(0, 2, 3, 1).reshape(batchsize, NUM_PATCHES, -1)
        y = targets.numpy()
        
        predictions = eqx.filter_jit(vmap_model)(x, keys)
        pred = jnp.argmax(predictions, axis=1)
        
        acc = jnp.mean(pred == y)
        accs.append(acc)
    return jnp.mean(jnp.array(accs))


key = jrand.PRNGKey(args.seed)
model = TransformerModel(PATCHSIZE, NUM_CLASSES, NUM_PATCHES, DEPTH, HEADS, EMBEDDING_SIZE, 4*EMBEDDING_SIZE, key)
optim = optax.adamw(args.lr)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

pbar = tqdm.tqdm(range(EPOCHS))
for epoch in pbar:
    for inputs, targets in train_loader:
        model_key, key = jrand.split(key, 2)
        keys = jrand.split(key, BATCHSIZE)
        
        x = inputs.numpy()
        x = x.transpose(0, 2, 3, 1).reshape(BATCHSIZE, NUM_PATCHES, -1)
        y = targets.numpy()
            
        loss, model, opt_state = update(model, opt_state, x, y, keys)
        pbar.set_description(f"loss: {loss}")
        
    train_acc = evaluate(model, key, train_loader)
    print("Train accuracy:", train_acc)
    val_acc = evaluate(model, key, test_loader)
    print("Validation accuracy:", val_acc)

