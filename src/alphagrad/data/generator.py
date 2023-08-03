import argparse

import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.dataset import Graph2File, get_prompt_list, LLMSampler, RandomSampler


parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int,
                    default=123, help="Random seed for graph generation.")

parser.add_argument("--path", type=str, 
                    default="./_samples", help="Path where files should be stored.")

parser.add_argument("--prefix", type=str, 
                    default="comp_graph_examples", help="Name prefix of the files.")

parser.add_argument("--num_samples", type=int, 
                    default=16384, help="Total number of samples.")

parser.add_argument("--batchsize", type=int, 
                    default=128, help="Sampling batchsize.")

parser.add_argument("--storage_shape", type=int, nargs="+",
                    default=[20, 105, 20], help="Shape of the stored files.")

parser.add_argument("--sampling_shape", type=int, nargs="+",
                    default=[20, 105, 20], help="Shape of the generated files.")

parser.add_argument("--scalar", type=int,
                    default=0, help="Sample scalar or vector/matrix functions.")

args = parser.parse_args()


API_KEY = "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"
PROMPT_LIST = get_prompt_list("./prompt_list.txt")    
key = jrand.PRNGKey(args.seed)

sampler = RandomSampler(storage_shape=args.storage_shape,
                        min_num_intermediates=50)

gen = Graph2File(sampler,
                args.path,
                fname_prefix=args.prefix,
                num_samples=args.num_samples, 
                batchsize=args.batchsize,
                storage_shape=args.storage_shape)


if args.scalar == 0:
    gen.generate(key=key, sampling_shape=args.sampling_shape)
else:
    gen.generate(key=key,
                sampling_shape=args.sampling_shape,
                primal_p=jnp.array([1., 0., 0.]), 
                prim_p=jnp.array([.2, .8, 0., 0., 0.]))

