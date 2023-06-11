import argparse

import jax
import jax.random as jrand

from graphax.dataset import Graph2File, get_prompt_list, LLMSampler, RandomSampler


parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, 
                    default="./samples", help="Path where files should be stored.")

parser.add_argument("--prefix", type=str, 
                    default="comp_graph_examples", help="Name prefix of the files.")

parser.add_argument("--num_samples", type=int, 
                    default=20000, help="Total number of samples.")

parser.add_argument("--samples_per_file", type=int, 
                    default=10000, help="Number of samples per file.")

parser.add_argument("--sampler_batchsize", type=int, 
                    default=32, help="Batchsize of the sampler.")

args = parser.parse_args()


API_KEY = "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"
PROMPT_LIST = get_prompt_list("./prompt_list.txt")    
key = jrand.PRNGKey(42)

sampler = RandomSampler(min_num_intermediates=8)

gen = Graph2File(sampler,
                args.path, 
                fname_prefix=args.prefix,
                num_samples=args.num_samples, 
                samples_per_file=args.samples_per_file,
                sampler_batchsize=32)

gen.generate(key=key, minval=0.025, maxval=0.5)

