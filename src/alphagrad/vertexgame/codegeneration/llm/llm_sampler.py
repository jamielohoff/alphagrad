from typing import Sequence, Tuple
import openai
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.random as jrand

from chex import Array, PRNGKey

from ...utils import sparsify
from ..sampler import ComputationalGraphSampler
from ...interpreter.from_jaxpr import make_graph
from ...transforms import safe_preeliminations, compress, embed, clean


# TODO refactor code such that we do no longer need the global variable
jaxpr = ""

class LLMSampler(ComputationalGraphSampler):
    """
    Class that implements a sampling function using ChatGPT to create realistic
    examples of computational graphs.
    
    Returns jaxpr objects or string defining the function
    """
    api_key: str
    prompt_list: Sequence[Tuple[str, str]]
    sleep_timer: int
    debug: bool
    
    def __init__(self, 
                api_key: str, 
                prompt_list: Sequence[Tuple[str, str]],
                *args,
                sleep_timer: int = 12,
                debug: bool = False,
                **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.api_key = api_key
        self.prompt_list = prompt_list
        self.sleep_timer = sleep_timer
        self.debug = debug
    
    def sample(self, 
               num_samples: int = 1, 
               key: PRNGKey = None,
               **kwargs) -> Sequence[tuple[str, Array]]:
        openai.api_key = self.api_key
        idx = jrand.randint(key, (), 0, len(self.prompt_list)-1)
        message, make_jaxpr = self.prompt_list[idx]
        make_jaxpr = "\n" + make_jaxpr

        # Define prompt
        messages = [{"role": "user", "content": message}]
        samples = []
        
        pbar = tqdm(total=num_samples)
        while len(samples) < num_samples:
            sample_size = num_samples - len(samples)
            
            # Use the API to generate a response
            responses = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    messages=messages,
                                                    n=sample_size,
                                                    stop=None, 
                                                    **kwargs)
            
            for response in responses.choices:
                code = response.message.content
                lines = code.split("\n")
                clean_code = []
                indicator = False
                for line in lines:
                    if "import" in line:
                        indicator = True
                    if indicator:
                        clean_code.append(line)
                    if "return" in line:
                        indicator = False
                
                if len(clean_code) == 0: continue

                code = "\n".join(clean_code)
                code += make_jaxpr
                
                try:
                    print(code)
                    exec(code, globals())
                    global jaxpr
                    edges = make_graph(jaxpr)
                    del jaxpr
                    
                    if self.debug:
                        print(code, edges)

                    edges = clean(edges)
                    edges = safe_preeliminations(edges)
                    
                    large_enough = edges.at[0, 0, 1].get() >= self.min_num_intermediates
                    if large_enough:
                        edges = compress(edges)
                        shape = edges.at[0, 0, 0:3].get()
                        
                        edges = embed(key, edges, jnp.array(self.storage_shape))
                        header, sparse_edges = sparsify(edges)
                        
                        samples.append((code, header, sparse_edges))
                        print(f"{len(samples)}/{num_samples} samples with shape {shape}.")
                        pbar.update(1)
                    else: 
                        print("Sample of shape", edges.at[0, 0, 0:3].get().tolist(), "rejected!")
                        continue
                except Exception as e:
                    print(e)
                    continue
            
        return samples

