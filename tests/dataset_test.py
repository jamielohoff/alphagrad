from torch.utils.data import DataLoader
from graphax.dataset import GraphDataset
# from alphagrad.data.generator import make_batched_vertex_games
from graphax.dataset import read

import time

train_dataset = GraphDataset("./samples")
train_dataloader = DataLoader(train_dataset, 
                            batch_size=200, 
                            shuffle=False,
                            num_workers=8)

st = time.time()
edges, info, vertices, attn_mask = next(iter(train_dataloader))
print(time.time() - st)

# print(read("./samples/comp_graph_examples-0.hdf5", [1,2,4,5]))

