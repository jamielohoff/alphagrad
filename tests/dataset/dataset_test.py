from torch.utils.data import DataLoader
from alphagrad.vertexgame import GraphDataset, read

import time

train_dataset = GraphDataset("./src/alphagrad/data/samples")
train_dataloader = DataLoader(train_dataset, 
                            batch_size=8, 
                            shuffle=False,
                            num_workers=2)

st = time.time()
edges = next(iter(train_dataloader))
print(edges, edges.shape)
print(time.time() - st)

# print(read("./samples/comp_graph_examples-0.hdf5", [1,2,4,5]))

