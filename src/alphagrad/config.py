import os
import yaml
from importlib import import_module


def load_config(path: str):
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def setup_experiment(task: str, path: str):
    config = load_config(os.path.join(path, task, ".yaml"))
    
    # importing the make_function of the task and creating the graph from it
    package = "experiments"
    task_fn_name = "make_" + task

    module = import_module(package)
    make_fn = getattr(module, task_fn_name)
    graph, graph_shape, task_fn = make_fn()
    return config, graph, graph_shape, task_fn
    
