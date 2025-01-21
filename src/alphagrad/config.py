import os
import yaml
from importlib import import_module


def load_config(path: str):
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def setup_experiment(task: str, path: str, prefix: str = ""):
    config = load_config(os.path.join(path, prefix+task) + ".yaml")
    
    # importing the make_function of the task and creating the graph from it
    package = "alphagrad.experiments"
    module = import_module(package)
    
    task_fn_name = "make_" + task
    make_fn = getattr(module, task_fn_name)
    graph, graph_shape, task_fn = make_fn()
    return config, graph, graph_shape, task_fn


def setup_joint_experiment(path: str):
    config = load_config(os.path.join(path, "joint") + ".yaml")
    
    # importing the make_function of the task and creating the graph from it
    package = "alphagrad.experiments"
    module = import_module(package)
    
    task_fn_names = list(config["scores"])

    graphs, graph_shapes, task_fns = [], [], []
  
    for task_fn_name in task_fn_names:
        print(task_fn_name)
        make_fn = getattr(module, "make_" + task_fn_name)
        graph, graph_shape, task_fn = make_fn()
        graphs.append(graph)
        graph_shapes.append(graph_shape)
        task_fns.append(task_fn)
    return config, graphs, graph_shapes, task_fns
    
