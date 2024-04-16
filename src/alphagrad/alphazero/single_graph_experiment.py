import os
import time
from functools import partial, reduce
from typing import Callable, Dict, NamedTuple, Sequence, Tuple

from torch.utils.data import DataLoader

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

from jax.sharding import PositionalSharding
from jax.experimental import mesh_utils

import flashbax as fbx
import optax
import equinox as eqx
from chex import Array, PRNGKey

import wandb
from loguru import logger
from tqdm import tqdm

from ..vertexgame import step, GraphDataset

from ..utils import (A0_loss,
                    get_masked_logits,
                    make_init_state,
                    postprocess_data)

from .environment_interaction import (make_recurrent_fn, 
                                    make_environment_interaction,
                                    make_environment_interaction_cpu)
from .evaluate import (evaluate_tasks, 
                        evaluate_benchmark,
                        make_Markowitz_reference_values)


class HyperParameters(NamedTuple):
    seed: PRNGKey
    num_epochs: int
    batchsize: int
    learning_rate: float
    
    envs_per_gpu: int
    num_simulations: int
    gumbel_scale: float
    replay_buffer_size: int
    
    value_weight: float
    L2_weight: float
    entropy_weight: float
    
    task_freq: int
    chckpt_freq: int
    bench_freq: int
    

DEFAULT_HYPERPARAMETERS = HyperParameters(seed=1234,
                                        num_epochs=1000,
                                        batchsize=512,
                                        learning_rate=1e-3,
                                        num_simulations=5,
                                        envs_per_gpu=1,
                                        replay_buffer_size=10000,
                                        gumbel_scale=.05,
                                        value_weight=1.,
                                        L2_weight=0.,
                                        entropy_weight=0.,
                                        task_freq=5,
                                        chckpt_freq=500,
                                        bench_freq=100)
        
    
# Helper functions
select_first = lambda x: x[0] if isinstance(x, jax.Array) else x
parallel_mean = lambda x: lax.pmean(x, "num_devices")
select_params = lambda x, y: y if eqx.is_inexact_array(x) else x


# Filtering low-value samples, TODO define low-value samples
def fill_buffer(replay_buffer, buffer_state, samples):
    def loop_fn(buffer_state, sample):
        # new_buffer_state = lax.cond(sample[-1] > 0,
        #                             lambda bs: replay_buffer.add(bs, sample),
        #                             lambda bs: bs,
        #                             buffer_state)
        new_buffer_state = replay_buffer.add(buffer_state, sample)
        return new_buffer_state, None
    updated_buffer_state, _ = lax.scan(loop_fn, buffer_state, samples)
    return updated_buffer_state

@partial(jax.jit, static_argnums=(1, 2))
def replicate_graph(graph: Array, num_devices: int, batchsize: int) -> Array:
    return jnp.tile(graph[jnp.newaxis, jnp.newaxis, ...], (num_devices, batchsize, 1, 1, 1))


def tree_to_sharding(pytree, devices):
    sharding = PositionalSharding(devices)
    filtered_pytree = eqx.filter(pytree, eqx.is_array)
    def dev_put(x):
        ones = [1]*len(x.shape)
        x = jnp.tile(x[jnp.newaxis, ...], (len(devices), *ones))
        return jax.device_put(x, sharding.reshape(-1, *ones))
    dev_pytree = jtu.tree_map(lambda x: dev_put(x), filtered_pytree) # jax.device_put(filtered_pytree, sharding.replicate(axis=0))
    return jax.tree_map(lambda x, y: y if eqx.is_array(x) else x, pytree, dev_pytree)


def tree_to_device(pytree, device):
    filtered_pytree = eqx.filter(pytree, eqx.is_inexact_array)
    dev_pytree = jax.tree_map(lambda x: jax.device_put(x, device), filtered_pytree)
    return jax.tree_map(lambda x, y: y if eqx.is_inexact_array(x) else x, pytree, dev_pytree)


class Paths(NamedTuple):
    task_file_path: str
    benchmark_file_path: str
    chckpt_path: str
    weights_path: str


class SingleGraphExperiment:
    name: str
    graph: Array
    paths: Paths
    use_wandb: bool
    
    num_inputs: int
    num_outputs: int
    num_actions: int
    state_shape: Tuple[int]
    split_idxs: Tuple[int]
    
    model: eqx.Module
    optim: optax.GradientTransformation
    hyperparameters: HyperParameters
    
    pid: int
    num_nodes: int
    coordinator_address: str
    devices: Sequence[jax.Device]
    cpu_devices: Sequence[jax.Device]
    
    task_dataloader: DataLoader
    benchmark_dataloader: DataLoader
    
    tree_search: Callable
    train_fn: Callable
    eval_env_interaction: Callable
    replay_buffer: fbx.ItemBuffer
    item_prototype: Array
    
    best_performance: Dict
    reference_values: Dict
    
    buffer_states: Array
    opt_states: Array
    
    def __init__(self, 
                name: str, 
                model: eqx.Module,
                graph: Array,
                paths: Paths,
                devices: Sequence[jax.Device],
                cpu_devices: Sequence[jax.Device],
                hyperparameters: HyperParameters = DEFAULT_HYPERPARAMETERS,
                num_nodes: int = 1,
                coordinator_address: str = None,
                pid: int = 0,
                num_inputs: int = 20,
                num_outputs: int = 20,
                num_actions: int = 105,
                use_wandb: bool = False) -> None:
        
        # General configuration of the experiment
        self.name = name
        self.hyperparameters = hyperparameters
        self.graph = graph
        self.use_wandb = use_wandb
        self.paths = paths
        
        # Multi-node configuration
        self.pid = pid
        self.devices = devices
        self.cpu_devices = cpu_devices
        self.num_nodes = num_nodes
        self.coordinator_address = coordinator_address
        
        if coordinator_address is not None and num_nodes > 1:
            logger.warning("Starting multi-node experiment...")
            jax.distributed.initialize(coordinator_address=coordinator_address,
								        num_processes=num_nodes,
								        process_id=pid)
            self.devices = jax.local_devices()
            
        # Graph information
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_actions = num_actions
        
        # Load model from checkpoint if available
        if paths.weights_path is not None:
            self.model = eqx.tree_deserialise_leaves(paths.weights_path, model)
        else:
            self.model = model

        # Initialize optimizer
        self.optim = optax.adam(learning_rate=hyperparameters.learning_rate)                     
        
        # Needed to reassemble data
        self.state_shape = (5, num_inputs+num_actions+1, num_actions)
        state_idx = jnp.prod(jnp.array(self.state_shape))
        policy_idx = state_idx + num_actions
        reward_idx = policy_idx + 1
        self.split_idxs = (state_idx, policy_idx, reward_idx)
        
        # Initialize replay buffer
        size = hyperparameters.replay_buffer_size
        self.replay_buffer = fbx.make_item_buffer(max_length=size, min_length=1, sample_batch_size=hyperparameters.batchsize//len(cpu_devices))
        sharding = PositionalSharding(self.cpu_devices)
        item_prototype = jnp.zeros((len(self.cpu_devices), reward_idx+1))
        self.item_prototype = jax.device_put(item_prototype, sharding.reshape(len(self.cpu_devices), 1))
        print("item prototype sharding")
        jax.debug.visualize_array_sharding(self.item_prototype)
        
        # Initialization of a parallel replay buffer on multiple devices
        self.buffer_states = self.replay_buffer.init(self.item_prototype)
        
        # Initialization of optimizer state on multiple devices
        params = eqx.filter(self.model, eqx.is_inexact_array)
        optim_init = lambda params, g: self.optim.init(params)
        self.opt_states = optim_init(params, jnp.zeros(1))
        
        # Initialize metrics to track
        self.best_performance = {}
        self.reference_values = []
        
        self._init_dataloaders()
        self._init_tree_search()
        self._init_train_fn()
        
        # Initialize wandb
        if use_wandb and pid == 0:
            wandb.login(key="local-f6fac6ab04ebeaa9cc3f9d44207adbb1745fe4a2", 
                        host="https://wandb.fz-juelich.de")
            wandb.init(entity="lohoff", project="AlphaGrad")
            wandb.run.name = name
            wandb.config = hyperparameters._asdict()
        
    
    def run(self) -> None:
        key = jrand.PRNGKey(self.hyperparameters.seed)
        # self._init_metrics(key)
        
        pbar = tqdm(range(self.hyperparameters.num_epochs))
        for epoch in pbar:
            start_time = time.time()
            train_key, key = jrand.split(key, 2)
            batch = jnp.tile(self.graph, (self.hyperparameters.envs_per_gpu, 1, 1, 1))
            sst = time.time()
            losses, auxs, models, opt_states = self.train_fn(self.model, batch, self.opt_states, train_key)	
            print("train time", time.time() - sst)
            
            loss = losses[0]
            aux = auxs[0]
            # TODO also do not tree_map this?
            self.model = jtu.tree_map(select_first, models, is_leaf=eqx.is_inexact_array)
            self.opt_states = jtu.tree_map(select_first, opt_states, is_leaf=eqx.is_inexact_array)
            
            # self.log_wandb(loss, aux)
            # self.checkpoint_model(counter)
            # self.eval_tasks(counter, key)
            # self.eval_benchmark(counter, key)
            st = time.time()
            pbar.set_description(f"loss: {loss}")
            print("logging time", time.time() - st)
            # counter += 1
            logger.info(f"Training time {time.time() - start_time}")
                    
                        
    def eval_tasks(self, epoch: int, key: PRNGKey) -> None:
        if epoch % self.hyperparameters.task_freq == 0 and self.pid == 0:
            st = time.time()
            names, fmas, orders = evaluate_tasks(self.model, 
                                                self.task_dataloader, 
                                                self.eval_env_interaction, 
                                                key)
            self._update_best_scores(names, fmas, orders)
            logger.info(f"Evaluation time {time.time() - st}")

            if self.use_wandb and self.pid == 0:	
                num_computations = {name:-int(nops) for name, nops in zip(names, fmas)}
                wandb.log(num_computations)

            # TODO make this work again
            # if (jnp.array(fmas) >= jnp.array(best_rews)).all():
            #     logger.info("Saving model...")
            #     best_rews = fmas
            #     self._checkpoint()
    
    
    def eval_benchmark(self, epoch: int, key: PRNGKey) -> None:
        if epoch % self.hyperparameters.bench_freq == 0 and self.pid == 0:
            st = time.time()
            performance_delta = evaluate_benchmark(self.model, 
                                                    self.benchmark_dataloader, 
                                                    self.reference_values, 
                                                    jax.local_devices(), 
                                                    self.eval_env_interaction, 
                                                    key)
            logger.info(f"Benchmarking time {time.time() - st}")
            if self.use_wandb and self.pid == 0:
                wandb.log({"Performance vs Markowitz": float(performance_delta)})
    
    
    def checkpoint_model(self, epoch: int) -> None:
        if epoch % self.hyperparameters.chckpt_freq == 0 and self.pid == 0:
            self._checkpoint()
                    
        
    def log_wandb(self, loss: Array, aux: Array) -> None:
        if self.use_wandb and self.pid == 0:
            wandb.log({"loss": loss.tolist(),
                        "policy_loss": aux[0].tolist(),
                        "value_loss": aux[1].tolist(),
                        "L2_reg": aux[2].tolist(),
                        "entropy_reg": aux[3].tolist()})
            
            
    def _checkpoint(self) -> None:
        logger.info("Checkpointing model...")
        path = os.path.join(self.paths.chckpt_path, self.name + "_chckpt.eqx")
        eqx.tree_serialise_leaves(path, self.model)
    
    
    def _init_dataloaders(self) -> None:
        # Loading of the vertex dataset
        task_graph_dataset = GraphDataset(self.paths.task_file_path, 
                                        include_code=True)
        benchmark_graph_dataset = GraphDataset(self.paths.benchmark_file_path)

        # Initialize data loaders
        self.task_dataloader = DataLoader(task_graph_dataset, 
                                            batch_size=8, 
                                            shuffle=False, 
                                            num_workers=1)
        self.benchmark_dataloader = DataLoader(benchmark_graph_dataset, 
                                               batch_size=100, 
                                               shuffle=False, 
                                               num_workers=4)
        
        
    def _init_tree_search(self) -> None:
        # Initialize environment interaction loops for tree search etc.
        # Setup of the necessary functions for the tree search
        num_actions = self.num_actions
        gumbel_scale = self.hyperparameters.gumbel_scale
        num_simulations = self.hyperparameters.num_simulations
        recurrent_fn = make_recurrent_fn(self.model, step, get_masked_logits)

        env_interaction = make_environment_interaction_cpu(num_actions, 
                                                        gumbel_scale,
                                                        num_simulations,
                                                        recurrent_fn,
                                                        step,
                                                        get_masked_logits)
        
        def tree_search(batch, model, key) -> Array:
            init_carry = make_init_state(batch, key)
            data = env_interaction(model, init_carry)
            return postprocess_data(data)
        
        self.tree_search = tree_search
        self.eval_env_interaction = make_environment_interaction_cpu(num_actions, 
                                                                    gumbel_scale,
                                                                    num_simulations,
                                                                    recurrent_fn,
                                                                    step,
                                                                    get_masked_logits)
        
        
    def _init_train_fn(self) -> None:
        num_actions = self.num_actions
        batchsize_per_gpu = self.hyperparameters.batchsize//len(self.cpu_devices)
        
        value_weight = self.hyperparameters.value_weight
        L2_weight = self.hyperparameters.L2_weight
        entropy_weight = self.hyperparameters.entropy_weight
        
        fill_buf = partial(fill_buffer, self.replay_buffer)
        sample_fn = self.replay_buffer.sample

        import time
        cpu_sharding = PositionalSharding(self.cpu_devices)
        
        @partial(jax.jit, donate_argnums=1)
        def fill_and_sample(data, buffer_states, key):
            # Fill replay buffer
            data = data.reshape(len(self.cpu_devices), -1, data.shape[-1])
            data = data.transpose(1, 0, 2)
            buffer_states = fill_buf(buffer_states, data)
            # Sample from replay buffer
            # TODO implement parallel sampling
            samples = sample_fn(buffer_states, key)
            samples = samples.experience.transpose(1, 0, 2)
            return samples, buffer_states
        
        def MCTS_tree_search_fn(model, batch, key):
            sample_key, key = jrand.split(key, 2)
            keys = jrand.split(key, len(self.devices))

            # Do a MCTS sweep of the environments
            st = time.time()
            data = eqx.filter_pmap(self.tree_search, 
                                    donate="all", 
                                    devices=self.devices,
                                    in_axes=(None, None, 0), 
                                    axis_name="num_devices")(batch, model, keys)
            print("tree search time", time.time() - st)
            
            st = time.time()
            data = jax.device_put(data, cpu_sharding.reshape(-1, 1, 1, 1))
            samples, self.buffer_states = fill_and_sample(data, self.buffer_states, sample_key)
            print("buffer", self.buffer_states.experience)
            print("fill and sample time", time.time() - st)
            return samples
            
        # TODO multiple gradient descent steps after one MCTS sweep?
        def gradient_descent_fn(model, samples, opt_state, key):                       
            # Reshape data
            states, search_policy, search_value, _ = jnp.split(samples, self.split_idxs, axis=-1)
            search_policy = search_policy.reshape(-1, num_actions)
            search_value = search_value.reshape(-1, 1)
            states = states.reshape(-1, *self.state_shape)
                    
            # Compute gradients based on surrogate AlphaZero loss
            subkeys = jrand.split(key, batchsize_per_gpu)
            A0_grads = eqx.filter_value_and_grad(A0_loss, has_aux=True)
            val, grads = A0_grads(model, 
                                search_policy, 
                                search_value, 
                                states,
                                value_weight,
                                L2_weight,
                                entropy_weight,
                                subkeys)
            loss, aux = val
            # We might have parallelization across multiple nodes and thus
            # compute the mean over multiple nodes
            loss = lax.pmean(loss, axis_name="num_devices")
            # aux = lax.pmean(aux, axis_name="num_devices")
            # parallel_mean = lambda x: lax.pmean(x, axis_name="num_devices")
            # grads = jtu.tree_map(parallel_mean, grads)
            
            print("ll", loss)

            nn_params = eqx.filter(model, eqx.is_inexact_array)
            updates, opt_state = self.optim.update(grads, opt_state, params=nn_params)
            model = eqx.apply_updates(model, updates)
            return loss, aux, model, opt_state
        
        gpu_sharding = PositionalSharding(self.devices)
        def train_fn(model, batch, opt_states, key):
            search_key, train_key = jrand.split(key, 2)
            train_keys = jrand.split(train_key, len(self.devices))
            
            samples = MCTS_tree_search_fn(model, batch, search_key)
            print("sample shapes", samples.shape)
            samples = jax.device_put(samples, gpu_sharding.reshape(-1, 1, 1))

            st = time.time()
            print(self.devices)
            loss, aux, model, opt_states = eqx.filter_pmap(gradient_descent_fn,
                                                            donate="all", 
                                                            devices=self.devices,
                                                            in_axes=(None, 0, None, 0), 
                                                            axis_name="num_devices")(model, samples, opt_states, train_keys)
            print("gradient descent time", time.time() - st)
            print("loss", loss)
            return loss, aux, model, opt_states
        
        self.train_fn = train_fn
        

    def _init_metrics(self, key: PRNGKey) -> None:
        if self.pid == 0:
            logger.info("Initializing metrics...")
            names, rews, orders = evaluate_tasks(self.model, 
                                                self.task_dataloader, 
                                                self.eval_env_interaction, 
                                                key)
            for name, rew, order in zip(names, rews, orders):
                self.best_performance[name] = (rew, order)

            self.reference_values = make_Markowitz_reference_values(self.benchmark_dataloader, 
                                                                    jax.local_devices())

    
    def _update_best_scores(self, names, fmas, orders) -> None:
        for name, fma, order in zip(names, fmas, orders):
            if fma > self.best_performance[name][0]:
                self.best_performance[name] = (fma, order)
        logger.info(self.best_performance)

