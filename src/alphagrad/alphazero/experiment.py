import os
import time
from functools import partial
from typing import Callable, Dict, NamedTuple, Sequence, Tuple

from torch.utils.data import DataLoader

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import dejax
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
                    postprocess_data,
                    make_batch)

from .environment_interaction import make_recurrent_fn, make_environment_interaction
from .evaluate import (evaluate_tasks, 
                        evaluate_benchmark,
                        make_Markowitz_reference_values)


class HyperParameters(NamedTuple):
    seed: PRNGKey
    num_epochs: int
    batchsize: int
    learning_rate: float
    
    num_envs: int
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
                                        num_envs=16,
                                        num_simulations=200,
                                        replay_buffer_size=10000,
                                        gumbel_scale=.5,
                                        value_weight=1.,
                                        L2_weight=0.,
                                        entropy_weight=0.,
                                        task_freq=5,
                                        chckpt_freq=500,
                                        bench_freq=100)
    

class Paths(NamedTuple):
    file_path: str
    task_file_path: str
    benchmark_file_path: str
    chckpt_path: str
    weights_path: str
    
    
# Helper functions
select_first = lambda x: x[0] if isinstance(x, jax.Array) else x
parallel_mean = lambda x: lax.pmean(x, "num_devices")


# Filtering low-value samples, TODO define low-value samples
def fill_buffer(replay_buffer, buffer_state, samples):
    def loop_fn(buffer_state, sample):
        new_buffer_state = lax.cond(sample[-1] > 0,
                                    lambda bs: replay_buffer.add_fn(bs, sample),
                                    lambda bs: bs,
                                    buffer_state)
        # new_buffer_state = replay_buffer.add_fn(buffer_state, traj)
        return new_buffer_state, None
    updated_buffer_state, _ = lax.scan(loop_fn, buffer_state, samples)
    return updated_buffer_state


class AlphaGradExperiment:
    name: str
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
    
    train_dataloader: DataLoader
    task_dataloader: DataLoader
    benchmark_dataloader: DataLoader
    
    tree_search: Callable
    train_fn: Callable
    eval_env_interaction: Callable
    replay_buffer: dejax.base.ReplayBuffer
    item_prototype: Array
    
    best_performance: Dict
    reference_values: Dict
    
    def __init__(self, 
                name: str, 
                model: eqx.Module,
                paths: Paths,
                devices: Sequence[jax.Device],
                hyperparameters: HyperParameters = DEFAULT_HYPERPARAMETERS,
                num_nodes: int = 4,
                coordinator_address: str = None,
                pid: int = 0,
                num_inputs: int = 20,
                num_outputs: int = 20,
                num_actions: int = 105,
                use_wandb: bool = False) -> None:
        
        # General configuration of the experiment
        self.name = name
        self.hyperparameters = hyperparameters
        self.paths = paths
        self.use_wandb = use_wandb
        
        # Multi-node configuration
        self.pid = pid
        self.devices = devices
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
        if self.paths.weights_path is not None:
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
        self.replay_buffer = dejax.uniform_replay(max_size=size)
        self.item_prototype = jnp.zeros((4, reward_idx+1))
        
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
        
        # Initialization of optimizer state on multiple devices
        nn_params = eqx.filter(self.model, eqx.is_inexact_array)
        optim_init = lambda params, g: self.optim.init(params)
        opt_states = jax.pmap(optim_init, 
                                axis_name="num_devices", 
                                devices=self.devices,
                                in_axes=(None, 0))(nn_params, jnp.zeros(4))

        # Initialization of a parallel replay buffer on multiple devices
        buffer_states = jax.pmap(self.replay_buffer.init_fn, 
                                axis_name="num_devices", 
                                devices=self.devices)(self.item_prototype)
        
        # TODO Initialization of a parallel model
        # models = jax.pmap(self.model, 
        #                 axis_name="num_devices", 
        #                 devices=self.devices)(self.item_prototype)

        counter = 0
        pbar = tqdm(range(1, self.hyperparameters.num_epochs+1))
        for epoch in pbar:
            for edges in tqdm(self.train_dataloader):
                train_key, key = jrand.split(key, 2)
                batch = make_batch(edges, num_devices=len(self.devices))
                
                start_time = time.time()
                train_keys = jrand.split(train_key, len(self.devices))
                output = self.train_fn(self.model, batch, opt_states, buffer_states, train_keys)	
                losses, aux, models, opt_states, buffer_states = output
                logger.info(f"Training time {time.time() - start_time}")
                
                loss = losses[0]
                aux = aux[0]
                # TODO also do not tree_map this?
                self.model = jtu.tree_map(select_first, models, is_leaf=eqx.is_inexact_array)
                
                # self.log_wandb(loss, aux)
                # self.checkpoint_model(counter)
                # self.eval_tasks(counter, key)
                # self.eval_benchmark(counter, key)

                pbar.set_description(f"loss: {loss}")
                counter += 1
                    
                        
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
        train_dataset = GraphDataset(self.paths.file_path)
        logger.info(f"Training dataset size: {len(train_dataset)}")
        task_graph_dataset = GraphDataset(self.paths.task_file_path, 
                                        include_code=True)
        benchmark_graph_dataset = GraphDataset(self.paths.benchmark_file_path)

        # Initialize data loaders
        num_envs = self.hyperparameters.num_envs
        self.train_dataloader = DataLoader(train_dataset, 
                                            batch_size=num_envs, 
                                            shuffle=True, 
                                            num_workers=8, 
                                            drop_last=True)
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

        env_interaction = make_environment_interaction(num_actions, 
                                                        gumbel_scale,
                                                        num_simulations,
                                                        recurrent_fn,
                                                        step)
        
        def tree_search(training_batch, model, key) -> Array:
            init_carry = make_init_state(training_batch, key)
            data = env_interaction(model, init_carry)
            return postprocess_data(data)
        
        self.tree_search = tree_search
        self.eval_env_interaction = make_environment_interaction(num_actions, 
                                                                gumbel_scale,
                                                                num_simulations,
                                                                recurrent_fn,
                                                                step)
        
        
    def _init_train_fn(self) -> None:
        num_actions = self.num_actions
        batchsize = self.hyperparameters.batchsize
        value_weight = self.hyperparameters.value_weight
        L2_weight = self.hyperparameters.L2_weight
        entropy_weight = self.hyperparameters.entropy_weight
        fill_buf = partial(fill_buffer, self.replay_buffer)
        
        def train_fn(model, batch, opt_state, buffer_state, key):
            search_key, sample_key, key = jrand.split(key, 3)
            # Do a MCTS sweep of the environments
            data = self.tree_search(batch, model, search_key)
            data = data.reshape(-1, 66257)
            
            # Fill replay buffer
            buffer_state = fill_buf(buffer_state, data)
            
            # Sample from replay buffer
            samples = self.replay_buffer.sample_fn(buffer_state, sample_key, batchsize)
            
            # Reshape data
            state, search_policy, search_value, _ = jnp.split(samples, self.split_idxs, axis=-1)
            search_policy = search_policy.reshape(-1, num_actions)
            search_value = search_value.reshape(-1, 1)
            state = state.reshape(-1, *self.state_shape)
        
            # Compute gradients based on surrogate AlphaZero loss
            subkeys = jrand.split(key, batchsize)
            A0_grads = eqx.filter_value_and_grad(A0_loss, has_aux=True)
            val, grads = A0_grads(model, 
                                search_policy, 
                                search_value, 
                                state,
                                value_weight,
                                L2_weight,
                                entropy_weight,
                                subkeys)
            loss, aux = val
            
            # We might have parallelization across multiple nodes and thus
            # compute the mean over multiple nodes
            loss = lax.pmean(loss, axis_name="num_devices")
            aux = lax.pmean(aux, axis_name="num_devices")
            grads = jtu.tree_map(parallel_mean, grads)

            nn_params = eqx.filter(model, eqx.is_inexact_array)
            updates, opt_state = self.optim.update(grads, opt_state, params=nn_params)
            model = eqx.apply_updates(model, updates)
            return loss, aux, model, opt_state, buffer_state
        
        self.train_fn = eqx.filter_pmap(train_fn, 
                                        in_axes=(None, 0, 0, 0, 0), 
                                        axis_name="num_devices", 
                                        devices=self.devices, 
                                        donate="all")


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

