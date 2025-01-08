# AlphaGrad

This package contains the core implementation of the AlphaGrad reinforcement 
learning algorithm.

## Installation
The following packages are required to successfully run the algorithms.
- jax GPU installation from https://github.com/google/jax
- mctx package for tree search from https://github.com/google-deepmind/mctx
- optax
- equinox
- numpy, scipy, matplotlib
- distrax from https://github.com/google-deepmind/distrax
- flashbax from https://instadeepai.github.io/flashbax/
- tqdm
Finally, the Graphax package needs to be installed as well since it contains the
implementation of the sample task. The package is contained in the ZIP file of
this project.
It can be installed using `pip install -e .` in the root directory of the package.
Similarly, the AlphaGrad package itself has to be installed by executing
`pip install -e .` in the root directory of this package.

## Usage
To start a run of the RL algorithm, use the following command.
`CUDA_VISIBLE_DEVICES=0,1,2,3 vertex_A0.py --task RoeFlux_1d --name test --seed 123`
The **config** subfolder contains .yaml files to configure the hyperparameters
of the experiments.
Similarly, you can run the experiments of `separate_models_vertex_ppo.py` and
`vertex_A0_joint.py`.
Note that it is necessary to set up `wandb` to log the experiments.
Use `--wandb disabled` to deactivate it.

## Directory structure
The project structure is described in the following section:
- alphagrad
    - src
        - alphagrad
            - **alphazero**
                - *environment_interaction.py*
                    Contains the implementation of the Monte-Carlo Tree Search.
                - *vertex_A0.py*
                    Run this script for a single task experiment with AlphaZero.
                - *vertex_A0_joint.py*
                    Run this script for a single task experiment with AlphaZero.
            - **eval**
                This folder contains the evaluation of the elimination orders
                found by the RL algorithm. Also contains the reward curves and
                Jupyter notebooks used to create the figures from the paper.
            - **ppo**
                - *runtime_vertex_ppo.py*
                    Run this script for a single task experiment with PPO and
                    actual runtime as reward for the model. Not tested yet. 
                - *separate_models_vertex_ppo.py*
                    Run this script for a single task experiment with PPO where
                    We use a separate model for policy and value networks.
                - *vertex_ppo_joint.py*
                    Run this script for joint experiments with PPO. This
                    script is decomissioned.
                - *vertex_ppo.py*
                    Run this script for a single task experiment with PPO where
                    we use a joint model for policy and value functions.
            - **transformer**
                Implementation of transformer model used in this work.
            - **vertexgame**
                - **interpreter**
                    This folder contains the functions that trace the python
                    function to create the computational graph representation
                    used as the state for the RL algorithm.
                - **codegeneration**
                    This folder contains the source code that was used to generate
                    the random functions *f* and *g*.
                - **transforms**
                    This folder contains a set of transformations similar to 
                    image augumentations in computer vision.
                - *vertex_game.py*
                    Implementation of the *VertexGame* reinforcement learning game
                    using the number of multiplications as a reward.
                - *runtime_game.py*
                    Implementation of the *VertexGame* reinforcement learning game
                    using the runtime as a reward. 
                - *core.py*
                    Core implementation of the environment dynamics model of
                    cross-country elimination with sparsity types, Jacobian shapes etc.
                
    - docs
    - tests

## Plots and results
The **eval** subfolder contains the code used to evaluate the elimination orders.
For every experiment, the folder contains an appropriate subfolder with a 
`.ipynb` notebook.
The `graphax.jacve(f, order=order, argnums=argnums)` command computes the Jacobian
with Graphax for a given elimination order `order`. The syntax is similar to 
the syntax of `jax.jacfwd` and `jax.jacrev`.
Runtime performances are tested with the `graphax.perf` package.

