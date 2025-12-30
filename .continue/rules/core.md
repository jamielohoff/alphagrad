---
description: AlphaGrad rule
---

# Project Architecture

This is research code which implements the AlphaGrad RL agent.
This agent is able to use RL to find the best possible automatic differentiation 
algorithm for a given function/computational graph.

- ML models are in `/src/alphagrad/transformer/`
- AlphaGrad implementation with AlphaZero is in `/src/alphagrad/alphazero/`
- AlphaGrad implementation with PPO is in `/src/alphagrad/ppo/`
- RL environment implementation is in `/src/alphagrad/vertexgame/`
- Examples for evaluation of the RL agent are in `/src/alphagrad/eval/`
- Relevant unit tests can be found under `/tests`

## Coding Standards

- The whole project is written in Python and JAX
- Follow the existing naming conventions
- Follow the PEP8 standard

# External Dependencies

The ML models, e.g. transformer implementation is based on the equinox package:

- [equinox](https://github.com/patrick-kidger/equinox)

The AlphaZero implementation is based on:

- [mctx](https://github.com/google-deepmind/mctx)
- [Example implementation](https://github.com/google-deepmind/mctx/tree/main/examples)

When implementing auth features, reference these patterns.

# Repository Access

You can use the `gh` CLI to:

- Search for issues: `gh issue list --repo jamielohoff/alphagrad`
- View pull requests: `gh pr list --repo jamielohoff/alphagrad`
- Clone repositories: `gh repo clone jamielohoff/alphagrad`