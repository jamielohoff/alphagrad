from functools import partial
from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex


@chex.dataclass(frozen=True)
class GridworldState:
    """TODO docstring
    """
    t: chex.Array
    position: chex.Array
    moves: chex.Array  


class GridworldGame2D:
    """
    Simple gridworld.
    """
    walls: chex.Array
    goal: chex.Array
    move_map: chex.Array
    
    def __init__(self, walls: chex.Array, goal: chex.Array, max_steps: int = 50) -> None:
        super().__init__()
        self.walls = jnp.pad(walls.astype(jnp.float32), 1, mode="constant", constant_values=((1, 1), (1, 1)))
        self.goal = goal+1
        self.move_map = jnp.asarray([[-1, 0], [0, 1], [1, 0], [0, -1]])
        self.starting_positions = jnp.stack(jnp.nonzero(self.walls == 0)).transpose(1, 0)
        self.num_starting_positions = len(self.starting_positions)
        self.i_max = self.walls.shape[0] - 1
        self.j_max = self.walls.shape[1] - 1
        self.max_steps = max_steps

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
            state: GridworldState, 
            action: int) -> Tuple[chex.Array, chex.Array, float, bool]:  
        done = False
        step = state.t
        position = state.position
        moves = state.moves
        
        direction = self.move_map[action]        
        new_i = position[0] + direction[0]
        new_j = position[1] + direction[1]
                
        reward, new_i, new_j = lax.cond(self.walls[new_i.astype(jnp.int32), new_j.astype(jnp.int32)] == 1,
                                lambda: (-0.75, position[0], position[1]),
                                lambda: (-0.05, new_i, new_j))
        
        reward = lax.cond(moves[new_i.astype(jnp.int32), new_j.astype(jnp.int32)] > 0,
                                lambda: -0.25,
                                lambda: -0.05)
                        
        new_position = jnp.array([new_i, new_j])

        # Check if the goal is reached and give reward        
        reward, done = lax.cond(jnp.array_equal(new_position, self.goal),
                                lambda: (1., True),
                                lambda: (reward, False))    
            
        # Check if we reached max steps      
        new_position, reward, done = lax.cond(step >= self.max_steps,
                                                lambda: (position, -1., True),
                                                lambda: (new_position, reward, done))     
        
        # Check if the goal is reached and give reward        
        new_position, reward, done = lax.cond(jnp.array_equal(position, self.goal),
                                                lambda: (position, 0., True),
                                                lambda: (new_position, reward, done))  
        
        new_i, new_j = new_position.astype(jnp.int32)
        new_moves = moves.at[new_i, new_j].add(1.)
        
        new_state = GridworldState(t=step+1, position=new_position, moves=new_moves)
        return new_state, self.get_obs(new_moves), reward, done
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jrand.PRNGKey) -> chex.Array:
        idx = jrand.randint(key, (1,), 0, self.num_starting_positions)[0]
        i, j = self.starting_positions[idx].astype(jnp.int32)
        moves = jnp.zeros_like(self.walls)
        moves = moves.at[i, j].set(1.)
        return GridworldState(t=0, position=self.starting_positions[idx], moves=moves)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, moves: chex.Array):
        obs = jnp.zeros_like(self.walls)
        obs = obs.at[:, :].set(moves)
        return jnp.ravel(obs)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_surroundings(self, position):
        i, j = position.astype(jnp.int32)
        top = self.walls.at[].get()
        return jnp.array([top, right, bottom, left])
    
    def num_actions(self):
        return len(self.move_map)

key = jrand.PRNGKey(123)
goal = jnp.array([0, 4])
walls = jnp.array( [[0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0]])  

env = GridworldGame2D(walls, goal)
state = env.reset(key)
print(env.walls)

state, obs, rew, done = env.step(state, 1)
print(obs, state, rew)
state, obs, rew, done = env.step(state, 2)
print(obs, state, rew)
state, obs, rew, done = env.step(state, 2)
print(obs, state)

    