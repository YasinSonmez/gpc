"""Ant environment wrapper for GPC training with Gymnasium/Brax rewards."""

import jax
import jax.numpy as jnp
from mujoco import mjx

from gpc.envs import TrainingEnv
from gpc.envs.tasks.ant_gym import AntGym


class AntGymEnv(TrainingEnv):
    """Training environment for Ant with Gymnasium/Brax-compatible rewards.
    
    This exactly matches the Brax Ant environment's behavior for perfect
    compatibility between RL and GPC training.
    """
    
    def __init__(self, episode_length: int, render_camera: str = "track", 
                 terminate_when_unhealthy: bool = True,
                 sim_steps_per_control_step: int = 5, **kwargs) -> None:
        """Initialize Ant environment.
        
        Args:
            episode_length: Maximum episode length.
            render_camera: Camera for rendering.
            terminate_when_unhealthy: If True, terminate episode when ant falls.
            sim_steps_per_control_step: Number of simulation steps per control step.
            **kwargs: Additional arguments for TrainingEnv.
        """
        # Default to 'track' camera for better visualization if 'floating' is requested
        if render_camera == "floating":
            render_camera = "track"
            
        super().__init__(
            task=AntGym(terminate_when_unhealthy=terminate_when_unhealthy),
            episode_length=episode_length,
            sim_steps_per_control_step=sim_steps_per_control_step,
            render_camera=render_camera,
            **kwargs
        )
        self._prev_x_pos = None
    
    def episode_over(self, state) -> bool:
        """Check if episode should terminate early due to falling."""
        if self.task.terminate_when_unhealthy:
            is_healthy = self.task._is_healthy(state.data)
            return (~is_healthy) | (state.t >= self.episode_length)
        return state.t >= self.episode_length
    
    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset using Brax Ant's exact initialization."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        
        # Brax Ant: uniform noise [-0.1, 0.1] for q, normal (std=0.1) for qd
        low, hi = -0.1, 0.1
        q = self.task.model.qpos0 + jax.random.uniform(
            rng1, (self.task.model.nq,), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.task.model.nv,))
        
        return data.replace(qpos=q, qvel=qd)
    
    
    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Get observations matching Brax Ant exactly (27-dim).
        
        Excludes x,y position: [z_pos, quaternion(4), joint_angles(8), velocities(14)]
        """
        qpos = data.qpos[2:]  # Exclude x, y position
        qvel = data.qvel
        return jnp.concatenate([qpos, qvel])
    
    @property
    def observation_size(self) -> int:
        """27-dimensional observation space."""
        return 27
    
    @property
    def action_size(self) -> int:
        """8-dimensional action space."""  
        return 8
