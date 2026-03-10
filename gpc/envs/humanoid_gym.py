"""Humanoid environment wrapper for GPC training with Gymnasium/Brax rewards."""

import jax
import jax.numpy as jnp
from mujoco import mjx

from gpc.envs import TrainingEnv
from gpc.envs.tasks.humanoid_gym import HumanoidGym


class HumanoidGymEnv(TrainingEnv):
    """Training environment for Humanoid with Gymnasium/Brax-compatible rewards.
    
    This exactly matches the Brax Humanoid environment's behavior.
    """
    
    def __init__(self, episode_length: int, render_camera: str = -1, 
                 terminate_when_unhealthy: bool = True,
                 sim_steps_per_control_step: int = 5, **kwargs) -> None:
        """Initialize Humanoid environment.
        
        Args:
            episode_length: Maximum episode length.
            render_camera: Camera for rendering.
            terminate_when_unhealthy: If True, terminate episode when humanoid falls.
            sim_steps_per_control_step: Frame skip (Brax default is 5).
            **kwargs: Additional arguments for TrainingEnv.
        """
        super().__init__(
            task=HumanoidGym(terminate_when_unhealthy=terminate_when_unhealthy),
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
    
    def step(self, state, action: jax.Array):
        """Take a simulation step with action scaling.
        
        Brax Humanoid maps actions from [-1, 1] to the actuator ctrlrange.
        """
        # Map action: [-1, 1] -> [ctrl_min, ctrl_max]
        ctrl_min = self.task.model.actuator_ctrlrange[:, 0]
        ctrl_max = self.task.model.actuator_ctrlrange[:, 1]
        scaled_action = (action + 1) * 0.5 * (ctrl_max - ctrl_min) + ctrl_min
        
        # Use the scaled action for simulation
        return super().step(state, scaled_action)
    
    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset using Brax Humanoid's exact initialization.
        
        qpos = init_q + uniform(-0.01, 0.01)
        qvel = uniform(-0.01, 0.01)
        """
        rng, rng1, rng2 = jax.random.split(rng, 3)
        
        low, hi = -0.01, 0.01
        q = self.task.model.qpos0 + jax.random.uniform(
            rng1, (self.task.model.nq,), minval=low, maxval=hi
        )
        qd = jax.random.uniform(
            rng2, (self.task.model.nv,), minval=low, maxval=hi
        )
        
        return data.replace(qpos=q, qvel=qd)
    
    
    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Get observations matching Brax Humanoid exactly (376-dim).
        
        1. qpos[2:] (exclude x, y)
        2. qvel
        3. cinert (CoM-based body inertia)
        4. cvel (CoM-based body velocity)
        5. qfrc_actuator
        """
        qpos = data.qpos[2:]  # Exclude x, y position
        qvel = data.qvel
        
        # mjx.Data provides cinert and cvel
        # cinert: (nbody, 10). ravel to (nbody * 10)
        # cvel: (nbody, 6). ravel to (nbody * 6)
        # qfrc_actuator: (nv,)
        
        cinert = data.cinert.ravel()
        cvel = data.cvel.ravel()
        qfrc_actuator = data.qfrc_actuator
        
        return jnp.concatenate([
            qpos, 
            qvel,
            cinert,
            cvel,
            qfrc_actuator
        ])
    
    @property
    def observation_size(self) -> int:
        """292-dimensional observation space (excludes external contact forces)."""
        return 292
    
    @property
    def action_size(self) -> int:
        """17-dimensional action space."""  
        return 17
