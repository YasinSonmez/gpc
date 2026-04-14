"""Humanoid task with Gymnasium/Brax-compatible rewards."""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.task_base import Task
from gpc.envs.tasks.ant_gym import _ensure_trace_site


class HumanoidGym(Task):
    """Humanoid biped tasked with forward locomotion, following Brax/Gymnasium rewards.
    
    This implementation exactly matches the Brax Humanoid environment's reward structure.
    Cost = -reward
    """
    
    def __init__(self, terminate_when_unhealthy: bool = True) -> None:
        """Load the MuJoCo model and set task parameters."""
        from pathlib import Path
        # Use the same asset path as the RL expert
        asset_path = Path(__file__).parent.parent.parent.parent / "brax" / "brax" / "envs" / "assets" / "humanoid.xml"
        
        if asset_path.exists():
            mj_model = mujoco.MjModel.from_xml_path(str(asset_path))
        else:
            # Fallback for when brax is installed as a package
            from brax import envs
            brax_env = envs.create('humanoid', backend='mjx', terminate_when_unhealthy=terminate_when_unhealthy)
            mj_model = brax_env.sys.mj_model
        
        mj_model, _site_names = _ensure_trace_site(mj_model, body_name="torso")

        super().__init__(mj_model, trace_sites=_site_names)
        
        # Brax Humanoid parameters
        self.forward_reward_weight = 1.25
        self.ctrl_cost_weight = 0.1
        self.healthy_reward = 10.0
        self.healthy_z_range = (1.0, 2.0)
        self.terminate_when_unhealthy = terminate_when_unhealthy
        
        # Store dt
        self.dt = mj_model.opt.timestep
    
    def _is_healthy(self, state: mjx.Data) -> jax.Array:
        """Check if humanoid is in healthy z-range."""
        # Index 2 is z-coordinate (0=x, 1=y, 2=z)
        z = state.qpos[2]
        return (z >= self.healthy_z_range[0]) & (z <= self.healthy_z_range[1])
    
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Calculate cost matching Brax Humanoid exactly."""
        # Forward reward = CoM x-velocity
        # com_vel = sum(mass_i * v_i) / total_mass
        # state.cvel[:, 3:6] is the linear velocity of each body's CoM in global frame
        # state.subtree_com[1] is the CoM of the humanoid (subtree of body 1)
        
        # We use the mass-weighted average of body velocities
        mass = self.model.body_mass[1:] # Exclude world
        lin_vel = state.cvel[1:, 3:6]
        com_vel = jnp.sum(mass[:, None] * lin_vel, axis=0) / jnp.sum(mass)
        forward_reward = self.forward_reward_weight * com_vel[0]
        
        # Control cost
        # control is the physical torque (u)
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(control))
        
        # Healthy reward
        is_healthy = self._is_healthy(state)
        healthy_reward = jnp.where(is_healthy, self.healthy_reward, 0.0)
        
        # Combine reward
        reward = forward_reward + healthy_reward - ctrl_cost
        
        # Cost = -reward
        return -reward
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return jnp.array(0.0)
