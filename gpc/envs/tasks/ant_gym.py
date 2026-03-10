"""Ant task with Gymnasium/Brax-compatible rewards."""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.task_base import Task


class AntGym(Task):
    """Ant quadruped tasked with forward locomotion, following Brax/Gymnasium rewards.
    
    This implementation exactly matches the Brax Ant environment's reward structure.
    Cost = -reward
    """
    
    def __init__(self, terminate_when_unhealthy: bool = True) -> None:
        """Load the MuJoCo model and set task parameters."""
        from pathlib import Path
        asset_path = Path(__file__).parent.parent.parent / "assets" / "ant_quality.xml"
        
        if asset_path.exists():
            mj_model = mujoco.MjModel.from_xml_path(str(asset_path))
        else:
            from brax import envs
            brax_env = envs.create('ant', backend='mjx', terminate_when_unhealthy=terminate_when_unhealthy)
            mj_model = brax_env.sys.mj_model
        
        super().__init__(mj_model, trace_sites=[])
        
        # Brax Ant parameters
        self.ctrl_cost_weight = 0.5
        self.healthy_reward = 1.0
        self.healthy_z_range = (0.2, 1.0)
        self.healthy_up_vector_limit = 0.2
        self.contact_cost_weight = 5e-4
        self.terminate_when_unhealthy = terminate_when_unhealthy
        
        # Store dt - note: forward_reward uses qvel[0] which is already in m/s
        # No need to multiply by dt since we just want instantaneous velocity
        self.dt = mj_model.opt.timestep
    
    def _is_healthy(self, state: mjx.Data) -> jax.Array:
        """Check if ant is healthy (z-range and upright)."""
        z = state.qpos[2]
        # Torso quaternion (w, x, y, z) for the free joint
        q = state.qpos[3:7]
        # Up-vector z-component: 1 - 2(x^2 + y^2)
        up_z = 1.0 - 2.0 * (jnp.square(q[1]) + jnp.square(q[2]))
        
        is_healthy = (z >= self.healthy_z_range[0]) & (z <= self.healthy_z_range[1])
        is_healthy &= (up_z >= self.healthy_up_vector_limit)
        return is_healthy
    
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Calculate cost matching Brax Ant exactly."""
        # Forward reward = torso x-velocity
        forward_reward = state.qvel[0]
        
        # Control cost
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(control))
        
        # Healthy reward
        is_healthy = self._is_healthy(state)
        healthy_reward = jnp.where(is_healthy, self.healthy_reward, 0.0)
        
        # Contact cost (minimal impact in default Brax)
        contact_cost = 0.0 
        
        # Combine reward (Unscaled, as requested)
        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        
        # Cost = -reward if healthy, or penalty if fallen
        return jnp.where(is_healthy, -reward, 10.0)
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return jnp.array(0.0)
