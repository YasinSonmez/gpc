import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class WalkerGym(Task):
    """A planar biped tasked with walking forward, following Gymnasium rewards."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/walker/scene.xml"
        )
        super().__init__(
            mj_model,
            trace_sites=["torso_site"],
        )

        # Get sensor ids
        self.torso_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_position"
        )
        self.torso_velocity_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_subtreelinvel"
        )
        self.torso_zaxis_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_zaxis"
        )

        # Standard Gymnasium parameters
        self.healthy_reward = 1.0
        self.ctrl_cost_weight = 0.001
        self.forward_reward_weight = 1.0
        
        # Healthy ranges (Gymnasium defaults)
        self.healthy_z_range = (0.8, 2.0)
        self.healthy_angle_range = (-1.0, 1.0)

    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        """Get the absolute height of the torso center.
        
        The 'torso' body in walker.xml is at z=1.3m in the default pose (qpos[0]=0).
        """
        return 1.3 + state.qpos[0]

    def _get_torso_velocity(self, state: mjx.Data) -> jax.Array:
        """Get the horizontal velocity of the torso."""
        sensor_adr = self.model.sensor_adr[self.torso_velocity_sensor]
        return state.sensordata[sensor_adr]

    def _get_torso_angle(self, state: mjx.Data) -> jax.Array:
        """Get the absolute tilt angle of the torso (orientation in X-Z plane).
        
        In the planar walker model, the torso tilt is the rooty joint (qpos[2]).
        """
        return jnp.abs(state.qpos[2])

    def _is_healthy(self, state: mjx.Data) -> jax.Array:
        """Check if the walker is in a healthy state."""
        height = self._get_torso_height(state)
        angle = self._get_torso_angle(state)
        
        # angle threshold is 1.0 rad (~57 deg) by default.
        # Since we use abs(qpos[2]), we check against 1.0.
        is_healthy = (height > self.healthy_z_range[0]) & \
                     (height < self.healthy_z_range[1]) & \
                     (angle < self.healthy_angle_range[1]) # Absolute tilt threshold
        return is_healthy

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ), formulated from Gymnasium rewards."""
        # Standard Gymnasium Reward Components:
        # reward = healthy_reward + forward_reward - ctrl_cost
        
        forward_reward = self.forward_reward_weight * self._get_torso_velocity(state)
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(control))
        
        # In Gymnasium, being "unhealthy" terminates the episode (future reward 0)
        # In GPC, we use a penalty to discourage falling
        is_healthy = self._is_healthy(state)
        
        # When healthy, cost = -reward
        reward_if_healthy = self.healthy_reward + forward_reward - ctrl_cost
        
        # Return -reward if healthy, or a large penalty if fallen
        return jnp.where(is_healthy, -reward_if_healthy, 1000.0)

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T). 
        
        Note: Gymnasium has no terminal reward. We set this to 0.0 to match.
        The optimizer still sees penalties from the running_cost if it falls 
        within its planning horizon.
        """
        return jnp.array(0.0)
