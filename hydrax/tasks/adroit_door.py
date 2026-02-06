from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class AdroitDoor(Task):
    """Door open task for Adroit hand."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/adroit_hand/adroit_door.xml"
        )
        super().__init__(mj_model)

        # # Get sensor and site ids
        # self.orientation_sensor_id = mj_model.sensor("imu_in_torso_quat").id
        # self.velocity_sensor_id = mj_model.sensor("imu_in_torso_linvel").id
        # self.torso_id = mj_model.site("imu_in_torso").id

        # # Set the target height
        # self.target_height = 0.9

        # # Standing configuration
        # self.qstand = jnp.array(mj_model.keyframe("stand").qpos)
        self.door_hinge_addrs = mj_model.joint("door_hinge").id
        self.grasp_site_id = mj_model.site("S_grasp").id
        self.handle_site_id = mj_model.site("S_handle").id
        self.door_body_id = mj_model.body("frame").id
        self.sparse_reward = False

    # def _get_torso_orientation(self, state: mjx.Data) -> jax.Array:
    #     """Get the rotation from the current torso orientation to upright."""
    #     sensor_adr = self.model.sensor_adr[self.orientation_sensor_id]
    #     quat = state.sensordata[sensor_adr : sensor_adr + 4]
    #     upright = jnp.array([0.0, 0.0, 1.0])
    #     return mjx._src.math.rotate(upright, quat)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        goal_distance = state.qpos[self.door_hinge_addrs]
        goal_achieved = goal_distance >= 1.35

        reward = jnp.where(goal_achieved, 10.0, -0.1)

        # override reward if not sparse reward
        if not self.sparse_reward:  # Fine because condition is static
            handle_pos = state.site_xpos[self.handle_site_id].ravel()
            palm_pos = state.site_xpos[self.grasp_site_id].ravel()

            # get to handle
            reward = 0.1 * jnp.linalg.norm(palm_pos - handle_pos)
            # open door
            reward += -0.1 * (goal_distance - 1.57) * (goal_distance - 1.57)
            # velocity cost
            reward += -1e-5 * jnp.sum(state.qvel**2)

            # Bonus reward
            reward += jnp.where(goal_distance > 0.2, 2, 0)
            reward += jnp.where(goal_distance > 0.8, 2, 0)
            reward += jnp.where(goal_distance > 1.35, 2, 0)

        return reward

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(state, jnp.zeros(self.model.nu))

    # def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
    #     """Randomize the friction parameters."""
    #     n_geoms = self.model.geom_friction.shape[0]
    #     multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
    #     new_frictions = self.model.geom_friction.at[:, 0].set(
    #         self.model.geom_friction[:, 0] * multiplier
    #     )
    #     return {"geom_friction": new_frictions}

    # def domain_randomize_data(
    #     self, data: mjx.Data, rng: jax.Array
    # ) -> Dict[str, jax.Array]:
    #     """Randomly perturb the measured base position and velocities."""
    #     rng, q_rng, v_rng = jax.random.split(rng, 3)
    #     q_err = 0.01 * jax.random.normal(q_rng, (7,))
    #     v_err = 0.01 * jax.random.normal(v_rng, (6,))

    #     qpos = data.qpos.at[0:7].set(data.qpos[0:7] + q_err)
    #     qvel = data.qvel.at[0:6].set(data.qvel[0:6] + v_err)

    #     return {"qpos": qpos, "qvel": qvel}
