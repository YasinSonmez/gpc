from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from hydrax import ROOT
from hydrax.task_base import Task


class AdroitHammer(Task):
    """Door open task for Adroit hand."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/adroit_hand/adroit_hammer.xml"
        )
        super().__init__(mj_model)

        # # Get sensor and site ids
        # self.orientation_sensor_id = mj_model.sensor("imu_in_torso_quat").id
        # self.velocity_sensor_id = mj_model.sensor("imu_in_torso_linvel").id
        # self.torso_id = mj_model.site("imu_in_torso").id

        # # Set the target height
        # self.target_height = 0.9
        # change actuator sensitivity

        # # Standing configuration
        # self.qstand = jnp.array(mj_model.keyframe("stand").qpos)
        self.grasp_site_id = mj_model.site("S_grasp").id
        self.target_obj_site_id = mj_model.site("S_target").id
        self.obj_body_id = mj_model.body("Object").id
        self.tool_site_id = mj_model.site("tool").id
        self.goal_site_id = mj_model.site("nail_goal").id
        self.target_body_id = mj_model.body("nail_board").id

        self.sparse_reward = False

    # def _get_torso_orientation(self, state: mjx.Data) -> jax.Array:
    #     """Get the rotation from the current torso orientation to upright."""
    #     sensor_adr = self.model.sensor_adr[self.orientation_sensor_id]
    #     quat = state.sensordata[sensor_adr : sensor_adr + 4]
    #     upright = jnp.array([0.0, 0.0, 1.0])
    #     return mjx._src.math.rotate(upright, quat)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""

        hamm_pos = state.xpos[self.obj_body_id].ravel()
        palm_pos = state.site_xpos[self.grasp_site_id].ravel()
        head_pos = state.site_xpos[self.tool_site_id].ravel()
        nail_pos = state.site_xpos[self.target_obj_site_id].ravel()
        goal_pos = state.site_xpos[self.goal_site_id].ravel()

        goal_distance = jnp.linalg.norm(nail_pos - goal_pos)
        goal_achieved = goal_distance < 0.01
        reward = jnp.where(goal_achieved, 10.0, -0.1)

        # override reward if not sparse reward
        if not self.sparse_reward:
            # get the palm to the hammer handle
            reward = 0.1 * jnp.linalg.norm(palm_pos - hamm_pos)
            # take hammer head to nail
            reward = reward - jnp.linalg.norm(head_pos - nail_pos)
            # make nail go inside
            reward = reward + 10 * jnp.linalg.norm(nail_pos - goal_pos)
            # velocity penalty
            reward = reward - 1e-2 * jnp.linalg.norm(state.qvel.ravel())

            reward += jnp.where(
                jnp.logical_and(hamm_pos[2] > 0.04, head_pos[2] > 0.04), 2, 0
            )

            reward += jnp.where(goal_distance < 0.02, 25, 0)
            reward += jnp.where(goal_distance < 0.01, 75, 0)

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
