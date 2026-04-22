import os
from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.task_base import Task

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "avoid")


class Avoid(Task):
    """Planar point mass must reach a goal while avoiding a wall obstacle."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            os.path.join(MODEL_DIR, "scene.xml")
        )
        super().__init__(mj_model, trace_sites=["pointmass"])
        self.pointmass_id = mj_model.site("pointmass").id

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost encourages target tracking with control penalty."""
        state_cost = self.terminal_cost(state)
        control_cost = jnp.sum(jnp.square(control))
        return state_cost + 0.1 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost penalizes distance to goal and velocity."""
        position_cost = jnp.sum(
            jnp.square(
                state.site_xpos[self.pointmass_id, 0:2] - state.mocap_pos[0, 0:2]
            )
        )
        velocity_cost = jnp.sum(jnp.square(state.qvel))
        return 1.0 * position_cost + 0.1 * velocity_cost

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomly perturb the actuator gains."""
        multiplier = jax.random.uniform(
            rng, self.model.actuator_gainprm[:, 0].shape, minval=0.9, maxval=1.1
        )
        new_gains = self.model.actuator_gainprm[:, 0] * multiplier
        new_gains = self.model.actuator_gainprm.at[:, 0].set(new_gains)
        return {"actuator_gainprm": new_gains}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly shift the measured particle position."""
        shift = jax.random.uniform(rng, (2,), minval=-0.01, maxval=0.01)
        return {"qpos": data.qpos + shift}

