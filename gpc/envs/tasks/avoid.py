import os
from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.task_base import Task

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "avoid")

SCENE_FILES = {
    "sphere": "scene.xml",
    "vertical_block": "scene_vertical_block.xml",
    "u_trap": "scene_u_trap.xml",
}


class Avoid(Task):
    """Planar point mass must reach a goal while avoiding a central obstacle."""

    def __init__(self, variant: str = "sphere") -> None:
        """Load the MuJoCo model and set task parameters."""
        if variant not in SCENE_FILES:
            raise ValueError(
                f"Unknown avoid variant '{variant}'. Available: {sorted(SCENE_FILES)}"
            )

        mj_model = mujoco.MjModel.from_xml_path(
            os.path.join(MODEL_DIR, SCENE_FILES[variant])
        )
        super().__init__(mj_model, trace_sites=["pointmass"])
        self.variant = variant
        self.pointmass_id = mj_model.site("pointmass").id
        goal_body_id = mj_model.body("goal").id
        obstacle_body_id = mj_model.body("obstacle").id
        self.goal_mocap_id = int(mj_model.body_mocapid[goal_body_id])
        self.obstacle_mocap_id = int(mj_model.body_mocapid[obstacle_body_id])
        self.obstacle_clearance = 0.08
        self.obstacle_cost_weight = 25.0

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost encourages target tracking with control penalty."""
        state_cost = self.terminal_cost(state)
        control_cost = jnp.sum(jnp.square(control))
        return state_cost + 0.1 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost penalizes distance to goal and velocity."""
        pointmass_pos = state.site_xpos[self.pointmass_id, 0:2]
        position_cost = jnp.sum(
            jnp.square(
                pointmass_pos - state.mocap_pos[self.goal_mocap_id, 0:2]
            )
        )
        velocity_cost = jnp.sum(jnp.square(state.qvel))
        obstacle_cost = self._obstacle_cost(
            pointmass_pos, state.mocap_pos[self.obstacle_mocap_id, 0:2]
        )
        return position_cost + 0.1 * velocity_cost + obstacle_cost

    def obstacle_signed_distance(self, state: mjx.Data) -> jax.Array:
        """Return signed distance from the point mass to the obstacle geometry."""
        pointmass_pos = state.site_xpos[self.pointmass_id, 0:2]
        obstacle_pos = state.mocap_pos[self.obstacle_mocap_id, 0:2]
        return self._obstacle_signed_distance(pointmass_pos, obstacle_pos)

    def _obstacle_cost(
        self, pointmass_pos: jax.Array, obstacle_pos: jax.Array
    ) -> jax.Array:
        signed_distance = self._obstacle_signed_distance(
            pointmass_pos, obstacle_pos
        )
        return self.obstacle_cost_weight * jnp.square(
            jnp.maximum(self.obstacle_clearance - signed_distance, 0.0)
        )

    def _obstacle_signed_distance(
        self, pointmass_pos: jax.Array, obstacle_pos: jax.Array
    ) -> jax.Array:
        if self.variant == "sphere":
            return jnp.linalg.norm(pointmass_pos - obstacle_pos) - 0.1

        if self.variant == "vertical_block":
            return self._box_signed_distance(
                pointmass_pos, obstacle_pos, jnp.array([0.02, 0.12])
            )

        back_wall = self._box_signed_distance(
            pointmass_pos,
            obstacle_pos + jnp.array([0.04, 0.0]),
            jnp.array([0.02, 0.10]),
        )
        top_arm = self._box_signed_distance(
            pointmass_pos,
            obstacle_pos + jnp.array([-0.01, 0.10]),
            jnp.array([0.05, 0.02]),
        )
        bottom_arm = self._box_signed_distance(
            pointmass_pos,
            obstacle_pos + jnp.array([-0.01, -0.10]),
            jnp.array([0.05, 0.02]),
        )
        return jnp.minimum(back_wall, jnp.minimum(top_arm, bottom_arm))

    @staticmethod
    def _box_signed_distance(
        pointmass_pos: jax.Array,
        box_center: jax.Array,
        half_extents: jax.Array,
    ) -> jax.Array:
        delta = jnp.abs(pointmass_pos - box_center) - half_extents
        outside = jnp.linalg.norm(jnp.maximum(delta, 0.0))
        inside = jnp.minimum(jnp.max(delta), 0.0)
        return outside + inside

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

