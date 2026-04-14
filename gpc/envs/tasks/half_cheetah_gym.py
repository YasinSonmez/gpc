"""Half Cheetah task with Brax/Gymnasium-compatible rewards.

Implements the Brax HalfCheetah reward structure as a GPC cost function.
Cost = -reward = -(forward_velocity - ctrl_cost_weight * sum(action^2))
"""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from pathlib import Path

from hydrax.task_base import Task
from gpc.envs.tasks.ant_gym import _ensure_trace_site


class HalfCheetahGym(Task):
    """A planar cheetah tasked with running forward, following Brax/Gymnasium rewards.

    Reward = forward_reward_weight * x_velocity - ctrl_cost_weight * sum(action^2)
    Cost = -reward (no healthy reward, no termination)
    """

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        asset_path = Path(__file__).parent.parent.parent / "assets" / "half_cheetah.xml"

        if asset_path.exists():
            mj_model = mujoco.MjModel.from_xml_path(str(asset_path))
        else:
            raise FileNotFoundError(
                f"Half Cheetah XML not found at {asset_path}. "
                "Copy it from brax/brax/envs/assets/half_cheetah.xml"
            )

        mj_model, _site_names = _ensure_trace_site(mj_model, body_name="torso")

        super().__init__(mj_model, trace_sites=_site_names)

        # Brax HalfCheetah parameters (matching brax/envs/half_cheetah.py)
        self.forward_reward_weight = 1.0
        self.ctrl_cost_weight = 0.1

        # Store dt for velocity calculation
        self.dt = mj_model.opt.timestep

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ), matching Brax HalfCheetah exactly.

        In Brax, forward_reward = (x_after - x_before) / dt = x_velocity.
        Since MuJoCo gives us qvel[0] (rootx velocity) directly, we use that.
        """
        # Forward velocity (rootx is qvel[0] for the slide joint)
        forward_reward = self.forward_reward_weight * state.qvel[0]

        # Control cost
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(control))

        # Total reward (no healthy reward for half cheetah)
        reward = forward_reward - ctrl_cost

        # Cost = -reward
        return -reward

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """No terminal cost (Brax HalfCheetah has no terminal reward)."""
        return jnp.array(0.0)
