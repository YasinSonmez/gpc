"""Half Cheetah environment wrapper for GPC training with Brax/Gymnasium rewards."""

import jax
import jax.numpy as jnp
from mujoco import mjx

from gpc.envs import TrainingEnv
from gpc.envs.tasks.half_cheetah_gym import HalfCheetahGym


class HalfCheetahGymEnv(TrainingEnv):
    """Training environment for Half Cheetah with Brax-compatible rewards.

    Matches the Brax HalfCheetah environment exactly:
    - 9 qpos (rootx, rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot)
    - 9 qvel (corresponding velocities)
    - 6 actuators (6 hinge joints, excluding the 3 root joints)
    - 17-dim observations (exclude rootx position)
    - n_frames=5 (action_repeat=5)
    - No early termination
    """

    def __init__(
        self,
        episode_length: int,
        render_camera: str = -1,
        sim_steps_per_control_step: int = 5,
        **kwargs,
    ) -> None:
        """Initialize Half Cheetah environment.

        Args:
            episode_length: Maximum episode length.
            render_camera: Camera for rendering.
            sim_steps_per_control_step: Number of simulation steps per control step.
                Brax uses n_frames=5 by default.
        """
        super().__init__(
            task=HalfCheetahGym(),
            episode_length=episode_length,
            sim_steps_per_control_step=sim_steps_per_control_step,
            render_camera=render_camera,
            **kwargs
        )

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset using Brax HalfCheetah's exact initialization.

        Brax uses:
        - qpos = init_q + uniform(-0.1, 0.1)
        - qvel = 0.1 * normal(0, 1)
        """
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -0.1, 0.1
        qpos = self.task.model.qpos0 + jax.random.uniform(
            rng1, (self.task.model.nq,), minval=low, maxval=hi
        )
        qvel = hi * jax.random.normal(rng2, (self.task.model.nv,))

        return data.replace(qpos=qpos, qvel=qvel)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Get observations matching Brax HalfCheetah exactly (17-dim).

        Brax's _get_obs with exclude_current_positions_from_observation=True:
        position = qpos[1:]  (exclude rootx)
        velocity = qvel
        obs = concat(position, velocity)
        """
        position = data.qpos[1:]  # Exclude rootx (8 dims)
        velocity = data.qvel  # All velocities (9 dims)
        return jnp.concatenate([position, velocity])

    @property
    def observation_size(self) -> int:
        """17-dimensional observation space (8 positions + 9 velocities)."""
        return 17

    @property
    def action_size(self) -> int:
        """6-dimensional action space."""
        return 6
