import jax
import jax.numpy as jnp
from mujoco import mjx

from gpc.envs import TrainingEnv
from gpc.envs.tasks.avoid import Avoid


class AvoidEnv(TrainingEnv):
    """Training environment for the avoid task with wall obstacle."""

    def __init__(
        self, episode_length: int = 200, render_camera: str = -1, **kwargs
    ) -> None:
        """Set up the avoid training environment."""
        super().__init__(
            task=Avoid(),
            episode_length=episode_length,
            render_camera=render_camera,
            **kwargs,
        )

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset with start on left side, goal on right side.

        The wall obstacle at x=0 forces a left/right (above/below) path choice,
        creating a clear multimodal planning problem.
        """
        rng, pos_rng, vel_rng, goal_rng = jax.random.split(rng, 4)

        # Start: left side, y offset randomized
        start_y = jax.random.uniform(pos_rng, (), minval=-0.08, maxval=0.08)
        # qpos is joint displacement from body origin (-0.25, 0)
        qpos = jnp.array([0.0, start_y])
        qvel = jnp.zeros(2)

        # Goal: right side, y offset randomized
        goal_y = jax.random.uniform(goal_rng, (), minval=-0.08, maxval=0.08)
        mocap_pos = data.mocap_pos.at[0, 0:2].set(jnp.array([0.25, 0]))
        # mocap_pos = data.mocap_pos.at[0, 0:2].set(jnp.array([0.25, goal_y]))

        return data.replace(qpos=qpos, qvel=qvel, mocap_pos=mocap_pos)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe position relative to goal, velocity, and position relative to obstacle."""
        particle_pos = data.site_xpos[self.task.pointmass_id, 0:2]
        goal_pos = data.mocap_pos[0, 0:2]
        obstacle_pos = data.mocap_pos[1, 0:2]

        pos_to_goal = particle_pos - goal_pos
        pos_to_obstacle = particle_pos - obstacle_pos
        vel = data.qvel[:]
        return jnp.concatenate([pos_to_goal, vel, pos_to_obstacle])

    @property
    def observation_size(self) -> int:
        """The size of the observation space."""
        return 6
