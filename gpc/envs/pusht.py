import jax
import jax.numpy as jnp
from hydrax.tasks.pusht import PushT
from mujoco import mjx

from gpc.envs import TrainingEnv


class PushTEnv(TrainingEnv):
    """Training environment for the pusher-T task."""

    def __init__(self, episode_length: int, render_camera: str = -1) -> None:
        """Set up the walker training environment."""
        super().__init__(
            task=PushT(),
            episode_length=episode_length,
            render_camera=render_camera,
        )

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng, goal_pos_rng, goal_ori_rng = jax.random.split(
            rng, 5
        )

        # Random configuration for the pusher and the T
        q_min = jnp.array([-0.2, -0.2, -jnp.pi, -0.2, -0.2])
        q_max = jnp.array([0.2, 0.2, jnp.pi, 0.2, 0.2])
        qpos = jax.random.uniform(pos_rng, (5,), minval=q_min, maxval=q_max)

        # Velocities fixed at zero
        qvel = jax.random.uniform(vel_rng, (5,), minval=-0.0, maxval=0.0)

        # Goal position and orientation fixed at zero
        goal = jax.random.uniform(goal_pos_rng, (2,), minval=-0.0, maxval=0.0)
        mocap_pos = data.mocap_pos.at[0, 0:2].set(goal)
        theta = jax.random.uniform(goal_ori_rng, (), minval=0.0, maxval=0.0)
        mocap_quat = jnp.array([[jnp.cos(theta / 2), 0, 0, jnp.sin(theta / 2)]])

        return data.replace(
            qpos=qpos, qvel=qvel, mocap_pos=mocap_pos, mocap_quat=mocap_quat
        )

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe positions relative to the target."""
        pusher_pos = data.qpos[-2:]
        block_pos = data.qpos[0:2]
        block_ori = self.task._get_orientation_err(data)[0:1]
        return jnp.concatenate([pusher_pos, block_pos, block_ori])

    @property
    def observation_size(self) -> int:
        """The size of the observation space."""
        return 5
