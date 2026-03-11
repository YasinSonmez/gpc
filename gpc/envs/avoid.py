import jax
import jax.numpy as jnp
from hydrax.tasks.avoid import Avoid
from mujoco import mjx

from gpc.envs import TrainingEnv


class AvoidEnv(TrainingEnv):
    """Training environment for the particle task."""

    def __init__(
        self, episode_length: int = 100, render_camera: str = -1, **kwargs
    ) -> None:
        """Set up the particle training environment."""
        super().__init__(
            task=Avoid(),
            episode_length=episode_length,
            render_camera=render_camera,
            **kwargs,
        )

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng, mocap_rng = jax.random.split(rng, 4)
        qpos = jax.random.uniform(pos_rng, (2,), minval=-0.29, maxval=0.29)
        qvel = jax.random.uniform(vel_rng, (2,), minval=-0.5, maxval=0.5)
        target = jax.random.uniform(mocap_rng, (2,), minval=-0.29, maxval=0.29)
        mocap_pos = data.mocap_pos.at[0, 0:2].set(target)
        return data.replace(qpos=qpos, qvel=qvel, mocap_pos=mocap_pos)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the position relative to the target and the velocity."""
        pos = (
            data.site_xpos[self.task.pointmass_id, 0:2] - data.mocap_pos[0, 0:2]
        )
        vel = data.qvel[:]
        return jnp.concatenate([pos, vel])

    @property
    def observation_size(self) -> int:
        """The size of the observation space."""
        return 4
