import jax
import jax.numpy as jnp
from mujoco import mjx

from gpc.envs import TrainingEnv
from gpc.envs.tasks.walker_gym import WalkerGym


class WalkerGymEnv(TrainingEnv):
    """Training environment for the walker task with Gymnasium rewards."""

    def __init__(self, episode_length: int, render_camera: str = -1) -> None:
        """Set up the walker training environment."""
        super().__init__(
            task=WalkerGym(),
            episode_length=episode_length,
            sim_steps_per_control_step=1,
            render_camera=render_camera,
        )

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)

        # Joint limits are zero for the floating base
        q_min = self.task.model.jnt_range[:, 0]
        q_max = self.task.model.jnt_range[:, 1]
        
        # Start in a slightly random upright position
        q_min = q_min.at[2].set(-0.2)  # orientation
        q_max = q_max.at[2].set(0.2)
        qpos = jax.random.uniform(pos_rng, (9,), minval=q_min, maxval=q_max)
        
        # Ensure height is healthy at start (relative to body pos 1.3)
        qpos = qpos.at[0].set(0.0) # z = 1.3m absolute
        
        qvel = jax.random.uniform(vel_rng, (9,), minval=-0.1, maxval=0.1)

        return data.replace(qpos=qpos, qvel=qvel)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe everything in the state except the horizontal position."""
        pz = data.qpos[0]  # base coordinates are (z, x, theta)
        theta = data.qpos[2]
        base_pos_data = jnp.array([jnp.cos(theta), jnp.sin(theta), pz])
        return jnp.concatenate([base_pos_data, data.qpos[3:], data.qvel])

    @property
    def observation_size(self) -> int:
        """The size of the observation space."""
        return 18
