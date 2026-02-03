import jax
import jax.numpy as jnp
from hydrax.tasks.cart_pole import CartPole
from mujoco import mjx

from gpc.envs import TrainingEnv


class CartPoleEnv(TrainingEnv):
    """Training environment for the cartpole swingup task."""

    def __init__(self, episode_length: int, render_camera: str = -1) -> None:
        """Set up the cartpole training environment."""
        super().__init__(
            task=CartPole(),
            episode_length=episode_length,
            render_camera=render_camera,
        )

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, theta_rng, pos_rng, vel_rng = jax.random.split(rng, 4)

        theta = jax.random.uniform(theta_rng, (), minval=-3.14, maxval=3.14)
        pos = jax.random.uniform(pos_rng, (), minval=-1.8, maxval=1.8)
        qvel = jax.random.uniform(vel_rng, (2,), minval=-2.0, maxval=2.0)
        qpos = jnp.array([pos, theta])

        return data.replace(qpos=qpos, qvel=qvel)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the velocity and sin/cos of the angle."""
        p = data.qpos[0]
        theta = data.qpos[1]
        v = data.qvel[0]
        theta_dot = data.qvel[1]
        return jnp.array([p, jnp.cos(theta), jnp.sin(theta), v, theta_dot])

    @property
    def observation_size(self) -> int:
        """The size of the observation space (includes sin and cos)."""
        return 5
