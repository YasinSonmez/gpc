import jax
import jax.numpy as jnp
from hydrax.tasks.crane import Crane
from mujoco import mjx

from gpc.envs import TrainingEnv


class CraneEnv(TrainingEnv):
    """Training environment for the luffing crane end-effetor tracking task."""

    def __init__(self, episode_length: int = 100, render_camera: str = -1) -> None:
        """Set up the particle training environment."""
        super().__init__(
            task=Crane(),
            episode_length=episode_length,
            render_camera=render_camera,
        )

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng, target_rng = jax.random.split(rng, 4)

        # Crane state
        # TODO: figure out a more principled way to initialize these.
        # Right now the crane swings wildly at initialization, which prevents
        # gathering a lot of data near the target (which is where the system is
        # mostly at run time).
        q_lim = jnp.array(
            [
                [-1.0, 1.0],  # slew
                [0.0, 1.0],  # luff
                [-1.0, 1.0],  # payload x-pos
                [1.0, 2.2],  # payload y-pos
                [0.3, 1.0],  # payload z-pos
                [1.0, 1.0],  # payload orientation (fixed upright)
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        qpos = self.task.model.qpos0 + jax.random.uniform(
            pos_rng,
            (self.task.model.nq,),
            minval=q_lim[:, 0],
            maxval=q_lim[:, 1],
        )
        qvel = jax.random.uniform(
            vel_rng, (self.task.model.nv,), minval=-0.1, maxval=0.1
        )

        # Target position
        # TODO: figure out a better set of potential target positions
        pos_min = jnp.array([-1.5, 1.0, 0.0])
        pos_max = jnp.array([1.5, 3.0, 1.5])
        target_pos = jax.random.uniform(
            target_rng, (3,), minval=pos_min, maxval=pos_max
        )
        mocap_pos = data.mocap_pos.at[0].set(target_pos)

        # Target orientation - this is unused but must be set so vectorization
        # (which is determined by the size of rng) works properly.
        target_quat = jnp.array([1.0, 0.0, 0.0, 0.0]) + jax.random.uniform(
            target_rng, (4,), minval=-0.0, maxval=0.0
        )
        mocap_quat = data.mocap_quat.at[0].set(target_quat)

        return data.replace(
            qpos=qpos, qvel=qvel, mocap_pos=mocap_pos, mocap_quat=mocap_quat
        )

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the full crane state, plus end-effector pos/vel."""
        ee_pos = self.task._get_payload_position(data)
        ee_vel = self.task._get_payload_velocity(data)
        return jnp.concatenate([ee_pos, ee_vel, data.qpos, data.qvel])

    @property
    def observation_size(self) -> int:
        """The size of the observation space."""
        return 23
