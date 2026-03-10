import jax
import jax.numpy as jnp
from mujoco import mjx

from gpc.envs.pusht import PushTEnv


class PushTRLEnv(PushTEnv):
    """Push-T environment with enriched observations for RL training.

    Adds velocities, relative positions, and sensor-based errors to the
    base PushTEnv observation, making the task more tractable for model-free RL.

    Base obs (5-dim):  pusher_pos, block_pos, block_ori_err
    Enriched obs (18-dim):
        pusher_pos        (2)  - absolute pusher xy
        block_pos         (2)  - absolute block xy
        block_angle       (1)  - raw joint angle (theta)
        qvel              (5)  - all velocities (block_vx, block_vy, block_w, pusher_vx, pusher_vy)
        pusher_to_block   (2)  - relative position (block - pusher)
        position_err      (3)  - block position relative to goal (sensor)
        orientation_err   (3)  - block rotation error relative to goal (3D rotation vector)
    """

    def __init__(self, episode_length: int, render_camera: str = -1, **kwargs) -> None:
        super().__init__(
            episode_length=episode_length,
            render_camera=render_camera,
            **kwargs
        )

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Enriched observation for RL training."""
        # Positions
        pusher_pos = data.qpos[-2:]                        # (2,)
        block_pos = data.qpos[0:2]                         # (2,)
        block_angle = data.qpos[2:3]                       # (1,)

        # Velocities
        qvel = data.qvel                                   # (5,)

        # Relative positions
        pusher_to_block = block_pos - pusher_pos            # (2,)

        # Sensor-based errors (relative to goal)
        position_err = self.task._get_position_err(data)    # (3,)
        orientation_err = self.task._get_orientation_err(data)  # (3,)

        return jnp.concatenate([
            pusher_pos,         # 2
            block_pos,          # 2
            block_angle,        # 1
            qvel,               # 5
            pusher_to_block,    # 2
            position_err,       # 3
            orientation_err,    # 3
        ])  # Total: 18

    @property
    def observation_size(self) -> int:
        return 18
