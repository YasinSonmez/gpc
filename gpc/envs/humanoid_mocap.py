"""Humanoid mocap training environment — Unitree G1 tracking LocoMuJoCo reference."""
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from gpc.envs import TrainingEnv
from gpc.envs.tasks.humanoid_mocap_unconstrained import HumanoidMocapUnconstrained


def _humanoid_mocap_camera(mj_model: mujoco.MjModel) -> mujoco.MjvCamera:
    """Tracking camera that follows the humanoid pelvis."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    
    # Try to find pelvis or torso body ID
    pelvis_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        pelvis_id = 1 # Fallback
    
    cam.trackbodyid = pelvis_id
    cam.distance = 3.5  # Much closer than default
    cam.azimuth = 90.0
    cam.elevation = -15.0  # Lower angle (closer to ground level)
    return cam


def _normalize_quat(q: jax.Array) -> jax.Array:
    """Normalize quaternion (w,x,y,z) at indices 3:7 of qpos."""
    quat = q[3:7]
    return q.at[3:7].set(quat / jnp.linalg.norm(quat))


class HumanoidMocapEnv(TrainingEnv):
    """Training environment for humanoid (Unitree G1) mocap tracking.

    Uses the same task as zoo-rob HumanoidMocapUnconstrained: same model,
    cost, and reference indexing so rendering and costs match.
    """

    def __init__(
        self,
        episode_length: int,
        reference_filename: str = "Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
        start: int = 100,
        render_camera: str | int | mujoco.MjvCamera = -1,
    ) -> None:
        task = HumanoidMocapUnconstrained(
            reference_filename=reference_filename,
            start=start,
        )
        # Default: free camera from model stat so the robot is in frame
        if render_camera in (-1, None):
            render_camera = _humanoid_mocap_camera(task.mj_model)
        super().__init__(
            task=task,
            episode_length=episode_length,
            render_camera=render_camera,
        )

    @property
    def renderer(self) -> mujoco.Renderer:
        """Renderer with shadow and softer lighting for humanoid (overrides base)."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.task.mj_model)
            # Enable shadow for natural depth (base disables it for speed)
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
            # Note: MjvScene has no .opt in the Python API; headlight uses defaults.
        return self._renderer

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset to a random time along the reference with small perturbation."""
        rng, time_rng, q_rng, v_rng = jax.random.split(rng, 4)

        duration = (self.task.reference.shape[0] - 1) / self.task.reference_fps
        t_start = jax.random.uniform(
            time_rng, (), minval=0.0, maxval=jnp.maximum(0.01, duration - 0.5)
        )

        q_ref = self.task._get_reference_configuration(t_start)
        qpos = q_ref + 0.01 * jax.random.normal(q_rng, (self.task.model.nq,))
        qpos = _normalize_quat(qpos)

        qvel = 0.01 * jax.random.normal(v_rng, (self.task.model.nv,))

        return data.replace(qpos=qpos, qvel=qvel, time=t_start)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observation: qpos, qvel, config error, foot position/orientation errors."""
        qpos = data.qpos
        qvel = data.qvel
        q_ref = self.task._get_reference_configuration(data.time)
        q_err = qpos - q_ref

        left_pos_err, right_pos_err = self.task._get_foot_position_errors(data)
        left_ori_err, right_ori_err = self.task._get_foot_orientation_errors(
            data
        )

        return jnp.concatenate([
            qpos,
            qvel,
            q_err,
            left_pos_err,
            right_pos_err,
            left_ori_err,
            right_ori_err,
        ])

    def render(self, states, fps: int = 10):
        """Render with the configured camera."""
        # Tracking camera automatically follows trackbodyid in update_scene
        return super().render(states, fps=fps)

    @property
    def observation_size(self) -> int:
        """nq + nv + nq + 3 + 3 + 3 + 3 (foot ori errors are 3D from quat_sub)."""
        return (
            self.task.model.nq * 2
            + self.task.model.nv
            + 3
            + 3
            + 3
            + 3
        )
