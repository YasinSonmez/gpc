from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax.struct import dataclass
from hydrax.task_base import Task
from mujoco import mjx


@dataclass
class SimulatorState:
    """A dataclass for storing the simulator state.

    Attributes:
        data: The mjx simulator data.
        t: The current time step.
        rng: The random number generator key.
        running_cost: Accumulated cost for the most recent control step.
    """

    data: mjx.Data
    t: int
    rng: jax.Array
    running_cost: jax.Array = 0.0  # Accumulated cost of the last control step


class TrainingEnv(ABC):
    """Abstract class defining a training environment."""

    def __init__(
        self,
        task: Task,
        episode_length: int,
        sim_steps_per_control_step: int = 1,
        render_camera: str = -1,
        render_resolution: tuple[int, int] = (640, 480),
        planning_horizon: int = 0,  # TODO: set
    ) -> None:
        """Initialize the training environment.

        Args:
            task: The hydrax task.
            episode_length: Number of control steps in an episode.
            sim_steps_per_control_step: Number of physics steps per control step.
            render_camera: Camera name or ID for rendering.
        """
        self.task = task
        self.planning_horizon = planning_horizon
        self.episode_length = episode_length
        self.sim_steps_per_control_step = sim_steps_per_control_step
        self.render_camera = render_camera if render_camera is not None else -1
        self.render_resolution = render_resolution
        self._renderer = None

    @property
    def renderer(self) -> mujoco.Renderer:
        """Lazy-initialize the renderer only when needed."""
        if self._renderer is None:
            width, height = self.render_resolution

            # Clamp resolution to model's offscreen framebuffer limits to avoid ValueError
            offwidth = self.task.mj_model.vis.global_.offwidth
            offheight = self.task.mj_model.vis.global_.offheight

            if offwidth > 0 and width > offwidth:
                print(
                    f"WARNING: Requested width {width} exceeds model offwidth {offwidth}. Clamping."
                )
                width = offwidth
            if offheight > 0 and height > offheight:
                print(
                    f"WARNING: Requested height {height} exceeds model offheight {offheight}. Clamping."
                )
                height = offheight

            self._renderer = mujoco.Renderer(
                self.task.mj_model, width=width, height=height
            )
            # Disable shadows and reflections for faster rendering unless high quality is needed
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = (
                False
            )
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
        return self._renderer

    def close(self) -> None:
        """Close the renderer and free resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def init_state(self, rng: jax.Array) -> SimulatorState:
        """Initialize the simulator state."""
        state = SimulatorState(
            data=mjx.make_data(self.task.model), t=0, rng=rng
        )
        return self._reset_state(state)

    def render(self, states: SimulatorState, fps: int = 30) -> np.ndarray:
        """Render video frames from a state trajectory.

        Note that this is not a pure jax function, and should only be used for
        visualization.

        Args:
            states: Sequence of states (vmapped over time).
            fps: The frames per second for the video.

        Returns:
            A sequence of video frames, with shape (T, C, H, W).
        """
        sim_dt = self.task.model.opt.timestep
        render_dt = 1.0 / fps
        render_every = int(round(render_dt / sim_dt))
        steps = np.arange(0, len(states.t), render_every)

        frames = []
        for i in steps:
            mjx_data = jax.tree.map(lambda x: x[i], states.data)  # noqa: B023
            mj_data = mjx.get_data(self.task.mj_model, mjx_data)
            try:
                self.renderer.update_scene(mj_data, camera=self.render_camera)
            except ValueError:
                # Fallback to free camera if specified camera doesn't exist
                self.renderer.update_scene(mj_data, camera=-1)
            pixels = self.renderer.render()  # H, W, C
            frames.append(pixels.transpose(2, 0, 1))  # C, H, W

        return np.stack(frames)

    def render_qpos(
        self,
        qpos: np.ndarray,
        fps: int = 30,
        proposal_stats: dict | None = None,
        proposal_data: dict | None = None,
    ) -> np.ndarray:
        """Render video frames from a qpos trajectory (memory-efficient).

        Only requires qpos arrays instead of full SimulatorState/mjx.Data,
        dramatically reducing GPU memory during training.

        Args:
            qpos: Joint positions, shape (T, nq).
            fps: Frames per second for the video.
            proposal_stats: Optional dict with per-timestep arrays:
                'policy_cost', 'spc_cost', 'episode_cost' (each shape (T,))
                for proposal overlay.
            proposal_data: Optional dict with per-timestep proposal data for
                drawing trajectories and ghost bodies on the MuJoCo scene:
                'trace_sites': (T, S, H+1, num_sites, 3)
                'costs': (T, S)
                'best_idx': (T,)
                'controls': (T, S, H, nu)
                'num_policy_samples': int
                'num_display': int (0 = all)
                'ghost_count': int (0 = disabled)
                'ghost_steps': int

        Returns:
            Video frames with shape (N, C, H, W).
        """
        from gpc.proposal_overlay import (
            add_proposal_geoms_to_scene,
            add_ghost_geoms_to_scene,
            build_stats_lines,
            text_overlay,
            color_legend_overlay,
            get_palette,
        )

        sim_dt = self.task.model.opt.timestep
        render_dt = 1.0 / fps
        render_every = max(1, int(round(render_dt / sim_dt)))
        total_steps = qpos.shape[0]
        steps = np.arange(0, total_steps, render_every)

        mj_data = mujoco.MjData(self.task.mj_model)
        frames = []

        has_proposals = proposal_data is not None
        if has_proposals:
            nps = proposal_data.get("num_policy_samples", 0)
            num_display = proposal_data.get("num_display", 0)
            ghost_count = proposal_data.get("ghost_count", 0)
            ghost_steps = proposal_data.get("ghost_steps", 4)
            pal = get_palette(proposal_data.get("palette", "tableau10"))
            n_spc = proposal_data["costs"].shape[1] - nps

            # Precompute cumulative stats over ALL timesteps (not just rendered frames)
            inst_costs = np.asarray(proposal_data.get("inst_costs", np.zeros(total_steps)))
            cum_costs = np.cumsum(inst_costs)
            is_policy_best = np.asarray(proposal_data["best_idx"]) < nps
            cum_policy_best = np.cumsum(is_policy_best.astype(float))

        for frame_idx, i in enumerate(steps):
            mj_data.qpos[:] = np.asarray(qpos[i])
            mujoco.mj_forward(self.task.mj_model, mj_data)
            try:
                self.renderer.update_scene(mj_data, camera=self.render_camera)
            except ValueError:
                self.renderer.update_scene(mj_data, camera=-1)

            if has_proposals:
                traces_t = np.asarray(proposal_data["trace_sites"][i])
                costs_t = np.asarray(proposal_data["costs"][i])
                best_t = int(proposal_data["best_idx"][i])

                # Draw proposal lines on the scene
                add_proposal_geoms_to_scene(
                    self.renderer._scene,
                    traces_t,
                    costs_t,
                    best_t,
                    num_policy_samples=nps,
                    pal=pal,
                    num_display=num_display,
                )

                # Draw ghost bodies
                if ghost_count > 0 and "controls" in proposal_data:
                    controls_t = np.asarray(proposal_data["controls"][i])
                    add_ghost_geoms_to_scene(
                        self.renderer._scene,
                        self.task.mj_model,
                        mj_data.qpos.copy(),
                        mj_data.qvel.copy(),
                        controls_t,
                        costs_t,
                        best_t,
                        num_policy_samples=nps,
                        pal=pal,
                        num_ghosts=ghost_count,
                        ghost_steps=ghost_steps,
                    )

            pixels = self.renderer.render().copy()  # H, W, C

            # Burn stats overlay if proposals are present
            if has_proposals:
                cum_cost = float(cum_costs[i])
                pct_policy_best = 100.0 * float(cum_policy_best[i]) / (i + 1)

                lines = build_stats_lines(
                    i, total_steps, costs_t, best_t, nps,
                    cum_cost, pct_policy_best,
                )
                # text_overlay/color_legend_overlay expect BGR, convert round-trip
                frame_bgr = pixels[:, :, ::-1]  # RGB → BGR
                frame_bgr = text_overlay(frame_bgr, lines)
                frame_bgr = color_legend_overlay(frame_bgr, pal, n_spc)
                pixels = frame_bgr[:, :, ::-1].copy()  # BGR → RGB

            frames.append(pixels.transpose(2, 0, 1))  # C, H, W

        result = np.stack(frames)

        # Legacy text-only overlay (when proposal_data is not provided)
        if proposal_stats is not None and not has_proposals:
            result = overlay_proposal_stats(
                result, proposal_stats, steps, total_steps
            )

        return result

    @abstractmethod
    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""

    @abstractmethod
    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Get the observation from the simulator state."""

    @property
    @abstractmethod
    def observation_size(self) -> int:
        """The size of the observation space."""

    def _reset_state(self, state: SimulatorState) -> SimulatorState:
        """Reset the simulator state to start a new episode."""
        rng, reset_rng = jax.random.split(state.rng)
        data = self.reset(state.data, reset_rng)
        data = mjx.forward(self.task.model, data)  # update sensor data
        return SimulatorState(data=data, t=0, rng=rng)

    def _update_goal(self, state: SimulatorState) -> SimulatorState:
        """Update the goal state during the middle of an episode."""
        rng, goal_rng = jax.random.split(state.rng)
        data = self.update_goal(state.data, goal_rng)
        return state.replace(data=data, rng=rng)

    def _get_observation(self, state: SimulatorState) -> jax.Array:
        """Get the observation from the simulator state."""
        return self.get_obs(state.data)

    def episode_over(self, state: SimulatorState) -> bool:
        """Check if the episode is over.

        Override this method if the episode should terminate early.
        """
        return state.t >= self.episode_length

    def goal_reached(self, state: SimulatorState) -> bool:
        """Check if we've achieved a sub-goal.

        This gives us the opportunity to update the goal before the episode
        ends. For example, we might want to choose a new target configuration
        once the old one has been reached.
        """
        return False

    def update_goal(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Update the goal state during the middle of an episode.

        Typically this is done via mocap_pos and mocap_quat, and by default we
        do nothing.
        """
        return data

    def step(self, state: SimulatorState, action: jax.Array) -> SimulatorState:
        """Take a simulation step and return the new state."""

        def _step_fn(data, _):
            next_data = mjx.step(self.task.model, data.replace(ctrl=action))
            cost = self.task.running_cost(next_data, action)
            return next_data, cost

        def _do_step(_):
            next_data, costs = jax.lax.scan(
                _step_fn,
                state.data,
                None,
                length=self.sim_steps_per_control_step,
            )
            # Accumulate cost over all integration steps during this control step
            total_step_cost = jnp.sum(costs)
            return state.replace(
                data=next_data, t=state.t + 1, running_cost=total_step_cost
            )

        def _do_reset(_):
            # If episode is over, return reset state and 0 cost
            return self._reset_state(state).replace(running_cost=jnp.array(0.0))

        return jax.lax.cond(
            self.episode_over(state), _do_reset, _do_step, operand=None
        )

        # Check if we've reached a sub-goal that needs updating
        next_state = jax.lax.cond(
            self.goal_reached(next_state),
            lambda _: self._update_goal(next_state),
            lambda _: next_state,
            operand=None,
        )

        return next_state


def overlay_proposal_stats(
    frames: np.ndarray,
    stats: dict,
    steps: np.ndarray,
    episode_length: int,
) -> np.ndarray:
    """Overlay proposal statistics text on video frames.

    Args:
        frames: Video frames, shape (N, C, H, W).
        stats: Dict with per-timestep arrays (each shape (T,)):
            'policy_cost', 'spc_cost', 'episode_cost'.
        steps: Timestep indices corresponding to each frame.
        episode_length: Total episode length.

    Returns:
        Frames with overlay, shape (N, C, H, W).
    """
    from PIL import Image, ImageDraw, ImageFont

    policy_costs = np.asarray(stats["policy_cost"])
    spc_costs = np.asarray(stats["spc_cost"])
    episode_costs = np.asarray(stats["episode_cost"])

    cum_cost = np.cumsum(episode_costs)
    policy_better = np.cumsum(policy_costs < spc_costs)
    totals = np.arange(1, episode_length + 1)
    policy_best_pct = policy_better / totals * 100

    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    BLUE = (100, 150, 255)
    ORANGE = (255, 165, 0)
    GREEN = (0, 255, 100)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    result = []
    for i, t in enumerate(steps):
        frame_hwc = frames[i].transpose(1, 2, 0).copy()
        img = Image.fromarray(frame_hwc)
        draw = ImageDraw.Draw(img)

        p_cost = float(policy_costs[t])
        s_cost = float(spc_costs[t])
        best_src = "Policy" if p_cost < s_cost else "SPC"

        lines = [
            (f"Step {t}/{episode_length}", WHITE),
            (f"Policy: {p_cost:.2f}", BLUE),
            (f"SPC:    {s_cost:.2f}", ORANGE),
            (f"Best: {best_src}", GREEN if best_src == "Policy" else ORANGE),
            (f"Ep Cost: {cum_cost[t]:.1f}", WHITE),
            (f"Policy Best: {policy_best_pct[t]:.1f}%", WHITE),
        ]

        y_pos = 8
        for text, color in lines:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((10 + dx, y_pos + dy), text, fill=BLACK, font=font)
            draw.text((10, y_pos), text, fill=color, font=font)
            y_pos += 20

        result.append(np.array(img).transpose(2, 0, 1))

    return np.stack(result)
