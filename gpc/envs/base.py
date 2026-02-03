from abc import ABC, abstractmethod

import jax
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
    """

    data: mjx.Data
    t: int
    rng: jax.Array


class TrainingEnv(ABC):
    """Abstract class defining a training environment."""

    def __init__(
        self,
        task: Task,
        episode_length: int,
        sim_steps_per_control_step: int = 1,
        render_camera: str = -1,
    ) -> None:
        """Initialize the training environment.
        
        Args:
            task: The hydrax task.
            episode_length: Number of control steps in an episode.
            sim_steps_per_control_step: Number of physics steps per control step.
            render_camera: Camera name or ID for rendering.
        """
        self.task = task
        self.episode_length = episode_length
        self.sim_steps_per_control_step = sim_steps_per_control_step
        self.render_camera = render_camera if render_camera is not None else -1
        self._renderer = None

    @property
    def renderer(self) -> mujoco.Renderer:
        """Lazy-initialize the renderer only when needed."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.task.mj_model)
            # Disable shadows and reflections for faster rendering
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
        return self._renderer

    def init_state(self, rng: jax.Array) -> SimulatorState:
        """Initialize the simulator state."""
        state = SimulatorState(
            data=mjx.make_data(self.task.model), t=0, rng=rng
        )
        return self._reset_state(state)

    def render(self, states: SimulatorState, fps: int = 10) -> np.ndarray:
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
        """Take a simulation step.

        Args:
            state: The simulator state.
            action: The action to take.

        Returns:
            The new simulator state and the new time step.
        """
        def _step_fn(data, _):
            return mjx.step(self.task.model, data), None

        # Check if the episode is over
        next_state = jax.lax.cond(
            self.episode_over(state),
            lambda _: self._reset_state(state),
            lambda _: state.replace(
                data=jax.lax.scan(
                    _step_fn,
                    state.data.replace(ctrl=action),
                    None,
                    length=self.sim_steps_per_control_step,
                )[0],
                t=state.t + 1,
            ),
            operand=None,
        )

        # Check if we've reached a sub-goal that needs updating
        next_state = jax.lax.cond(
            self.goal_reached(next_state),
            lambda _: self._update_goal(next_state),
            lambda _: next_state,
            operand=None,
        )

        return next_state
