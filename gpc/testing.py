import time
from functools import partial
from pathlib import Path
from typing import Optional, Union

import imageio
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from gpc.envs import SimulatorState, TrainingEnv
from gpc.policy import Policy


def test_interactive(
    env: TrainingEnv,
    policy: Policy,
    mj_data: mujoco.MjData = None,
    inference_timestep: float = 0.1,
    warm_start_level: float = 1.0,
) -> None:
    """Test a GPC policy with an interactive simulation.

    Args:
        env: The environment, which defines the system to simulate.
        policy: The GPC policy to test.
        mj_data: The initial state for the simulation.
        inference_timestep: The timestep dt to use for flow matching inference.
        warm_start_level: The warm start level to use for the policy.
    """
    rng = jax.random.key(0)
    task = env.task

    # Set up the policy
    policy = policy.replace(dt=inference_timestep)
    policy.model.eval()
    jit_policy = jax.jit(
        partial(policy.apply, warm_start_level=warm_start_level)
    )

    # Set up the mujoco simultion
    mj_model = task.mj_model
    if mj_data is None:
        mj_data = mujoco.MjData(mj_model)

    # Initialize the action sequence
    actions = jnp.zeros((task.planning_horizon, task.model.nu))

    # Set up an observation function
    mjx_data = mjx.make_data(task.model)

    @jax.jit
    def get_obs(mjx_data: mjx.Data) -> jax.Array:
        """Get an observation from the mujoco data."""
        mjx_data = mjx.forward(task.model, mjx_data)  # update sites & sensors
        return env.get_obs(mjx_data)

    # Run the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            st = time.time()

            # Get an observation
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
            )
            obs = get_obs(mjx_data)

            # Update the action sequence
            inference_start = time.time()
            rng, policy_rng = jax.random.split(rng)
            actions = jit_policy(actions, obs, policy_rng)
            mj_data.ctrl[:] = actions[0]

            inference_time = time.time() - inference_start
            obs_time = inference_start - st
            print(
                f"  Observation time: {obs_time:.5f}s "
                f" Inference time: {inference_time:.5f}s",
                end="\r",
            )

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            elapsed = time.time() - st
            if elapsed < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed)

    # Save what was last in the print buffer
    print("")


def test_and_record(
    env: TrainingEnv,
    policy: Policy,
    num_episodes: int = 3,
    output_dir: Union[Path, str] = "./videos",
    inference_timestep: float = 0.1,
    warm_start_level: float = 1.0,
    video_fps: int = 30,
) -> None:
    """Test a GPC policy and record videos (works in headless mode).

    Args:
        env: The environment, which defines the system to simulate.
        policy: The GPC policy to test.
        num_episodes: Number of episodes to run and record.
        output_dir: Directory to save videos.
        inference_timestep: The timestep dt to use for flow matching inference.
        warm_start_level: The warm start level to use for the policy.
        video_fps: Frames per second for the output video.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.key(0)
    task = env.task

    # Set up the policy
    policy = policy.replace(dt=inference_timestep)
    policy.model.eval()
    jit_policy = jax.jit(
        partial(policy.apply, warm_start_level=warm_start_level)
    )

    @jax.jit
    def get_obs(mjx_data: mjx.Data) -> jax.Array:
        """Get an observation from the mujoco data."""
        mjx_data = mjx.forward(task.model, mjx_data)  # update sites & sensors
        return env.get_obs(mjx_data)

    @jax.jit
    def simulate_episode(
        rng: jax.Array,
    ) -> tuple[SimulatorState, jax.Array]:
        """Simulate a single episode."""
        # Initialize state
        state = env.init_state(rng)
        actions = jnp.zeros((task.planning_horizon, task.model.nu))

        def step_fn(carry: tuple, t: int) -> tuple:
            state, actions, rng = carry
            rng, policy_rng = jax.random.split(rng)

            # Get observation and compute action
            obs = get_obs(state.data)
            actions = jit_policy(actions, obs, policy_rng)

            # Apply first action and step
            u = actions[0]
            state = env.step(state, u)

            return (state, actions, rng), state

        _, states = jax.lax.scan(
            step_fn, (state, actions, rng), jnp.arange(env.episode_length)
        )

        # Compute total cost
        total_cost = jnp.sum(
            jax.vmap(task.running_cost, in_axes=(0, None))(
                states.data, jnp.zeros(task.model.nu)
            )
        )

        return states, total_cost

    print(f"Running {num_episodes} evaluation episodes...")
    costs = []

    for ep in range(num_episodes):
        print(f"  Episode {ep + 1}/{num_episodes}...", end=" ")
        episode_start = time.time()

        # Run episode
        rng, ep_rng = jax.random.split(rng)
        states, cost = simulate_episode(ep_rng)
        costs.append(float(cost))

        episode_time = time.time() - episode_start
        print(f"cost={cost:.4f}, time={episode_time:.2f}s")

        # Render and save video
        print("    Rendering video...", end=" ")
        render_start = time.time()
        frames = env.render(states, fps=video_fps)

        video_path = output_dir / f"episode_{ep + 1:03d}.mp4"
        imageio.mimsave(
            video_path,
            [frame.transpose(1, 2, 0) for frame in frames],  # C,H,W -> H,W,C
            fps=video_fps,
        )
        render_time = time.time() - render_start
        print(f"saved to {video_path} ({render_time:.2f}s)")

    print(f"\nAverage cost: {np.mean(costs):.4f} ± {np.std(costs):.4f}")
    print(f"Videos saved to: {output_dir.absolute()}")
