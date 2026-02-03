import os

# Set MUJOCO_GL to egl for headless rendering to avoid GLFW conflicts
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'

import argparse
import functools
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
import yaml
import wandb
import imageio
from brax.io import image, html, model

from brax import envs
from brax.training.agents.sac import train as sac
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Transition

# Register GPC envs
from gpc import envs as gpc_envs
from gpc.replay_buffer import ReplayBuffer

# Monkey-patch HumanoidStandup to fix orientation cost bug
# The original cost was jnp.sum(jnp.square(orientation)), which is constant 1.0
# because orientation is a rotated unit vector [0, 0, 1].
from hydrax.tasks.humanoid_standup import HumanoidStandup
def patched_running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
    current_up = self._get_torso_orientation(state)
    target_up = jnp.array([0.0, 0.0, 1.0])
    orientation_cost = jnp.sum(jnp.square(current_up - target_up))
    
    height_cost = jnp.square(self._get_torso_height(state) - self.target_height)
    nominal_cost = jnp.sum(jnp.square(state.qpos[7:] - self.qstand[7:]))
    # Note: we use the same weights as original task
    return 10.0 * orientation_cost + 10.0 * height_cost + 0.1 * nominal_cost

HumanoidStandup.running_cost = patched_running_cost


def render_mjx_trajectory(sys: Any, trajectory: Any, width: int = 640, height: int = 480) -> np.ndarray:
    """Render a Brax MJX trajectory using standard MuJoCo renderer with tracking."""
    if not hasattr(sys, 'mj_model'):
        raise ValueError("System must have mj_model for MuJoCo rendering")
        
    mj_model = sys.mj_model
    renderer = mujoco.Renderer(mj_model, width=width, height=height)
    
    # improved aesthetics
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    
    # Configure tracking camera
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    
    # Try to track torso/pelvis (usually body 1 for humanoids)
    # Brax humanoid usually has 'torso'
    track_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    if track_id < 0:
        track_id = 1
        
    cam.trackbodyid = track_id
    cam.distance = 3.5
    cam.elevation = -15.0
    cam.azimuth = 90.0
    
    frames = []
    
    # Convert trajectory (list of States or batched State) to list of mjx.Data
    # If passed as a list of pipeline states:
    if isinstance(trajectory, list):
        data_list = trajectory
    elif hasattr(trajectory, 'q'): # It's a batched state
        # This path handles the default Brax State object which has batch dim for time
        # We need to unstack it. But record_expert_video passes list of PipelineStates (mjx.Data)
        # correction: record_expert_video passes [s.pipeline_state for s in trajectory] which is list of mjx.Data
        data_list = trajectory
    else:
        raise ValueError(f"Unknown trajectory type: {type(trajectory)}")

    for mjx_data in data_list:
        # Transfer data to host
        mj_data = mjx.get_data(mj_model, mjx_data)
        
        # Update renderer
        renderer.update_scene(mj_data, camera=cam)
        pixels = renderer.render()
        frames.append(pixels)
        
    return np.stack(frames)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def progress(num_steps, metrics):
    # Extract training metrics
    train_reward = metrics.get('training/episode_reward')
    actor_loss = metrics.get('training/actor_loss')
    critic_loss = metrics.get('training/critic_loss')
    alpha_loss = metrics.get('training/alpha_loss')
    alpha = metrics.get('training/alpha')
    entropy = metrics.get('training/entropy')
    
    # Extract eval metrics (if any)
    eval_reward = metrics.get('eval/episode_reward')
    
    log_str = f"Step {num_steps}:"
    if train_reward is not None:
        log_str += f" train/reward={train_reward:.2f} (cost={-train_reward:.2f})"
    if actor_loss is not None:
        log_str += f" actor_loss={actor_loss:.4f}"
    if critic_loss is not None:
        log_str += f" critic_loss={critic_loss:.4f}"
    if alpha is not None:
        log_str += f" alpha={alpha:.4f}"
    if entropy is not None:
        log_str += f" entropy={entropy:.2f}"
    if eval_reward is not None:
        log_str += f" eval/reward={eval_reward:.2f}"
    
    print(log_str)
    if wandb.run is not None:
        wandb.log(metrics, step=num_steps)

class GPCtoBraxWrapper(envs.Env):
  """Wrapper to make GPC TrainingEnv compatible with Brax Env interface."""
  def __init__(self, gpc_env):
    self.gpc_env = gpc_env
  
  def reset(self, rng: jax.Array) -> envs.State:
    # GPC init_state returns SimulatorState(data, t, rng)
    sim_state = self.gpc_env.init_state(rng)
    obs = self.gpc_env.get_obs(sim_state.data)
    # Brax State: (obs, reward, done, metrics, info, pipeline_state)
    return envs.State(
        obs=obs,
        reward=jnp.zeros(()),
        done=jnp.zeros(()),
        metrics={},
        info={},
        pipeline_state=sim_state
    )

  def step(self, state: envs.State, action: jax.Array) -> envs.State:
    # state.pipeline_state is SimulatorState
    sim_state = state.pipeline_state
    
    # We use mjx.step directly to avoid the auto-reset in gpc_env.step
    # but we still want the t increment and data update.
    next_data = mjx.step(self.gpc_env.task.model, sim_state.data.replace(ctrl=action))
    next_sim_state = sim_state.replace(data=next_data, t=sim_state.t + 1)
    
    obs = self.gpc_env.get_obs(next_sim_state.data)
    
    # Check health for early termination if available
    is_healthy_fn = getattr(self.gpc_env.task, '_is_healthy', None)
    if is_healthy_fn:
        is_healthy = is_healthy_fn(next_sim_state.data)
    else:
        is_healthy = jnp.array(True)
    
    # Calculate reward. reward = -cost.
    # Note: we use the task's running_cost directly.
    cost = self.gpc_env.task.running_cost(next_sim_state.data, action)
    reward = -cost
    
    # Terminate if unhealthy or time limit reached
    # SAC needs to know if it's a "real" termination (masking bootstrap) or truncation.
    # Brax SAC uses the 'done' signal. For simplicity, we signal done on both.
    done = jnp.where(~is_healthy | self.gpc_env.episode_over(next_sim_state), 1.0, 0.0)
    
    return state.replace(
        obs=obs,
        reward=reward,
        done=done,
        pipeline_state=next_sim_state
    )

  @property
  def observation_size(self) -> int:
    return self.gpc_env.observation_size

  @property
  def action_size(self) -> int:
    return self.gpc_env.task.model.nu

  @property
  def backend(self) -> str:
    return 'mjx'

  @property
  def dt(self) -> float:
    return self.gpc_env.task.dt
  
  @property
  def sys(self):
    # Brax render functions (like brax.io.image) expect .sys to be an object 
    # with an .mj_model attribute for MJX environments.
    return self.gpc_env.task

  def render(self, trajectory: Any, **kwargs):
      return self.gpc_env.render(trajectory, **kwargs)

def run_training(config: Dict[str, Any]):
    env_name = config['task_name']
    
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'gpc_rl_expert'),
            config=config,
            name=f"{env_name}_expert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Create environment
    if env_name == "walker_gym":
        gpc_env = gpc_envs.WalkerGymEnv(
            episode_length=config['episode_length'],
            render_camera=config.get('render_camera', -1)
        )
        train_env = GPCtoBraxWrapper(gpc_env)
    elif env_name == "humanoid":
        gpc_env = gpc_envs.HumanoidEnv(
            episode_length=config['episode_length'],
            render_camera=config.get('render_camera', -1)
        )
        train_env = GPCtoBraxWrapper(gpc_env)
    elif env_name == "humanoid_mocap":
        gpc_env = gpc_envs.HumanoidMocapEnv(
            episode_length=config['episode_length'],
            render_camera=config.get('render_camera', -1)
        )
        train_env = GPCtoBraxWrapper(gpc_env)
    elif env_name == "humanoid_brax":
        train_env = envs.create("humanoid", backend='mjx')
    else:
        # Fallback to brax registry
        train_env = envs.create(env_name, backend='jax')

    # Training function with SAC or PPO
    if not config.get('render_only', False):
        print(f"Starting training on {env_name}...")
        
        # PPO path
        if config.get('algorithm', 'sac').lower() == 'ppo' or 'ppo' in config.get('output_dir', ''):
             print("Using PPO Algorithm...")
             network_factory = functools.partial(
                ppo_networks.make_ppo_networks,
                policy_hidden_layer_sizes=config.get('hidden_layers', [512, 512, 512])
             )
             
             make_inference_fn, params, _ = ppo.train(
                environment=train_env,
                num_timesteps=int(config['total_timesteps']),
                episode_length=int(config['episode_length']),
                normalize_observations=bool(config['normalize_observations']),
                action_repeat=int(config['action_repeat']),
                entropy_cost=float(config.get('entropy_cost', 1e-2)),
                discounting=float(config['discounting']),
                learning_rate=float(config['learning_rate']),
                num_envs=int(config['num_envs']),
                batch_size=int(config['batch_size']),
                num_minibatches=int(config.get('num_minibatches', 32)),
                num_updates_per_batch=int(config.get('num_updates_per_batch', 4)),
                unroll_length=int(config.get('unroll_length', 20)),
                max_devices_per_host=config.get('max_devices_per_host'),
                num_evals=int(config['num_evals']),
                reward_scaling=float(config['reward_scaling']),
                seed=int(config['seed']),
                network_factory=network_factory,
                progress_fn=progress
             )

        else:
             # SAC Path
             print("Using SAC Algorithm...")
             # Create network factory with custom architecture if specified
             network_factory = functools.partial(
                sac_networks.make_sac_networks,
                hidden_layer_sizes=config.get('hidden_layers', [512, 512, 512])
             )

             make_inference_fn, params, _ = sac.train(
                environment=train_env,
                num_timesteps=int(config['total_timesteps']),
                episode_length=int(config['episode_length']),
                normalize_observations=bool(config['normalize_observations']),
                action_repeat=int(config['action_repeat']),
                discounting=float(config['discounting']),
                learning_rate=float(config['learning_rate']),
                num_envs=int(config['num_envs']),
                batch_size=int(config['batch_size']),
                grad_updates_per_step=int(config['grad_updates_per_step']),
                max_devices_per_host=config.get('max_devices_per_host'),
                max_replay_size=int(config['max_replay_size']),
                min_replay_size=int(config['min_replay_size']),
                num_evals=int(config['num_evals']),
                reward_scaling=float(config['reward_scaling']),
                seed=int(config['seed']),
                network_factory=network_factory,
                progress_fn=progress
             )
        # Save parameters
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        params_path = output_dir / "expert_params.pkl"
        model.save_params(params_path, params)
        print(f"Saved expert params to {params_path}")
    else:
        print("Skipping training, loading existing params for rendering/recording...")
        output_dir = Path(config['output_dir'])
        params_path = output_dir / "expert_params.pkl"
        params = model.load_params(params_path)
        
        # Determine obs/act size for network factory
        # We'll create the env anyway to get its properties
        environment = config['task_name']
        if environment == "walker_gym":
            gpc_env = gpc_envs.WalkerGymEnv(episode_length=config['episode_length'])
            train_env = GPCtoBraxWrapper(gpc_env)
        elif environment == "humanoid":
             gpc_env = gpc_envs.HumanoidEnv(episode_length=config['episode_length'])
             train_env = GPCtoBraxWrapper(gpc_env)
        elif environment == "humanoid_mocap":
             gpc_env = gpc_envs.HumanoidMocapEnv(episode_length=config['episode_length'])
             train_env = GPCtoBraxWrapper(gpc_env)
        elif environment == "humanoid_brax":
             train_env = envs.create("humanoid", backend='mjx')
        else:
             train_env = envs.create(environment, backend='jax')

        if config.get('algorithm', 'sac').lower() == 'ppo' or 'ppo' in config.get('output_dir', ''):
            ppo_network = ppo_networks.make_ppo_networks(
                train_env.observation_size,
                train_env.action_size,
                policy_hidden_layer_sizes=config.get('hidden_layers', [512, 512, 512])
            )
            make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
        else:
            sac_network = sac_networks.make_sac_networks(
                train_env.observation_size,
                train_env.action_size,
                hidden_layer_sizes=config.get('hidden_layers', [512, 512, 512])
            )
            make_inference_fn = sac_networks.make_inference_fn(sac_network)
    
    # --- Final Evaluation (128 episodes) ---
    print("\nRunning final evaluation (128 episodes)...")
    final_eval_reward = run_final_evaluation(train_env, make_inference_fn, params, config)
    print(f"Final Evaluation Reward: {final_eval_reward:.2f} (Cost: {-final_eval_reward:.2f})")
    if wandb.run is not None:
        wandb.log({"final_eval/reward": final_eval_reward, "final_eval/cost": -final_eval_reward})
    
    # --- Record Expert Buffer ---
    if config.get('record_buffer', False):
        print("Recording expert replay buffer...", flush=True)
        record_expert_buffer(train_env, make_inference_fn, params, config, output_dir)
        print("Expert replay buffer recording SUCCESS.", flush=True)

    # --- Record Expert Video ---
    if config.get('record_video', True):
        print("Recording expert video...")
        try:
            record_expert_video(train_env, make_inference_fn, params, config, output_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error recording expert video: {e}. Skipping video recording.")

def run_final_evaluation(env, make_inference_fn, params, config):
    num_eval_envs = 128
    episode_length = config['episode_length']
    
    from brax.envs.wrappers import training as training_wrappers
    eval_env = training_wrappers.VmapWrapper(env)
    
    reset_fn = jax.jit(eval_env.reset)
    step_fn = jax.jit(eval_env.step)
    inference_fn = make_inference_fn(params)
    vmap_inference_fn = jax.vmap(inference_fn)
    jit_inference_fn = jax.jit(vmap_inference_fn)
    
    rng = jax.random.PRNGKey(config['seed'] + 42)
    reset_rngs = jax.random.split(rng, num_eval_envs)
    state = reset_fn(reset_rngs)
    
    total_reward = jnp.zeros(num_eval_envs)
    active_mask = jnp.ones(num_eval_envs)
    
    for _ in range(episode_length):
        rng, act_rng = jax.random.split(rng)
        act_rngs = jax.random.split(act_rng, num_eval_envs)
        act, _ = jit_inference_fn(state.obs, act_rngs)
        state = step_fn(state, act)
        
        # Accumulate reward only for active environments
        total_reward += state.reward * active_mask
        # Update mask: if done once, stay done
        active_mask *= (1.0 - state.done)
        
    return float(jnp.mean(total_reward))

def record_expert_buffer(env, make_inference_fn, params, config, output_dir):
    buffer_size = config['buffer_size']
    num_envs = config.get('buffer_num_envs', 1)
    episode_length = config['episode_length']
    
    # GPC-specific knot/horizon parameters
    num_knots = config.get('num_knots', 4)
    plan_horizon = config.get('plan_horizon', 0.5)
    
    # Use actual env dt
    if hasattr(env, 'dt'):
         sim_dt = env.dt
    elif hasattr(env, 'sys') and hasattr(env.sys, 'dt'):
         sim_dt = env.sys.dt
    elif hasattr(env, 'sys') and hasattr(env.sys, 'opt'):
         sim_dt = env.sys.opt.timestep
    else:
         # Fallback if unwrapped
         sim_dt = 0.01 
    
    plan_steps = int(round(plan_horizon / sim_dt))
    
    # Vectorize for faster rollout
    from brax.envs.wrappers import training as training_wrappers
    env = training_wrappers.VmapWrapper(env)
    
    init_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    
    rng = jax.random.PRNGKey(config['seed'] + 123)
    
    total_transitions = 0
    total_transitions = 0
    
    print(f"Recording {buffer_size} transitions via full episodes...")
    
    # Initialize unified buffer
    from gpc.replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(
        capacity=buffer_size,
        observation_size=env.observation_size,
        action_size=env.action_size,
        horizon=num_knots
    )
    
    while total_transitions < buffer_size:
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, num_envs)
        state = init_fn(reset_rngs)
        
        ep_obs = []
        ep_act = []
        ep_cost = []
        ep_done = []
        ep_next_obs = []
        
        for _ in range(episode_length):
            rng, act_rng = jax.random.split(rng)
            act, _ = jit_inference_fn(state.obs, act_rng)
            
            next_state = step_fn(state, act)
            
            ep_obs.append(state.obs)
            ep_act.append(act)
            ep_cost.append(-next_state.reward) # reward = -cost
            ep_done.append(next_state.done)
            ep_next_obs.append(next_state.obs)
            
            state = next_state
        
        # Convert to arrays: (T, Envs, ...)
        ep_obs = jnp.stack(ep_obs)
        ep_act = jnp.stack(ep_act)
        ep_cost = jnp.stack(ep_cost)
        ep_done = jnp.stack(ep_done)
        ep_next_obs = jnp.stack(ep_next_obs)
        
        # episode_length is T. we want to mask transitions after done
        # (T, Envs)
        cum_done = jnp.cumsum(ep_done, axis=0)
        # valid_mask is 1 if no done has happened yet in this env (including the current step that just finished)
        # Actually, the transition that makes done=1 is the LAST valid transition.
        # So we want mask = (cum_done <= 1) and (the previous cum_done was 0)
        mask = (cum_done <= 1.0).astype(jnp.float32)
        
        # Calculate knot sequences for this episode
        knot_indices = jnp.linspace(0, plan_steps, num_knots, endpoint=True).astype(jnp.int32)
        
        # (T, Envs, num_knots, nu)
        u_knots = []
        for t in range(episode_length):
            future_indices = (t + knot_indices)
            safe_indices = jnp.clip(future_indices, 0, episode_length - 1)
            knots = ep_act[safe_indices] # (num_knots, Envs, nu)
            u_knots.append(knots.transpose(1, 0, 2)) # (Envs, num_knots, nu)
        
        u_knots = jnp.stack(u_knots) # (T, Envs, num_knots, nu)
        
        # Add to unified buffer
        # Masked items should be removed or handled. 
        # For simplicity, we can flatten and filter by mask before adding to buffer
        # but ReplayBuffer.add expects (Batch, T, ...)
        # Re-applying mask to costs
        ep_cost = ep_cost * mask
        
        # Transpose to (Envs, T, ...)
        buffer.add(
            obs=ep_obs.transpose(1, 0, 2),
            action=ep_act.transpose(1, 0, 2),
            cost=ep_cost.transpose(1, 0),
            next_obs=ep_next_obs.transpose(1, 0, 2),
            done=ep_done.transpose(1, 0),
            u_sequences=u_knots.transpose(1, 0, 2, 3)
        )
        
        total_transitions += num_envs * episode_length
        print(f"Recorded {total_transitions}/{buffer_size} transitions...", end='\r')

    # Save using the unified buffer's save method
    buffer_path = output_dir / config['buffer_output_file']
    buffer.save(buffer_path)
    print(f"\nSaved expert buffer to {buffer_path}")

def record_expert_video(env, make_inference_fn, params, config, output_dir):
    """Render a video of the trained policy."""
    # The inference_fn handles observation normalization internally if enabled.

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    
    rng = jax.random.PRNGKey(config['seed'] + 999)
    state = jit_reset(rng)
    
    trajectory = []
    
    # Rollout
    for _ in range(config['episode_length']):
        trajectory.append(state)
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, act)

    # Convert list of States to a single State with a time dimension
    # (T, obs_dim), etc.
    trajectory_state = jax.tree.map(lambda *xs: jnp.stack(xs), *trajectory)
    # Extracts the SimulatorState trajectory
    sim_trajectory = trajectory_state.pipeline_state

    try:
        # sim_trajectory is SimulatorState
        if hasattr(env, 'gpc_env'):
             frames = env.gpc_env.render(sim_trajectory, fps=30)
             # GPC frames are (T, C, H, W), transpose to (T, H, W, C)
             if frames.ndim == 4 and frames.shape[1] in [1, 3]:
                 frames = frames.transpose(0, 2, 3, 1)
        else:
             # Native Brax env - prefer MuJoCo renderer if available (for tracking/quality)
             sys = getattr(env, 'sys', None)
             if sys is None:
                 raise AttributeError("Environment missing 'sys' for rendering.")
             
             # Convert trajectory states to pipeline states list
             pipeline_states = [s.pipeline_state for s in trajectory]
             
             if hasattr(sys, 'mj_model'):
                try:
                    print("Using native MuJoCo renderer with tracking...")
                    frames = render_mjx_trajectory(sys, pipeline_states)
                except Exception as e:
                    print(f"Native MuJoCo render failed ({e}), falling back to Brax render_array...")
                    frames = image.render_array(sys, pipeline_states, width=640, height=480)
             else:
                try:
                    # Prefer render_array (modern Brax) to get numpy arrays
                    frames = image.render_array(sys, pipeline_states, width=640, height=480)
                except (AttributeError, Exception):
                    frames = image.render(sys, pipeline_states, width=640, height=480)
    except Exception as e:
        print(f"Rendering failed: {e}. Skipping video.", flush=True)
        return
        
    # Verify frame type and save
    video_filename = config.get('video_output_file', 'expert.mp4')
    if not video_filename.endswith(('.mp4', '.gif')):
        video_filename += '.mp4'
    video_path = output_dir / video_filename
    if isinstance(frames, list):
         # Brax image.render returns list of bytes (GIF frames) usually, or we can use generic processing
         # For simplicity, if it's bytes (GIF), save as gif. If arrays, save as mp4.
         pass # Brax image.render usually returns bytes representing the GIF content directly if format='gif'
         # But image.render default returns a list of RGB arrays? No, it returns bytes if fmt='gif'.
         # Let's re-read brax.io.image in plan or assume standard behavior.
         # Standard behavior: image.render(sys, qps, ...) -> bytes (gif)
         
         # Retrying with explicit render_array if available
         pass

    # Use render_array if possible to get numpy arrays for video
    try:
        # Recent Brax versions have render_array
         frames = image.render_array(env.sys, [s.pipeline_state for s in trajectory], width=640, height=480)
    except Exception:
        # Fallback: image.render returns bytes (gif)
        pass

    # If frames is bytes, write to gif
    if isinstance(frames, bytes):
        gif_path = output_dir / video_filename.replace('.mp4', '.gif')
        with open(gif_path, "wb") as f:
            f.write(frames)
        print(f"Saved expert video to {gif_path}")
    else:
        # Assume numpy array
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Saved expert video to {video_path}")
        
    if config.get('use_wandb', False):
        if str(video_path).endswith('mp4'):
            wandb.log({"expert_video": wandb.Video(str(video_path), fps=30, format="mp4")})
        elif str(video_path).endswith('gif'):
             wandb.log({"expert_video": wandb.Video(str(video_path), fps=30, format="gif")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config yaml')
    parser.add_argument('--render-only', action='store_true', help='Only render/record, skip training')
    parser.add_argument('--total-timesteps', type=int, default=None, help='Override total_timesteps (for quick smoke test)')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.render_only:
        config['render_only'] = True
    if args.total_timesteps is not None:
        config['total_timesteps'] = args.total_timesteps
        print(f"Overriding total_timesteps to {args.total_timesteps}")
    run_training(config)
