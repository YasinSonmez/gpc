import jax
import jax.numpy as jnp
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


class ReplayBuffer:
    """A unified replay buffer for GPC and RL expert data.
    
    Stores both single-step transitions (for value functions/IQL) and 
    multi-step action sequences (for generative policies/GPC).
    """

    def __init__(
        self,
        capacity: int,
        observation_size: int,
        action_size: int,
        horizon: int,
    ):
        self.capacity = capacity
        self.observation_size = observation_size
        self.action_size = action_size
        self.horizon = horizon

        # Single-step transition data
        self.obs = np.zeros((capacity, observation_size), dtype=np.float32)
        self.next_obs = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros((capacity, action_size), dtype=np.float32)
        self.costs = np.zeros((capacity,), dtype=np.float32)
        self.returns_to_go = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        # Sequence data (optimized plans and initial guesses)
        # Shape: (capacity, horizon, action_size)
        self.u_sequences = np.zeros((capacity, horizon, action_size), dtype=np.float32)
        self.u_prev_sequences = np.zeros((capacity, horizon, action_size), dtype=np.float32)

        self.size = 0
        self.ptr = 0
        
        # Running statistics for cost normalization (GPC specific)
        self._cost_mean = 0.0
        self._cost_std = 1.0
        self._cost_min = float('inf')

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        cost: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        u_sequences: Optional[np.ndarray] = None,
        u_prev_sequences: Optional[np.ndarray] = None,
        discount: float = 0.99,
    ):
        """Add transitions and optional sequences to the buffer.
        
        Args:
            obs: (N, obs_dim) or (Envs, T, obs_dim)
            action: (N, act_dim) or (Envs, T, act_dim)
            cost: (N,) or (Envs, T) or (Envs,)
            next_obs: (N, obs_dim) or (Envs, T, obs_dim)
            done: (N,) or (Envs, T)
            u_sequences: (N, horizon, act_dim) or (Envs, T, horizon, act_dim)
            u_prev_sequences: (N, horizon, act_dim)
            discount: Discount factor for returns-to-go.
        """
        # Convert to numpy and ensure 2D/3D
        obs = np.asarray(obs)
        action = np.asarray(action)
        cost = np.asarray(cost)
        next_obs = np.asarray(next_obs)
        done = np.asarray(done)

        if obs.ndim == 3: # (Envs, T, obs_dim)
            num_envs, T = obs.shape[:2]
            obs = obs.reshape(-1, obs.shape[-1])
            action = action.reshape(-1, action.shape[-1])
            next_obs = next_obs.reshape(-1, next_obs.shape[-1])
            done = done.reshape(-1)
            
            if cost.ndim == 2:
                # Per-step costs
                cost = cost.reshape(-1)
            elif cost.ndim == 1:
                # Per-episode costs, broadcast to timesteps
                cost = np.broadcast_to(cost[:, None], (num_envs, T)).reshape(-1)
            
            if u_sequences is not None:
                u_sequences = np.asarray(u_sequences).reshape(-1, u_sequences.shape[-2], u_sequences.shape[-1])
            if u_prev_sequences is not None:
                u_prev_sequences = np.asarray(u_prev_sequences).reshape(-1, u_prev_sequences.shape[-2], u_prev_sequences.shape[-1])
        
        num_transitions = obs.shape[0]
        
        if num_transitions > self.capacity:
            # Slicing for safety
            start = num_transitions - self.capacity
            obs = obs[start:]
            action = action[start:]
            cost = cost[start:]
            next_obs = next_obs[start:]
            done = done[start:]
            if u_sequences is not None: u_sequences = u_sequences[start:]
            if u_prev_sequences is not None: u_prev_sequences = u_prev_sequences[start:]
            num_transitions = self.capacity

        # Compute returns-to-go
        returns = np.zeros(num_transitions, dtype=np.float32)
        running_return = 0.0
        for t in range(num_transitions - 1, -1, -1):
            running_return = cost[t] + discount * running_return * (1.0 - done[t])
            returns[t] = running_return

        # Indices logic
        idx1 = self.ptr
        idx2 = (self.ptr + num_transitions) % self.capacity

        def write_to_buffer(dest, src):
            if idx1 < idx2:
                dest[idx1:idx2] = src
            else:
                wrap_part = self.capacity - idx1
                dest[idx1:] = src[:wrap_part]
                dest[:idx2] = src[wrap_part:]

        write_to_buffer(self.obs, obs)
        write_to_buffer(self.next_obs, next_obs)
        write_to_buffer(self.actions, action)
        write_to_buffer(self.costs, cost)
        write_to_buffer(self.returns_to_go, returns)
        write_to_buffer(self.dones, done)

        if u_sequences is not None:
            write_to_buffer(self.u_sequences, u_sequences)
        else:
            # Broadcast single action to horizon if missing
            # (better than nothing for expert cloning)
            dummy_seq = np.broadcast_to(action[:, None, :], (num_transitions, self.horizon, self.action_size))
            write_to_buffer(self.u_sequences, dummy_seq)

        if u_prev_sequences is not None:
            write_to_buffer(self.u_prev_sequences, u_prev_sequences)
        else:
            # If no guess provided, use the sequence itself (weight ~0.13) 
            # or zeros. Let's use the sequence to be consistent with current code.
            if u_sequences is not None:
                write_to_buffer(self.u_prev_sequences, u_sequences)
            else:
               dummy_seq = np.broadcast_to(action[:, None, :], (num_transitions, self.horizon, self.action_size))
               write_to_buffer(self.u_prev_sequences, dummy_seq)

        self.ptr = (self.ptr + num_transitions) % self.capacity
        self.size = min(self.size + num_transitions, self.capacity)
        
        # Update cost statistics
        current_costs = self.costs[:self.size]
        self._cost_mean = float(np.mean(current_costs))
        self._cost_std = float(np.std(current_costs) + 1e-8)
        self._cost_min = float(np.min(current_costs))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch for Policy training (GPC)."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # GPC cost normalization (match cost_conditioned.py)
        # Normalize costs to [0, 1] range (0 = best, 1 = worst)
        all_costs = self.costs[indices]
        normalized_costs = (all_costs - self._cost_min) / (self._cost_std * 3 + 1e-8)
        normalized_costs = np.clip(normalized_costs, 0.0, 1.0)

        return {
            "obs": self.obs[indices],
            "actions": self.u_sequences[indices],
            "old_actions": self.u_prev_sequences[indices],
            "costs": normalized_costs,
        }

    def sample_n_step(
        self,
        batch_size: int,
        n: int,
        discount: float,
    ) -> Dict[str, np.ndarray]:
        """Sample n-step transitions for Value training (IQL)."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch_obs = self.obs[indices]
        batch_target_q = np.zeros(batch_size, dtype=np.float32)
        batch_last_obs = np.zeros((batch_size, self.observation_size), dtype=np.float32)
        batch_last_done = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            idx = indices[i]
            # Find n-step end, respecting episode boundaries (dones)
            step_idx = idx
            for _ in range(n):
                if self.dones[step_idx]:
                    break
                step_idx = (step_idx + 1) % self.capacity
                if step_idx >= self.size: # Handle case when buffer is not full
                    step_idx = (step_idx - 1) % self.capacity
                    break
            
            # Use pre-computed returns-to-go for n-step return
            # V(s_t) = r_t + y r_{t+1} + ... + y^n V(s_{t+n})
            # G(s_t) - y^n G(s_{t+n})
            batch_target_q[i] = self.returns_to_go[idx]
            batch_last_obs[i] = self.next_obs[step_idx]
            batch_last_done[i] = self.dones[step_idx]
            
            if not self.dones[step_idx] and step_idx != idx:
                # Need to handle circular distance
                actual_steps = (step_idx - idx) % self.capacity
                batch_target_q[i] -= (discount ** actual_steps) * self.returns_to_go[step_idx]

        return {
            "obs": batch_obs,
            "target_q": batch_target_q,
            "last_obs": batch_last_obs,
            "last_done": batch_last_done,
        }

    def sample_n_step_all(
        self,
        n: int,
        discount: float,
    ) -> Dict[str, np.ndarray]:
        """Vectorized n-step sampling for ALL transitions in the buffer.
        
        This avoids the Python loop overhead of sample_n_step.
        """
        indices = np.arange(self.size)
        
        # Calculate terminal indices (where done is True)
        done_indices = np.where(self.dones[:self.size])[0]
        
        # For each transition i, find the end of the n-step sequence.
        # This is min(i+n, next_done_index_after_i)
        
        # Precompute next done index for every position
        next_done_idx = np.full(self.size, self.size - 1)
        if len(done_indices) > 0:
            # For each done_idx, it's the 'next done' for all indices <= it
            # We can use searchsorted to find the first done index >= i
            done_positions = np.searchsorted(done_indices, indices)
            # Clip positions to valid range
            done_positions = np.minimum(done_positions, len(done_indices) - 1)
            next_done_idx = done_indices[done_positions]
            
        # step_idx = min(i + n, next_done_idx)
        step_idx = np.minimum(indices + n, next_done_idx)
        
        # Clip to size for safety
        step_idx = np.minimum(step_idx, self.size - 1)
        
        # Efficiently compute (returns[idx] - gamma^actual_steps * returns[step_idx])
        actual_steps = step_idx - indices
        
        batch_obs = self.obs[indices]
        batch_target_q = self.returns_to_go[indices].copy()
        
        # If we haven't reached a terminal state (or the end of the buffer),
        # subtract the discounted return of the n-step future state.
        mask = (step_idx != indices) & (self.dones[step_idx] == 0)
        batch_target_q[mask] -= (discount ** actual_steps[mask]) * self.returns_to_go[step_idx][mask]
        
        return {
            "obs": batch_obs,
            "target_q": batch_target_q,
            "last_obs": self.next_obs[step_idx],
            "last_done": self.dones[step_idx],
        }

    def get_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the primary data for mass fitting as a 4-tuple."""
        indices = np.arange(self.size)
        normalized_costs = (self.costs[indices] - self._cost_min) / (self._cost_std * 3 + 1e-8)
        normalized_costs = np.clip(normalized_costs, 0.0, 1.0)
        
        return (
            self.obs[indices],
            self.u_sequences[indices],
            self.u_prev_sequences[indices],
            normalized_costs,
        )

    def export_all(self) -> Dict[str, np.ndarray]:
        """Get all data in the buffer for saving or advanced processing."""
        indices = np.arange(self.size)
        return {
            "observations": self.obs[indices],
            "actions": self.u_sequences[indices],
            "old_actions": self.u_prev_sequences[indices],
            "costs": self.costs[indices],
            "single_actions": self.actions[indices],
            "next_observations": self.next_obs[indices],
            "dones": self.dones[indices],
        }

    def save(self, path: Union[Path, str]):
        """Save buffer as a pickle file."""
        data = self.export_all()
        data['meta'] = {
            'observation_size': self.observation_size,
            'action_size': self.action_size,
            'horizon': self.horizon,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Union[Path, str], capacity: int = 1_000_000) -> "ReplayBuffer":
        """Load buffer from a pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        meta = data.get('meta', {})
        obs_size = meta.get('observation_size', data['observations'].shape[-1])
        act_size = meta.get('action_size', data['single_actions'].shape[-1] if 'single_actions' in data else data['actions'].shape[-1])
        horizon = meta.get('horizon', data['actions'].shape[-2] if data['actions'].ndim == 3 else 1)
        
        buffer = cls(capacity, obs_size, act_size, horizon)
        
        # Load data. Key mapping for compatibility.
        obs = data['observations']
        u_seq = data['actions']
        u_prev_seq = data.get('old_actions', u_seq)
        single_act = data.get('single_actions', u_seq[:, 0] if u_seq.ndim == 3 else u_seq)
        costs = data['costs']
        next_obs = data['next_observations']
        dones = data['dones']
        
        buffer.add(
            obs, single_act, costs, next_obs, dones,
            u_sequences=u_seq, u_prev_sequences=u_prev_seq
        )
        return buffer

    @property
    def target_cost_normalized(self) -> float:
        return 0.0

# For compatibility
TrajectoryReplayBuffer = ReplayBuffer
