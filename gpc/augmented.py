import jax
import jax.numpy as jnp
from flax.struct import dataclass
from hydrax.alg_base import SamplingBasedController, Trajectory
from hydrax.task_base import Task
from mujoco import mjx
from typing import Callable, Any, Optional, Tuple


@dataclass
class PACParams:
    """Parameters for the policy-augmented controller.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        base_params: The parameters for the base controller.
        policy_samples: Control knots sampled from the policy.
        rng: Random number generator key for domain randomization.
        prev_elites: Previous step's elite samples for warm-starting (iCEM).
    """

    tk: jax.Array
    mean: jax.Array
    base_params: Any
    policy_samples: jax.Array
    rng: jax.Array
    prev_elites: Optional[jax.Array] = None  # (num_elites, num_knots, nu)
    value_params: Optional[Any] = None       # Parameters for the Value function
    value_alpha: float = 0.0                 # Weight for the terminal value function
    target_cost: Optional[float] = None      # Target cost for CFG
    cfg_scale: float = 1.0                   # Guidance scale for CFG


class ValueAugmentedTask(Task):
    """A task wrapper that adds a learned value function to the terminal cost."""

    def __init__(
        self, 
        base_task: Task, 
        value_fn: Callable[[mjx.Data], jax.Array], 
        alpha: float,
        use_base_terminal_cost: bool = True,
    ):
        self.base_task = base_task
        self.value_fn = value_fn
        self.alpha = alpha
        self.use_base_terminal_cost = use_base_terminal_cost
        
        # Essential attributes
        self.model = base_task.model
        self.u_min = base_task.u_min
        self.u_max = base_task.u_max
        
    def __getattr__(self, name):
        """Delegate missing attributes to the base task."""
        return getattr(self.base_task, name)

    def running_cost(self, data: mjx.Data, u: jax.Array) -> jax.Array:
        return self.base_task.running_cost(data, u)

    def terminal_cost(self, data: mjx.Data) -> jax.Array:
        term_cost = self.alpha * self.value_fn(data)
        if self.use_base_terminal_cost:
            term_cost = term_cost + self.base_task.terminal_cost(data)
        return term_cost


class PolicyAugmentedController(SamplingBasedController):
    """An SPC generalization where samples are augmented by a learned policy.
    
    Supports iCEM improvements:
    - Colored noise: Temporally correlated noise for smoother actions
    - Elite shifting: Warm-start from previous step's best samples
    - Value Function: Optional terminal cost bootstrapping
    """

    def __init__(
        self,
        base_ctrl: SamplingBasedController,
        num_policy_samples: int,
        use_colored_noise: bool = False,
        noise_beta: float = 2.0,
        shift_elites_fraction: float = 0.0,
        value_fn: Optional[Callable[[Any, mjx.Data], jax.Array]] = None,
        use_task_terminal_cost: bool = True,
        exploration_floor: float = 0.0,
    ) -> None:
        """Initialize the policy-augmented controller.

        Args:
            base_ctrl: The base controller to augment.
            num_policy_samples: The number of samples to draw from the policy.
            use_colored_noise: Use temporally correlated noise (iCEM).
            noise_beta: Colored noise exponent (0=white, 2+=smooth).
            shift_elites_fraction: Fraction of elites to warm-start between steps.
            value_fn: Optional terminal value function V(params, mjx_data).
            exploration_floor: Fraction of the base SPC sample budget (num_samples)
                replaced with Uniform[-1, 1] random knots each step (Solution 1).
                0.0 (default) preserves the original behaviour.
        """
        self.base_ctrl = base_ctrl
        self.num_policy_samples = num_policy_samples
        self.use_colored_noise = use_colored_noise
        self.noise_beta = noise_beta
        self.shift_elites_fraction = shift_elites_fraction
        self.value_fn = value_fn
        self.use_task_terminal_cost = use_task_terminal_cost
        # Number of base samples replaced with Uniform[-1,1] wide noise (Solution 1).
        self.n_wide: int = max(0, int(base_ctrl.num_samples * exploration_floor))
        
        # Calculate number of shifted elites
        if hasattr(base_ctrl, 'num_elites'):
            self.num_shift_elites = int(base_ctrl.num_elites * shift_elites_fraction)
        else:
            self.num_shift_elites = 0
        
        # Pass through all parameters from base controller
        super().__init__(
            base_ctrl.task,
            base_ctrl.num_randomizations,
            base_ctrl.risk_strategy,
            seed=0,
            plan_horizon=base_ctrl.plan_horizon,
            spline_type=base_ctrl.spline_type,
            num_knots=base_ctrl.num_knots,
            iterations=base_ctrl.iterations,
        )

    def init_params(self) -> PACParams:
        """Initialize the controller parameters."""
        base_params = self.base_ctrl.init_params()
        base_rng, our_rng = jax.random.split(base_params.rng)
        base_params = base_params.replace(rng=base_rng)
        # Policy samples are now knots, not full control sequences
        policy_samples = jnp.zeros(
            (
                self.num_policy_samples,
                self.num_knots,
                self.task.model.nu,
            )
        )
        # Initialize prev_elites with zeros (needed for JAX scan compatibility)
        if self.num_shift_elites > 0:
            prev_elites = jnp.zeros(
                (self.num_shift_elites, self.num_knots, self.task.model.nu)
            )
        else:
            prev_elites = jnp.zeros((0, self.num_knots, self.task.model.nu))
            
        return PACParams(
            tk=base_params.tk,
            mean=base_params.mean,
            base_params=base_params,
            policy_samples=policy_samples,
            rng=our_rng,
            prev_elites=prev_elites,
            value_params=None,
            value_alpha=0.0,
        )

    def sample_knots(self, params: PACParams) -> Tuple[jax.Array, PACParams]:
        """Sample control knots from the base controller and the policy.
        
        Includes iCEM improvements:
        - Colored noise for temporally correlated sampling
        - Shifted elites from previous MPC step for warm-starting
        """
        # Samples from the base controller
        base_samples, base_params = self.base_ctrl.sample_knots(
            params.base_params
        )
        
        # Apply colored noise if enabled (replace white noise with colored)
        if self.use_colored_noise:
            from gpc.icem import colored_noise
            rng, noise_rng = jax.random.split(base_params.rng)
            base_params = base_params.replace(rng=rng)
            
            # Generate colored noise and apply to mean
            colored = colored_noise(
                noise_rng, 
                base_samples.shape, 
                beta=self.noise_beta
            )
            # Get noise level from base controller if available
            noise_level = getattr(self.base_ctrl, 'noise_level', 0.1)
            if hasattr(self.base_ctrl, 'sigma_start'):
                # For CEM, use current covariance
                noise_level = getattr(base_params, 'cov', self.base_ctrl.sigma_start)
                if hasattr(noise_level, 'mean'):
                    noise_level = jnp.mean(noise_level)
            base_samples = base_params.mean + noise_level * colored

        # Add shifted elites from previous step
        if self.num_shift_elites > 0:
            from gpc.icem import shift_sequence
            rng, shift_rng = jax.random.split(params.rng)
            shifted = shift_sequence(
                params.prev_elites[:self.num_shift_elites],
                shift_rng,
                noise_level=0.1,
            )
            # Replace some base samples with shifted elites
            base_samples = base_samples.at[:self.num_shift_elites].set(shifted)
            params = params.replace(rng=rng)

        # Replace the last n_wide base samples with Uniform[-1, 1] wide-exploration
        # seeds (Solution 1 – Stratified SPC).  These are independent of both the
        # policy and the SPC mean, guaranteeing a non-vanishing probability of
        # discovering any mode every step.
        if self.n_wide > 0:
            rng, wide_rng = jax.random.split(params.rng)
            wide_samples = jax.random.uniform(
                wide_rng,
                (self.n_wide, self.num_knots, self.task.model.nu),
                minval=-1.0,
                maxval=1.0,
            )
            base_samples = base_samples.at[-self.n_wide :].set(wide_samples)
            params = params.replace(rng=rng)

        # Include samples from the policy (policy samples first for GPC strategy)
        samples = jnp.append(params.policy_samples, base_samples, axis=0)

        return samples, params.replace(
            tk=base_params.tk,
            mean=base_params.mean,
            base_params=base_params
        )

    def update_params(
        self, params: PACParams, rollouts: Trajectory
    ) -> PACParams:
        """Update the policy parameters according to the base controller.
        
        Also stores top elites for warm-starting the next MPC step (iCEM).
        """
        base_params = self.base_ctrl.update_params(params.base_params, rollouts)
        
        # Store elites for next step if using elite shifting
        prev_elites = params.prev_elites
        if self.num_shift_elites > 0:
            costs = jnp.sum(rollouts.costs, axis=1)
            elite_indices = jnp.argsort(costs)[:self.num_shift_elites]
            prev_elites = rollouts.knots[elite_indices]
        
        return params.replace(
            tk=base_params.tk,
            mean=base_params.mean,
            base_params=base_params,
            prev_elites=prev_elites,
        )

    def optimize(self, state: mjx.Data, params: PACParams) -> Tuple[PACParams, Trajectory]:
        """Perform SPC optimization, potentially with a terminal value function."""
        if self.value_fn is not None:
            # Wrap the task with Value function logic
            v_fn = lambda data: self.value_fn(params.value_params, data)
            va_task = ValueAugmentedTask(
                self.base_ctrl.task, 
                v_fn, 
                params.value_alpha,
                use_base_terminal_cost=self.use_task_terminal_cost
            )
            
            # Temporarily replace task in controllers
            old_base_task = self.base_ctrl.task
            old_pac_task = self.task
            self.base_ctrl.task = va_task
            self.task = va_task
            
            try:
                # Call base optimize (which calls sample_knots, rollout, update_params)
                # Note: SamplingBasedController.optimize is likely implemented in hydrax
                # We expect PAC to inherit it.
                new_params, rollouts = super().optimize(state, params)
            finally:
                # Restore tasks
                self.base_ctrl.task = old_base_task
                self.task = old_pac_task
                
            return new_params, rollouts
        else:
            return super().optimize(state, params)

    def get_action(self, params: PACParams, t: float) -> jax.Array:
        """Get the action from the base controller at a given time."""
        return self.base_ctrl.get_action(params.base_params, t)

    def get_action_sequence(self, params: PACParams) -> jax.Array:
        """Get the full control trajectory by interpolating knots."""
        # Get the full interpolated control trajectory from base controller
        # Use the interp_func to convert knots to control trajectory
        tq = jnp.arange(self.ctrl_steps) * self.dt  # Query times
        tk = params.tk  # Knot times
        knots = params.mean  # Control knots
        
        # Interpolate each action dimension
        controls = self.interp_func(tq, tk, knots.T).T
        
        return controls
