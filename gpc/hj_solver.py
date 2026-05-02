"""HJ solver + policy integration for avoid tasks (binary obstacle mode only)."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cloudpickle
import hj_reachability as hj
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


Array = jax.Array


@dataclass
class HJPolicyModel:
    """Tiny model holder so existing eval utils can infer horizon."""

    horizon: int

    def eval(self) -> "HJPolicyModel":
        return self

    def train(self) -> "HJPolicyModel":
        return self


@dataclass
class HJTablePolicy:
    """Policy object compatible with gpc.training/test utilities."""

    model: HJPolicyModel
    u_min: Array
    u_max: Array
    grid: hj.Grid
    policy_table: Array
    goal_pos: Array
    dt: float = 0.1

    def save(self, path: Path | str) -> None:
        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

    @staticmethod
    def load(path: Path | str) -> "HJTablePolicy":
        with open(path, "rb") as f:
            return cloudpickle.load(f)

    def replace(self, **kwargs) -> "HJTablePolicy":
        return replace(self, **kwargs)

    def apply(
        self,
        prev: Array,
        y: Array,
        rng: Array,
        warm_start_level: float = 0.0,
        target_cost: float | None = None,
        cfg_scale: float = 1.0,
    ) -> Array:
        del rng, warm_start_level, target_cost, cfg_scale
        pos = y[:2] + self.goal_pos
        vel = y[2:4]
        hj_state = jnp.concatenate([pos, vel])
        hj_state = jnp.clip(hj_state, self.grid.domain.lo, self.grid.domain.hi)
        u = self.grid.interpolate(self.policy_table, hj_state)
        u = jnp.nan_to_num(u, nan=0.0)
        u = jnp.clip(u, self.u_min, self.u_max)
        return jnp.broadcast_to(u[None, :], prev.shape)


def _to_python(x):
    if isinstance(x, dict):
        return {k: _to_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_python(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, jax.Array):
        arr = np.asarray(x)
        return arr.tolist() if arr.ndim > 0 else arr.item()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _save_array_chunked_axis0(
    path: Path,
    arr: Array,
    dtype: np.dtype,
    chunk_size0: int,
) -> None:
    shape = tuple(arr.shape)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    n0 = shape[0]
    for i0 in range(0, n0, chunk_size0):
        i1 = min(i0 + chunk_size0, n0)
        mm[i0:i1] = np.asarray(arr[i0:i1], dtype=dtype)
        if (i0 // chunk_size0) % 4 == 0 or i1 == n0:
            mm.flush()


def _box_signed_distance(pointmass_pos: Array, box_center: Array, half_extents: Array) -> Array:
    delta = jnp.abs(pointmass_pos - box_center) - half_extents
    outside = jnp.linalg.norm(jnp.maximum(delta, 0.0))
    inside = jnp.minimum(jnp.max(delta), 0.0)
    return outside + inside


def obstacle_signed_distance(pointmass_pos: Array, obstacle_pos: Array, variant: str) -> Array:
    if variant == "sphere":
        return jnp.linalg.norm(pointmass_pos - obstacle_pos) - 0.1
    if variant == "vertical_block":
        return _box_signed_distance(pointmass_pos, obstacle_pos, jnp.array([0.02, 0.12]))
    back_wall = _box_signed_distance(
        pointmass_pos, obstacle_pos + jnp.array([0.04, 0.0]), jnp.array([0.02, 0.10])
    )
    top_arm = _box_signed_distance(
        pointmass_pos, obstacle_pos + jnp.array([-0.01, 0.10]), jnp.array([0.05, 0.02])
    )
    bottom_arm = _box_signed_distance(
        pointmass_pos, obstacle_pos + jnp.array([-0.01, -0.10]), jnp.array([0.05, 0.02])
    )
    return jnp.minimum(back_wall, jnp.minimum(top_arm, bottom_arm))


@dataclass
class AvoidHJCost:
    """Binary-cost avoid objective."""

    goal_pos: Array
    obstacle_pos: Array
    variant: str
    control_weight: float
    obstacle_binary_weight: float

    def obstacle_cost(self, pos: Array) -> Array:
        sd = obstacle_signed_distance(pos, self.obstacle_pos, self.variant)
        return self.obstacle_binary_weight * (sd < 0.0).astype(pos.dtype)

    def terminal(self, state: Array) -> Array:
        pos = state[:2]
        vel = state[2:]
        pos_cost = jnp.sum(jnp.square(pos - self.goal_pos))
        vel_cost = 0.1 * jnp.sum(jnp.square(vel))
        return pos_cost + vel_cost + self.obstacle_cost(pos)

    def running(self, state: Array, control: Array) -> Array:
        return self.terminal(state) + self.control_weight * jnp.sum(jnp.square(control))


class AvoidAffineHJDynamics(hj.Dynamics):
    """4D affine point-mass dynamics for avoid."""

    def __init__(
        self,
        A: Array,
        B: Array,
        u_min: Array,
        u_max: Array,
        cost: AvoidHJCost,
    ) -> None:
        self.A = A
        self.B = B
        self.u_min = u_min
        self.u_max = u_max
        self.cost = cost
        super().__init__(
            control_mode="min",
            disturbance_mode="max",
            control_space=hj.sets.Box(u_min, u_max),
            disturbance_space=hj.sets.Box(jnp.zeros((0,)), jnp.zeros((0,))),
        )

    def __call__(self, state: Array, control: Array, disturbance: Array, time: float) -> Array:
        del disturbance, time
        return self.A @ state + self.B @ control

    def optimal_control_and_disturbance(
        self,
        state: Array,
        time: float,
        grad_value: Array,
    ) -> tuple[Array, Array]:
        del state, time
        control_dir = self.B.T @ grad_value
        denom = 2.0 * self.cost.control_weight
        u_star = -control_dir / denom
        u_star = jnp.clip(u_star, self.u_min, self.u_max)
        return u_star, jnp.zeros((0,))

    def hamiltonian(self, state: Array, time: float, value: Array, grad_value: Array) -> Array:
        u_star, _ = self.optimal_control_and_disturbance(state, time, grad_value)
        drift = self.A @ state + self.B @ u_star
        return grad_value @ drift + self.cost.running(state, u_star)

    def partial_max_magnitudes(
        self,
        state: Array,
        time: float,
        value: Array,
        grad_value_box: hj.sets.Box,
    ) -> Array:
        del time, value, grad_value_box
        umax = jnp.maximum(jnp.abs(self.u_min), jnp.abs(self.u_max))
        return jnp.abs(self.A @ state) + jnp.abs(self.B) @ umax


def _build_grid(env, grid_size: int, velocity_bound: float) -> hj.Grid:
    model = env.task.mj_model
    root_x_id = model.joint("root_x").id
    root_y_id = model.joint("root_y").id
    root_x_range = model.jnt_range[root_x_id]
    root_y_range = model.jnt_range[root_y_id]
    pointmass_body_id = model.body("pointmass").id
    pointmass_world_offset = model.body_pos[pointmass_body_id, :2]

    x_lo = float(pointmass_world_offset[0] + root_x_range[0])
    x_hi = float(pointmass_world_offset[0] + root_x_range[1])
    y_lo = float(pointmass_world_offset[1] + root_y_range[0])
    y_hi = float(pointmass_world_offset[1] + root_y_range[1])

    domain = hj.sets.Box(
        lo=jnp.array([x_lo, y_lo, -velocity_bound, -velocity_bound]),
        hi=jnp.array([x_hi, y_hi, velocity_bound, velocity_bound]),
    )
    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain,
        (grid_size, grid_size, grid_size, grid_size),
    )


def _add_obstacle_and_goal(ax: plt.Axes, variant: str, obstacle_pos: np.ndarray, goal_pos: np.ndarray) -> None:
    ox = float(obstacle_pos[0])
    oy = float(obstacle_pos[1])
    if variant == "sphere":
        ax.add_patch(
            patches.Circle((ox, oy), radius=0.1, facecolor="k", edgecolor="k", linewidth=1.5, alpha=0.18)
        )
    elif variant == "vertical_block":
        ax.add_patch(
            patches.Rectangle((ox - 0.02, oy - 0.12), 0.04, 0.24, facecolor="k", edgecolor="k", linewidth=1.5, alpha=0.18)
        )
    else:
        ax.add_patch(
            patches.Rectangle((ox + 0.04 - 0.02, oy - 0.10), 0.04, 0.20, facecolor="k", edgecolor="k", linewidth=1.5, alpha=0.18)
        )
        ax.add_patch(
            patches.Rectangle((ox - 0.01 - 0.05, oy + 0.10 - 0.02), 0.10, 0.04, facecolor="k", edgecolor="k", linewidth=1.5, alpha=0.18)
        )
        ax.add_patch(
            patches.Rectangle((ox - 0.01 - 0.05, oy - 0.10 - 0.02), 0.10, 0.04, facecolor="k", edgecolor="k", linewidth=1.5, alpha=0.18)
        )
    ax.plot(
        float(goal_pos[0]),
        float(goal_pos[1]),
        marker="*",
        markersize=13,
        markerfacecolor="gold",
        markeredgecolor="k",
        markeredgewidth=0.9,
        linestyle="None",
        zorder=5,
    )


def _obstacle_mask_xy(
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    variant: str,
    obstacle_pos: np.ndarray,
) -> np.ndarray:
    """Boolean mask where True means point lies inside obstacle."""
    X, Y = np.meshgrid(x_vec, y_vec, indexing="xy")
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)
    obs = jnp.asarray(obstacle_pos, dtype=jnp.float32)
    sdf = jax.vmap(lambda p: obstacle_signed_distance(jnp.asarray(p), obs, variant))(jnp.asarray(pts))
    inside = np.asarray(sdf < 0.0).reshape(Y.shape)
    return inside


def _plot_pair(
    out_path: Path,
    value_xy: np.ndarray,
    ux_xy: np.ndarray,
    uy_xy: np.ndarray,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    variant: str,
    obstacle_pos: np.ndarray,
    goal_pos: np.ndarray,
    interpolation: str,
) -> None:
    inside_mask = _obstacle_mask_xy(
        x_vec=x_vec,
        y_vec=y_vec,
        variant=variant,
        obstacle_pos=obstacle_pos,
    )
    value_masked = np.array(value_xy, copy=True)
    value_masked[inside_mask] = np.nan

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.96))
    outside_vals = value_masked[np.isfinite(value_masked)]
    if outside_vals.size > 0:
        vmin = float(np.percentile(outside_vals, 2))
        vmax = float(np.percentile(outside_vals, 98))
        if vmax <= vmin:
            vmin = float(np.min(outside_vals))
            vmax = float(np.max(outside_vals))
    else:
        vmin = None
        vmax = None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    extent = [x_vec[0], x_vec[-1], y_vec[0], y_vec[-1]]
    pcm0 = axes[0].imshow(
        value_masked,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation=interpolation,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    _add_obstacle_and_goal(axes[0], variant, obstacle_pos, goal_pos)
    axes[0].set_title("Value V(x,y,vx=0,vy=0)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True, linewidth=0.35, alpha=0.25)
    fig.colorbar(pcm0, ax=axes[0], fraction=0.046)

    pcm1 = axes[1].imshow(
        value_masked,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation=interpolation,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    _add_obstacle_and_goal(axes[1], variant, obstacle_pos, goal_pos)
    axes[1].set_title("Policy overlay on value")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].grid(True, linewidth=0.35, alpha=0.25)
    fig.colorbar(pcm1, ax=axes[1], fraction=0.046)

    # Sparser, cleaner arrows for publication-quality readability.
    step = max(2, len(x_vec) // 24)
    Xq, Yq = np.meshgrid(x_vec[::step], y_vec[::step], indexing="xy")
    Uq = np.array(ux_xy[::step, ::step], copy=True)
    Vq = np.array(uy_xy[::step, ::step], copy=True)
    inside_q = inside_mask[::step, ::step]
    Uq[inside_q] = np.nan
    Vq[inside_q] = np.nan
    axes[1].quiver(
        Xq,
        Yq,
        Uq,
        Vq,
        color="w",
        # alpha=0.85,
        # width=0.0026,
        scale_units="xy",
        scale=50,
        pivot="mid",
        # headwidth=1.4,
        # headlength=2.6,
        # headaxislength=0.6,
        zorder=4,
    )

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_value_policy(
    out_dir: Path,
    grid: hj.Grid,
    values: Array,
    policy: Array,
    variant: str,
    obstacle_pos: np.ndarray,
    goal_pos: np.ndarray,
) -> None:
    ix_vx0 = int(np.argmin(np.abs(np.asarray(grid.coordinate_vectors[2]))))
    ix_vy0 = int(np.argmin(np.abs(np.asarray(grid.coordinate_vectors[3]))))
    x_vec = np.asarray(grid.coordinate_vectors[0])
    y_vec = np.asarray(grid.coordinate_vectors[1])
    value_xy = np.asarray(values[:, :, ix_vx0, ix_vy0]).T
    ux_xy = np.asarray(policy[:, :, ix_vx0, ix_vy0, 0]).T
    uy_xy = np.asarray(policy[:, :, ix_vx0, ix_vy0, 1]).T
    _plot_pair(
        out_dir / "value_policy_bins.png",
        value_xy,
        ux_xy,
        uy_xy,
        x_vec,
        y_vec,
        variant,
        obstacle_pos,
        goal_pos,
        "nearest",
    )
    _plot_pair(
        out_dir / "value_policy_interpolated.png",
        value_xy,
        ux_xy,
        uy_xy,
        x_vec,
        y_vec,
        variant,
        obstacle_pos,
        goal_pos,
        "bicubic",
    )


def regenerate_avoid_hj_plots(hj_dir: Path) -> None:
    """Regenerate publication plots from saved HJ artifacts."""
    hj_dir = Path(hj_dir)
    with open(hj_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta: dict[str, Any] = json.load(f)

    coords = np.load(hj_dir / "grid_coordinates.npz")
    x_vec = coords["axis_0"]
    y_vec = coords["axis_1"]
    vx_vec = coords["axis_2"]
    vy_vec = coords["axis_3"]
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(
            lo=jnp.asarray(coords["domain_lo"]),
            hi=jnp.asarray(coords["domain_hi"]),
        ),
        (len(x_vec), len(y_vec), len(vx_vec), len(vy_vec)),
    )
    values = np.load(hj_dir / "value_function.npy")
    policy = np.load(hj_dir / "policy.npy").astype(np.float32)

    _plot_value_policy(
        out_dir=hj_dir,
        grid=grid,
        values=jnp.asarray(values),
        policy=jnp.asarray(policy),
        variant=meta["task_variant"],
        obstacle_pos=np.asarray(meta["obstacle_pos"], dtype=np.float32),
        goal_pos=np.asarray(meta["goal_pos"], dtype=np.float32),
    )


def solve_avoid_hj_binary(
    env,
    config,
    out_dir: Path,
) -> tuple[HJTablePolicy, dict]:
    """Solve HJ once and return a deterministic table policy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.task_name != "avoid":
        raise ValueError("HJ mode currently supports only task_name='avoid'.")
    if config.task_variant not in {"sphere", "vertical_block", "u_trap"}:
        raise ValueError("HJ mode requires avoid task_variant in {sphere, vertical_block, u_trap}.")

    grid = _build_grid(env, config.hj_grid_size, config.hj_velocity_bound)
    init_state = env.init_state(jax.random.key(config.seed))
    obstacle_pos = init_state.data.mocap_pos[env.task.obstacle_mocap_id, 0:2]
    goal_pos = init_state.data.mocap_pos[env.task.goal_mocap_id, 0:2]

    cost = AvoidHJCost(
        goal_pos=goal_pos,
        obstacle_pos=obstacle_pos,
        variant=config.task_variant,
        control_weight=config.hj_control_weight,
        obstacle_binary_weight=config.hj_obstacle_binary_weight,
    )

    model = env.task.mj_model
    damping = 0.5 * float(model.dof_damping[0] + model.dof_damping[1])
    mass = float(model.body_mass[model.body("pointmass").id])
    A = jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -damping / mass, 0.0],
            [0.0, 0.0, 0.0, -damping / mass],
        ]
    )
    B = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0 / mass, 0.0],
            [0.0, 1.0 / mass],
        ]
    )
    dynamics = AvoidAffineHJDynamics(A=A, B=B, u_min=env.task.u_min, u_max=env.task.u_max, cost=cost)

    flat_states = grid.states.reshape(-1, 4)
    flat_terminal = jax.vmap(cost.terminal)(flat_states)
    values = flat_terminal.reshape(grid.shape).astype(jnp.float32)

    solver_settings = hj.SolverSettings.with_accuracy(config.hj_solver_accuracy)
    horizon_seconds = env.episode_length * env.task.model.opt.timestep
    times = jnp.linspace(0.0, -horizon_seconds, config.hj_num_time_slices)
    for i in range(1, config.hj_num_time_slices):
        values = hj.step(
            solver_settings=solver_settings,
            dynamics=dynamics,
            grid=grid,
            time=float(times[i - 1]),
            values=values,
            target_time=float(times[i]),
            progress_bar=False,
        )

    grad_values = grid.grad_values(values, upwind_scheme=solver_settings.upwind_scheme)
    flat_grads = grad_values.reshape(-1, 4)
    flat_policy = jax.vmap(lambda s, g: dynamics.optimal_control(s, 0.0, g))(flat_states, flat_grads)
    policy_table = flat_policy.reshape((*grid.shape, env.task.model.nu)).astype(jnp.float32)

    _save_array_chunked_axis0(out_dir / "value_function.npy", values, np.float32, config.hj_save_chunk_size0)
    _save_array_chunked_axis0(out_dir / "policy.npy", policy_table, np.float16, config.hj_save_chunk_size0)
    np.savez(
        out_dir / "grid_coordinates.npz",
        axis_0=np.asarray(grid.coordinate_vectors[0], dtype=np.float32),
        axis_1=np.asarray(grid.coordinate_vectors[1], dtype=np.float32),
        axis_2=np.asarray(grid.coordinate_vectors[2], dtype=np.float32),
        axis_3=np.asarray(grid.coordinate_vectors[3], dtype=np.float32),
        spacings=np.asarray(grid.spacings, dtype=np.float32),
        domain_lo=np.asarray(grid.domain.lo, dtype=np.float32),
        domain_hi=np.asarray(grid.domain.hi, dtype=np.float32),
    )

    _plot_value_policy(
        out_dir=out_dir,
        grid=grid,
        values=values,
        policy=policy_table,
        variant=config.task_variant,
        obstacle_pos=np.asarray(obstacle_pos),
        goal_pos=np.asarray(goal_pos),
    )

    metadata = {
        "task_name": config.task_name,
        "task_variant": config.task_variant,
        "grid_size": config.hj_grid_size,
        "grid_shape": list(grid.shape),
        "velocity_bound": config.hj_velocity_bound,
        "num_time_slices": config.hj_num_time_slices,
        "solver_accuracy": config.hj_solver_accuracy,
        "horizon_seconds": float(horizon_seconds),
        "control_weight": config.hj_control_weight,
        "obstacle_binary_weight": config.hj_obstacle_binary_weight,
        "goal_pos": np.asarray(goal_pos).tolist(),
        "obstacle_pos": np.asarray(obstacle_pos).tolist(),
        "grid_domain_lo": np.asarray(grid.domain.lo).tolist(),
        "grid_domain_hi": np.asarray(grid.domain.hi).tolist(),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(_to_python(metadata), f, indent=2)

    policy = HJTablePolicy(
        model=HJPolicyModel(horizon=config.num_knots),
        u_min=env.task.u_min,
        u_max=env.task.u_max,
        grid=grid,
        policy_table=policy_table,
        goal_pos=goal_pos,
    )
    return policy, metadata
