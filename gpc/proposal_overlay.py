"""Proposal visualization for MuJoCo scenes.

Draws proposal trajectories as coloured lines and optional ghost bodies in the
MuJoCo scene, so they appear in the rendered 3D view.
"""

import mujoco
import numpy as np


# ── Colour palettes ─────────────────────────────────────────────────────────

_PALETTES = {
    "tableau10": {
        "policy": np.array([0.12, 0.47, 0.71, 1.0]),   # blue
        "spc":    np.array([1.00, 0.50, 0.05, 1.0]),   # orange
        "best":   np.array([0.17, 0.63, 0.17, 1.0]),   # green
    },
    "colorbrewer": {
        "policy": np.array([0.22, 0.49, 0.72, 1.0]),
        "spc":    np.array([0.89, 0.10, 0.11, 1.0]),
        "best":   np.array([0.30, 0.69, 0.29, 1.0]),
    },
    "pastel": {
        "policy": np.array([0.55, 0.63, 0.80, 1.0]),
        "spc":    np.array([1.00, 0.73, 0.47, 1.0]),
        "best":   np.array([0.60, 0.85, 0.60, 1.0]),
    },
}


def get_palette(name: str = "tableau10") -> dict:
    """Return a named colour palette dict."""
    return _PALETTES.get(name, _PALETTES["tableau10"])


def _proposal_rgba(
    cost: float,
    min_cost: float,
    max_cost: float,
    is_best: bool,
    is_policy: bool,
    pal: dict,
    alpha_range: tuple[float, float] = (0.05, 0.25),
) -> np.ndarray:
    """Compute RGBA for one proposal based on cost and source."""
    if is_best:
        return pal["best"].copy()

    base = pal["policy"] if is_policy else pal["spc"]
    rgba = base.copy()

    # Map cost to alpha: lower cost → higher alpha
    cost_range = max_cost - min_cost
    if cost_range > 1e-8:
        t = (cost - min_cost) / cost_range          # 0 = best, 1 = worst
        rgba[3] = alpha_range[1] - t * (alpha_range[1] - alpha_range[0])
    else:
        rgba[3] = alpha_range[1]

    return rgba


# ── Scene-level drawing ─────────────────────────────────────────────────────

def _add_capsule(scene, p0, p1, rgba, radius=0.003):
    """Add a thin capsule (line segment) between two 3D points."""
    if scene.ngeom >= scene.maxgeom:
        return
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    diff = p1 - p0
    length = np.linalg.norm(diff)
    if length < 1e-8:
        return

    mid = (p0 + p1) / 2.0
    d = diff / length

    # Build rotation matrix aligning capsule z-axis with direction
    up = np.array([0., 0., 1.]) if abs(d[2]) < 0.99 else np.array([0., 1., 0.])
    x = np.cross(d, up)
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    mat = np.column_stack([x, y, d])  # 3×3

    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, length / 2, 0], dtype=np.float64),
        mid,
        mat.flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_proposal_geoms_to_scene(
    scene,
    trace_sites: np.ndarray,
    costs: np.ndarray,
    best_idx: int,
    num_policy_samples: int,
    pal: dict | None = None,
    line_width: float = 0.002,
    num_display: int = 0,
):
    """Draw proposal trajectory lines in the MuJoCo scene.

    Args:
        scene: ``mujoco.MjvScene`` (typically ``renderer._scene``).
        trace_sites: ``(S, H+1, num_sites, 3)`` — 3D positions of trace sites
            for each sample at each horizon step.
        costs: ``(S,)`` — total cost per sample (summed over horizon).
        best_idx: Index of the best (selected) sample.
        num_policy_samples: Number of leading samples that came from the policy.
        pal: Colour palette dict.
        line_width: Radius of capsule lines.
        num_display: Max proposals to show (0 = all).
    """
    if pal is None:
        pal = get_palette()

    S, Hp1, n_sites, _ = trace_sites.shape
    if n_sites == 0:
        return

    # Subsample proposals if requested
    if num_display > 0 and S > num_display:
        rng = np.random.default_rng(42)
        others = np.setdiff1d(np.arange(S), [best_idx])
        chosen = rng.choice(others, num_display - 1, replace=False)
        chosen = np.sort(np.concatenate([[best_idx], chosen]))
        best_local = int(np.where(chosen == best_idx)[0][0])
        nps_local = int(np.sum(chosen < num_policy_samples))
    else:
        chosen = np.arange(S)
        best_local = best_idx
        nps_local = num_policy_samples

    costs_sel = costs[chosen]
    traces_sel = trace_sites[chosen]
    min_c, max_c = float(costs_sel.min()), float(costs_sel.max())

    # Draw non-best samples first, then best on top
    order = list(range(len(chosen)))
    if best_local in order:
        order.remove(best_local)
        order.append(best_local)

    for si in order:
        is_best = (si == best_local)
        is_policy = (si < nps_local)
        rgba = _proposal_rgba(
            float(costs_sel[si]), min_c, max_c, is_best, is_policy, pal
        )
        width = line_width * (2.5 if is_best else 1.0)

        for site in range(n_sites):
            for h in range(Hp1 - 1):
                _add_capsule(
                    scene,
                    traces_sel[si, h, site],
                    traces_sel[si, h + 1, site],
                    rgba,
                    radius=width,
                )


def add_ghost_geoms_to_scene(
    scene,
    mj_model: mujoco.MjModel,
    current_qpos: np.ndarray,
    current_qvel: np.ndarray,
    controls: np.ndarray,
    costs: np.ndarray,
    best_idx: int,
    num_policy_samples: int,
    pal: dict | None = None,
    num_ghosts: int = 3,
    ghost_steps: int = 4,
    ghost_alpha: float = 0.05,
):
    """Add semi-transparent ghost bodies for top proposals.

    Forward-simulates the top proposals from the current state using CPU
    MuJoCo, then copies body geoms semi-transparently into the scene.

    Args:
        scene: ``mujoco.MjvScene``.
        mj_model: The MuJoCo model.
        current_qpos: Current joint positions ``(nq,)``.
        current_qvel: Current joint velocities ``(nv,)``.
        controls: ``(S, H, nu)`` — control sequences per sample.
        costs: ``(S,)`` — total cost per sample.
        best_idx: Index of the best sample.
        num_policy_samples: Number of leading policy samples.
        pal: Colour palette.
        num_ghosts: Number of top proposals to render as ghosts.
        ghost_steps: Number of future steps to render per ghost.
        ghost_alpha: Opacity multiplier for ghost geoms.
    """
    if pal is None:
        pal = get_palette()

    S, H, nu = controls.shape
    if num_ghosts <= 0 or ghost_steps <= 0:
        return

    # Pick top proposals by cost (always include best)
    sorted_idx = np.argsort(costs)
    ghost_indices = []
    for idx in sorted_idx:
        if len(ghost_indices) >= num_ghosts:
            break
        ghost_indices.append(int(idx))

    step_stride = max(1, H // ghost_steps)
    mj_data = mujoco.MjData(mj_model)

    # Identify which geoms belong to actual bodies (skip worldbody geoms like floor)
    body_geom_ids = [
        g for g in range(mj_model.ngeom)
        if mj_model.geom_bodyid[g] > 0  # skip worldbody (id=0)
    ]

    for gi, sample_idx in enumerate(ghost_indices):
        is_policy = sample_idx < num_policy_samples
        is_best = sample_idx == best_idx
        base_color = pal["best"] if is_best else (pal["policy"] if is_policy else pal["spc"])

        # Reset to current state
        mj_data.qpos[:] = current_qpos
        mj_data.qvel[:] = current_qvel
        mujoco.mj_forward(mj_model, mj_data)

        ctrl_seq = controls[sample_idx]
        step_count = 0
        for h in range(H):
            mj_data.ctrl[:nu] = ctrl_seq[h]
            mujoco.mj_step(mj_model, mj_data)

            if (h + 1) % step_stride == 0:
                step_count += 1
                # Fade alpha over time
                t_fade = step_count / ghost_steps
                alpha = ghost_alpha * (1.0 - 0.5 * t_fade)

                for gid in body_geom_ids:
                    if scene.ngeom >= scene.maxgeom:
                        return
                    g = scene.geoms[scene.ngeom]
                    # Copy geom type and size from model
                    mujoco.mjv_initGeom(
                        g,
                        int(mj_model.geom_type[gid]),
                        mj_model.geom_size[gid].astype(np.float64),
                        mj_data.geom_xpos[gid].astype(np.float64),
                        mj_data.geom_xmat[gid].astype(np.float64),
                        np.array([
                            base_color[0],
                            base_color[1],
                            base_color[2],
                            alpha,
                        ], dtype=np.float32),
                    )
                    scene.ngeom += 1

                if step_count >= ghost_steps:
                    break


def build_stats_lines(
    t: int,
    T: int,
    costs: np.ndarray,
    best_idx: int,
    num_policy_samples: int,
    cum_cost: float,
    pct_policy_best: float,
    label: str = "",
) -> list[str]:
    """Build text lines for the stats overlay."""
    S = len(costs)
    nps = num_policy_samples
    n_spc = S - nps

    policy_costs = costs[:nps] if nps > 0 else np.array([0.0])
    spc_costs = costs[nps:] if n_spc > 0 else np.array([])

    lines = []
    if label:
        lines.append(label)
    lines.append(f"Step {t}/{T}")
    lines.append(f"Policy: {float(policy_costs.mean()):.1f} +/- {float(policy_costs.std()):.1f}  (n={nps})")
    if len(spc_costs) > 0:
        lines.append(f"SPC:    {float(spc_costs.mean()):.1f} +/- {float(spc_costs.std()):.1f}  (n={n_spc})")
    is_policy_best = best_idx < nps
    lines.append(f"Best: {'Policy' if is_policy_best else 'SPC'} [{best_idx}] = {float(costs[best_idx]):.1f}")
    lines.append(f"Cum. cost: {cum_cost:.1f}")
    lines.append(f"Policy best: {pct_policy_best:.0f}%")
    return lines


def text_overlay(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    """Burn text lines onto a BGR frame using PIL."""
    from PIL import Image, ImageDraw

    # Convert BGR → RGB for PIL
    frame_rgb = frame_bgr[:, :, ::-1].copy()
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)

    x, y0, dy = 8, 10, 16
    for i, line in enumerate(lines):
        y = y0 + i * dy
        # Black shadow then white text
        draw.text((x + 1, y + 1), line, fill=(0, 0, 0))
        draw.text((x, y), line, fill=(255, 255, 255))

    return np.array(img)[:, :, ::-1]  # RGB → BGR


def color_legend_overlay(
    frame_bgr: np.ndarray, pal: dict, n_spc: int
) -> np.ndarray:
    """Add a small colour legend to the frame."""
    from PIL import Image, ImageDraw

    frame_rgb = frame_bgr[:, :, ::-1].copy()
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    h, w = frame_bgr.shape[:2]
    x = w - 110
    y = 10

    def _draw_swatch(color, label, yy):
        rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        draw.rectangle([x, yy, x + 12, yy + 12], fill=rgb)
        draw.text((x + 18, yy - 1), label, fill=(255, 255, 255))

    _draw_swatch(pal["policy"], "Policy", y)
    if n_spc > 0:
        _draw_swatch(pal["spc"], "SPC", y + 18)
    _draw_swatch(pal["best"], "Best", y + (36 if n_spc > 0 else 18))

    return np.array(img)[:, :, ::-1]  # RGB → BGR
