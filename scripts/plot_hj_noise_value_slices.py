#!/usr/bin/env python3
"""Create one combined HJ value-noise figure for failure sigma rows.

Figure layout:
- rows: failure sigma pair (e.g., CEM-fail row, uniform-fail row)
- cols: base V, z (iid), Vhat (iid), z (corr), Vhat (corr)

All panels share one colormap scale: the base HJ free-space 2/98 percentile
limits, matching the HJ style baseline scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpc.hj_solver import _obstacle_mask_xy
from scripts.avoid_value_guidance_experiment import (
    positive_softplus_gaussian_field,
    smooth_noise_field,
)


def parse_float_list(text: str) -> list[float]:
    out = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one comma-separated float value.")
    return out


def parse_text_list(text: str) -> list[str]:
    out = [x.strip() for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one comma-separated label.")
    return out


def hj_style_limits(masked: np.ndarray) -> tuple[float | None, float | None]:
    vals = masked[np.isfinite(masked)]
    if vals.size == 0:
        return None, None
    vmin = float(np.percentile(vals, 2.0))
    vmax = float(np.percentile(vals, 98.0))
    if vmax <= vmin:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
    return vmin, vmax


def make_noise_field(
    shape: tuple[int, ...],
    sigma: float,
    mean: float,
    seed: int,
    smooth_passes: int,
) -> np.ndarray:
    if sigma == 0.0:
        return np.full(shape, mean, dtype=np.float32)
    rng = np.random.default_rng(seed)
    z0 = rng.standard_normal(shape, dtype=np.float32)
    z0 = smooth_noise_field(z0, smooth_passes=smooth_passes)
    z0 = (z0 - np.mean(z0)) / (np.std(z0) + 1e-8)
    return positive_softplus_gaussian_field(
        z0,
        mean=float(mean),
        small_noise_std=float(sigma),
    )


def masked(arr: np.ndarray, inside_mask: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out[inside_mask] = np.nan
    return out


def load_hj(hj_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    coords = np.load(hj_dir / "grid_coordinates.npz")
    x_vec = np.asarray(coords["axis_0"], dtype=np.float32)
    y_vec = np.asarray(coords["axis_1"], dtype=np.float32)
    vx_vec = np.asarray(coords["axis_2"], dtype=np.float32)
    vy_vec = np.asarray(coords["axis_3"], dtype=np.float32)
    values = np.asarray(np.load(hj_dir / "value_function.npy"), dtype=np.float32)
    with open(hj_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return x_vec, y_vec, vx_vec, vy_vec, values, metadata


def plot_combined_failure_rows(
    hj_dir: Path,
    out_path: Path,
    iid_sigmas: list[float],
    corr_sigmas: list[float],
    row_labels: list[str],
    noise_mean: float,
    noise_seed: int,
) -> None:
    x_vec, y_vec, vx_vec, vy_vec, values, metadata = load_hj(hj_dir)

    if len(iid_sigmas) != len(corr_sigmas):
        raise ValueError("iid_sigmas and corr_sigmas must have the same length.")
    if len(row_labels) != len(iid_sigmas):
        raise ValueError("row_labels length must match sigma list length.")

    ix_vx0 = int(np.argmin(np.abs(vx_vec)))
    ix_vy0 = int(np.argmin(np.abs(vy_vec)))
    vx0 = float(vx_vec[ix_vx0])
    vy0 = float(vy_vec[ix_vy0])

    variant = str(metadata.get("task_variant", "u_trap"))
    obstacle_pos = np.asarray(metadata["obstacle_pos"], dtype=np.float32)
    inside_mask = _obstacle_mask_xy(
        x_vec=x_vec,
        y_vec=y_vec,
        variant=variant,
        obstacle_pos=obstacle_pos,
    )

    base_xy = np.asarray(values[:, :, ix_vx0, ix_vy0]).T
    base_masked = masked(base_xy, inside_mask)

    base_vmin, base_vmax = hj_style_limits(base_masked)
    if base_vmin is None or base_vmax is None:
        raise ValueError("Could not compute base HJ limits from free-space cells.")

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.96))

    extent = [x_vec[0], x_vec[-1], y_vec[0], y_vec[-1]]

    n_rows = len(iid_sigmas)
    fig, axes = plt.subplots(n_rows, 5, figsize=(22, 5.2 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.asarray([axes])

    fig.suptitle(
        (
            f"HJ value slice at vx={vx0:.4f}, vy={vy0:.4f}: obstacle interior masked\n"
            "all panels use one shared base-HJ scale (free-space 2-98 percentiles)"
        ),
        fontsize=18,
    )

    im_ref = None
    for r, (sigma_iid, sigma_corr, row_label) in enumerate(zip(iid_sigmas, corr_sigmas, row_labels)):
        z_iid = make_noise_field(
            shape=values.shape,
            sigma=float(sigma_iid),
            mean=float(noise_mean),
            seed=int(noise_seed),
            smooth_passes=0,
        )
        z_corr = make_noise_field(
            shape=values.shape,
            sigma=float(sigma_corr),
            mean=float(noise_mean),
            seed=int(noise_seed),
            smooth_passes=4,
        )

        z_iid_xy = masked(np.asarray(z_iid[:, :, ix_vx0, ix_vy0]).T, inside_mask)
        z_corr_xy = masked(np.asarray(z_corr[:, :, ix_vx0, ix_vy0]).T, inside_mask)
        vhat_iid_xy = masked(np.asarray((values + z_iid * values)[:, :, ix_vx0, ix_vy0]).T, inside_mask)
        vhat_corr_xy = masked(np.asarray((values + z_corr * values)[:, :, ix_vx0, ix_vy0]).T, inside_mask)

        panels = [
            ("Base V", base_masked),
            (f"z iid, sigma={sigma_iid:g}", z_iid_xy),
            (f"Vhat iid, sigma={sigma_iid:g}", vhat_iid_xy),
            (f"z corr, sigma={sigma_corr:g}", z_corr_xy),
            (f"Vhat corr, sigma={sigma_corr:g}", vhat_corr_xy),
        ]

        for c, (title, arr) in enumerate(panels):
            im = axes[r, c].imshow(
                arr,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=cmap,
                vmin=base_vmin,
                vmax=base_vmax,
                interpolation="nearest",
            )
            axes[r, c].set_title(title)
            axes[r, c].set_xlabel("x")
            axes[r, c].set_ylabel("y")
            if im_ref is None:
                im_ref = im

        axes[r, 0].text(
            -0.35,
            0.5,
            row_label,
            transform=axes[r, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("shared base HJ scale")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hj-dir",
        type=Path,
        default=Path("experiments/avoid/hj_avoid_binary_u_trap_20260502_131354/hj"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "experiments/avoid_value_guidance_u_trap_uniform_noise/"
            "20260510_163252/value_slice_failure_rows_combined_hj_scale.png"
        ),
    )
    parser.add_argument("--noise-mean", type=float, default=0.1)
    parser.add_argument("--noise-seed", type=int, default=123)
    parser.add_argument(
        "--iid-fail-sigmas",
        type=str,
        default="0.048,3",
        help="Comma-separated iid failure sigmas in row order (e.g., CEM,uniform).",
    )
    parser.add_argument(
        "--corr-fail-sigmas",
        type=str,
        default="0.1,64",
        help="Comma-separated correlated failure sigmas in row order (e.g., CEM,uniform).",
    )
    parser.add_argument(
        "--row-labels",
        type=str,
        default="CEM fail sigma row,Uniform fail sigma row",
        help="Comma-separated row labels aligned with sigma rows.",
    )
    args = parser.parse_args()

    iid_sigmas = parse_float_list(args.iid_fail_sigmas)
    corr_sigmas = parse_float_list(args.corr_fail_sigmas)
    row_labels = parse_text_list(args.row_labels)

    plot_combined_failure_rows(
        hj_dir=args.hj_dir,
        out_path=args.out,
        iid_sigmas=iid_sigmas,
        corr_sigmas=corr_sigmas,
        row_labels=row_labels,
        noise_mean=float(args.noise_mean),
        noise_seed=int(args.noise_seed),
    )

    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
