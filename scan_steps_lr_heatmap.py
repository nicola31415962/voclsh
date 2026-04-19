"""Scan optimizer steps and learning rate for N=3 (fixed POVM plane) and plot a 2D heatmap.

Adjust STATE_CONSTRAINT externally (full/product_pure). This script keeps the POVM fixed (XZ)
to focus on optimizer sensitivity.
"""

import os
import time
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import states_functions as sf
import povm_functions as pf
import opt_variance_coef as ocf

jax.config.update("jax_enable_x64", True)

# --- configuration ---
STATE_MODE = os.getenv("STATE_CONSTRAINT", "full")  # or set before running
POVM_AXES = ("X", "Z")
N = 3
DENSITY = 2  # theta resolution for the sweep

# grids to scan
STEPS_GRID: List[int] = [50, 100, 200, 400, 800]
LR_GRID: List[float] = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]


def optimise_variance_custom(pcm, bm, observable, steps: int, lr: float):
    """Run the same loss as ocf.variance_optimisation but with custom Adam params."""
    d = len(observable)
    D = d ** 2
    flatobs = sf.flatten_in_basis(observable, bm)
    x0 = jnp.ones((D,), dtype=jnp.float64) / D

    loss = lambda x: ocf.var_state_optimisation(x, pcm, bm, flatobs)
    loss = jax.jit(loss)
    xs, neg_var = ocf._adam_minimize(loss, x0, steps=int(steps), lr=float(lr))

    flat_rho = sf.flat_state_from_mat(xs, bm)
    var = -neg_var
    return float(jnp.real(var)), flat_rho


def main():
    os.environ["STATE_CONSTRAINT"] = STATE_MODE
    sf.STATE_CONSTRAINT = STATE_MODE

    povm = pf.pauli_povm_single(*POVM_AXES)
    povm_N = pf.tensor_same(povm, N)
    pcm, bm = pf.povm_coef_matrix(povm_N)

    thetas = jnp.pi * jnp.arange(DENSITY) / (4 * DENSITY)

    heat_min = np.zeros((len(STEPS_GRID), len(LR_GRID)))
    heat_avg = np.zeros_like(heat_min)
    heat_time = np.zeros_like(heat_min)

    for i, steps in enumerate(STEPS_GRID):
        for j, lr in enumerate(LR_GRID):
            t0 = time.time()
            vars_here = []
            for th in thetas:
                singleobs = sf.qubit(th, 0.0)
                obs = pf.tensor_same(singleobs, N)
                var, _ = optimise_variance_custom(pcm, bm, obs, steps, lr)
                vars_here.append(var)
            elapsed = time.time() - t0
            heat_min[i, j] = np.min(vars_here)
            heat_avg[i, j] = np.mean(vars_here)
            heat_time[i, j] = elapsed
            print(f"steps={steps}, lr={lr}: min={heat_min[i,j]:.6f}, avg={heat_avg[i,j]:.6f}, time={elapsed:.2f}s")

    # --- plotting ---
    def _plot_heat(data, title, fname, cmap="viridis"):
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(data, origin="lower", cmap=cmap,
                       extent=[0, len(LR_GRID), 0, len(STEPS_GRID)],
                       aspect="auto")
        ax.set_xticks(np.arange(len(LR_GRID)) + 0.5)
        ax.set_xticklabels([f"{lr:g}" for lr in LR_GRID], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(STEPS_GRID)) + 0.5)
        ax.set_yticklabels([str(s) for s in STEPS_GRID])
        ax.set_xlabel("learning rate")
        ax.set_ylabel("steps")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.show()

    mode_tag = STATE_MODE
    _plot_heat(heat_min, f"Min variance (N={N}, mode={mode_tag})", f"heat_min_N{N}_{mode_tag}.png")
    _plot_heat(heat_avg, f"Avg variance (N={N}, mode={mode_tag})", f"heat_avg_N{N}_{mode_tag}.png")
    _plot_heat(heat_time, f"Runtime [s] (N={N}, mode={mode_tag})", f"heat_time_N{N}_{mode_tag}.png", cmap="magma")

    # Overlay runtime contours on min variance heatmap for a combined view
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heat_min, origin="lower", cmap="viridis",
                   extent=[0, len(LR_GRID), 0, len(STEPS_GRID)],
                   aspect="auto")
    cs = ax.contour(np.arange(len(LR_GRID)) + 0.5,
                    np.arange(len(STEPS_GRID)) + 0.5,
                    heat_time, colors="white", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f s")
    ax.set_xticks(np.arange(len(LR_GRID)) + 0.5)
    ax.set_xticklabels([f"{lr:g}" for lr in LR_GRID], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(STEPS_GRID)) + 0.5)
    ax.set_yticklabels([str(s) for s in STEPS_GRID])
    ax.set_xlabel("learning rate")
    ax.set_ylabel("steps")
    ax.set_title(f"Min variance with runtime contours (N={N}, mode={mode_tag})")
    fig.colorbar(im, ax=ax, label="min variance")
    plt.tight_layout()
    plt.savefig(f"heat_min_runtime_overlay_N{N}_{mode_tag}.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
