"""
Sweep equatorial observables (theta = pi/2, varying phi) and plot worst-case variance
for a Pauli-XY POVM. Shows optimiser (state+coeffs) and canonical coefficients baseline.
Works for single or multi-qubit (same equatorial projector tensorised).
"""

import os
import time
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import states_functions as sf
import povm_functions as pf
import opt_variance_coef as ocf

jax.config.update("jax_enable_x64", True)

# ---- Config (env overrides) ----
N = int(os.getenv("PHASE_N", 1))
STATE_MODE = os.getenv("STATE_CONSTRAINT", "full")
DENSITY_PHI = int(os.getenv("PHASE_DENSITY_PHI", 48))  # number of phi samples in [0, 2pi)

# state sampling (full mode)
SAMPLES_FULL_PURE = int(os.getenv("PHASE_SAMPLES_FULL_PURE", 800))
SAMPLES_FULL_MIX = int(os.getenv("PHASE_SAMPLES_FULL_MIX", 400))
# grids (product_pure mode)
DENSITY_STATE_TH = int(os.getenv("PHASE_STATE_TH", 64))
DENSITY_STATE_PH = int(os.getenv("PHASE_STATE_PH", 64))

# optimiser settings
OPT_STEPS = int(os.getenv("PHASE_OPT_STEPS", 2000))
OPT_LR = float(os.getenv("PHASE_OPT_LR", 5e-3))
OPT_RESTARTS = int(os.getenv("PHASE_OPT_RESTARTS", 4))
Y_PAD_FRAC = float(os.getenv("PHASE_Y_PAD_FRAC", 0.08))
N1_Y_HALFSPAN = float(os.getenv("PHASE_N1_Y_HALFSPAN", 2e-6))


def sample_states(bm) -> List[jnp.ndarray]:
    flat_states: List[jnp.ndarray] = []
    D = bm.shape[0]
    d = int(np.sqrt(D))

    if STATE_MODE.lower() == "product_pure":
        thetas = jnp.linspace(0, jnp.pi, DENSITY_STATE_TH)
        phis = jnp.linspace(0, 2 * jnp.pi, DENSITY_STATE_PH)
        for th in thetas:
            for ph in phis:
                rho1 = sf.qubit(th, ph)
                rhoN = pf.tensor_same(rho1, N)
                flat_states.append(sf.flatten_in_basis(rhoN, bm))
        return flat_states

    key = jax.random.PRNGKey(0)

    for _ in range(SAMPLES_FULL_PURE):
        key, sub = jax.random.split(key)
        psi = jax.random.normal(sub, (d,)) + 1j * jax.random.normal(sub, (d,))
        psi = psi / jnp.linalg.norm(psi)
        rho = jnp.outer(psi, jnp.conj(psi))
        flat_states.append(sf.flatten_in_basis(rho, bm))

    for _ in range(SAMPLES_FULL_MIX):
        key, sub = jax.random.split(key)
        G = jax.random.normal(sub, (d, d)) + 1j * jax.random.normal(sub, (d, d))
        rho = G @ jnp.conj(G.T)
        rho = rho / jnp.trace(rho)
        flat_states.append(sf.flatten_in_basis(rho, bm))

    rho_mm = jnp.eye(d, dtype=jnp.complex128) / d
    flat_states.append(sf.flatten_in_basis(rho_mm, bm))
    return flat_states


def _max_variance_with_restarts(loss, D, key, warm_start=None):
    """Maximise variance by minimising the corresponding negative-variance loss."""
    loss = jax.jit(loss)

    seeds = []
    if warm_start is not None:
        seeds.append(warm_start)
    while len(seeds) < max(1, OPT_RESTARTS):
        key, sub = jax.random.split(key)
        seeds.append(jax.random.normal(sub, (D,), dtype=jnp.float64))

    best_x = None
    best_val = None
    for x0 in seeds:
        xs, val = ocf._adam_minimize(loss, x0, steps=OPT_STEPS, lr=OPT_LR)
        if best_val is None or float(val) < float(best_val):
            best_val = val
            best_x = xs
    return float(-best_val), best_x, key


def canonical_worst(pcm, bm, obs, key, warm_start=None):
    flat_obs = sf.flatten_in_basis(obs, bm)
    can_em = jnp.linalg.pinv(pcm)
    coefs = can_em.T @ jnp.conj(flat_obs)
    D = bm.shape[0]
    loss = lambda x: jnp.real(ocf.variance_coef_freevars(x, pcm, bm, coefs))
    return _max_variance_with_restarts(loss, D, key, warm_start=warm_start)


def optimiser_worst(pcm, bm, obs, key, warm_start=None):
    d = len(obs)
    D = d ** 2
    flat_obs = sf.flatten_in_basis(obs, bm)
    loss = lambda x: ocf.var_state_optimisation(x, pcm, bm, flat_obs)
    return _max_variance_with_restarts(loss, D, key, warm_start=warm_start)


def main():
    sf.STATE_CONSTRAINT = STATE_MODE
    povm = pf.pauli_povm_single("X", "Y")
    povm_N = pf.tensor_same(povm, N)
    pcm, bm = pf.povm_coef_matrix(povm_N)

    phis = jnp.linspace(0, 2 * jnp.pi, DENSITY_PHI, endpoint=False)
    can_vals = []
    opt_vals = []
    key_can = jax.random.PRNGKey(0)
    key_opt = jax.random.PRNGKey(1)
    x_can = None
    x_opt = None

    t0 = time.time()
    for ph in phis:
        obs1 = sf.qubit(jnp.pi / 2, ph)
        obs = pf.tensor_same(obs1, N)
        can_v, x_can, key_can = canonical_worst(pcm, bm, obs, key_can, warm_start=x_can)
        opt_v, x_opt, key_opt = optimiser_worst(pcm, bm, obs, key_opt, warm_start=x_opt)
        can_vals.append(can_v)
        opt_vals.append(opt_v)
        print(f"phi={float(ph):.3f}: canonical={can_vals[-1]:.6f}, optimiser={opt_vals[-1]:.6f}")
    print(f"done in {time.time()-t0:.2f}s")

    ph_deg = np.array(phis) * 180 / np.pi
    can_np = np.array(can_vals)
    opt_np = np.array(opt_vals)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(ph_deg, can_np, marker="o", label="canonical coeffs (Penrose) max over states")
    ax.plot(ph_deg, opt_np, marker="x", linestyle="--", label="optimiser (state+coeffs)")
    ax.set_xlabel("phi (deg), equator theta=pi/2")
    ax.set_ylabel("worst-case variance")
    ax.set_title(f"N={N}, POVM=XY, mode={STATE_MODE}")
    all_vals = np.concatenate([can_np, opt_np])
    if N == 1:
        # N=1 XY equator is symmetry-flat around 0.5; show a fixed, readable band.
        ax.set_ylim(0.5 - N1_Y_HALFSPAN, 0.5 + N1_Y_HALFSPAN)
    else:
        y_min = float(np.min(all_vals))
        y_max = float(np.max(all_vals))
        y_span = max(y_max - y_min, 1e-12)
        pad = Y_PAD_FRAC * y_span
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.7f}"))
    ax.yaxis.offsetText.set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("phase_equator_variance.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
