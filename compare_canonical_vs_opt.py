"""
Compare worst-case variance using:
- Optimiser (state + coeffs jointly optimised).
- Canonical coefficients (Penrose pseudoinverse), maximised over states via sampling/grid.

Default: N=2, full Pauli POVM, STATE_CONSTRAINT=full.
Produces plot canonical_vs_opt.png.
"""

import os
import time
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import states_functions as sf
import povm_functions as pf
import opt_variance_coef as ocf

jax.config.update("jax_enable_x64", True)

# ---- Config (env overridable) ----
N = int(os.getenv("CANON_N", 2))
POVM_AXES = os.getenv("CANON_POVM", "full").lower()  # "full", "xz", "xy", or "yz"
STATE_MODE = os.getenv("STATE_CONSTRAINT", "full")

DENSITY_OBS = int(os.getenv("CANON_DENSITY_OBS", 16))  # observable angles in XZ plane
# Sampling for state space
DENSITY_STATE_TH = int(os.getenv("CANON_DENSITY_STATE_TH", 64))  # for product_pure
DENSITY_STATE_PH = int(os.getenv("CANON_DENSITY_STATE_PH", 64))  # for product_pure
SAMPLES_FULL_PURE = int(os.getenv("CANON_SAMPLES_FULL_PURE", 800))  # Haar pure
SAMPLES_FULL_MIX = int(os.getenv("CANON_SAMPLES_FULL_MIX", 400))    # random mixed

OPT_STEPS = int(os.getenv("CANON_OPT_STEPS", 2000))
OPT_LR = float(os.getenv("CANON_OPT_LR", 5e-3))


# ---- Helpers ----
def build_povm():
    if POVM_AXES == "full":
        return pf.pauli_povm_single()
    if POVM_AXES == "xz":
        return pf.pauli_povm_single("X", "Z")
    if POVM_AXES == "xy":
        return pf.pauli_povm_single("X", "Y")
    if POVM_AXES == "yz":
        return pf.pauli_povm_single("Y", "Z")
    raise ValueError("CANON_POVM must be 'full', 'xz', 'xy', or 'yz'")


def observable_sweep():
    # Unified in-plane angle alpha in [0, pi/2] for comparable XY/YZ/XZ paths.
    alphas = jnp.pi * jnp.arange(DENSITY_OBS) / (2 * DENSITY_OBS)
    if POVM_AXES == "xy":
        # (x, y, z) = (cos a, sin a, 0)
        return [(float(jnp.pi / 2), float(a), float(a)) for a in alphas]
    if POVM_AXES == "yz":
        # (x, y, z) = (0, cos a, sin a)
        return [(float(a), float(jnp.pi / 2), float(a)) for a in alphas]
    # xz and full: (x, y, z) = (cos a, 0, sin a)
    return [(float(a), 0.0, float(a)) for a in alphas]


def sample_states(bm) -> List[jnp.ndarray]:
    """Return a list of flat states according to STATE_MODE."""
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

    # full: sample pure + mixed + maximally mixed
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
    rho_mm_flat = sf.flatten_in_basis(rho_mm, bm)
    flat_states.append(rho_mm_flat)
    return flat_states


def canonical_variance_max(pcm, bm, obs):
    flat_obs = sf.flatten_in_basis(obs, bm)
    # canonical coefficients
    can_em = jnp.linalg.pinv(pcm)
    coefs = can_em.T @ jnp.conj(flat_obs)
    # Maximise over states with the same Adam machinery as the optimiser curve.
    max_var, _ = ocf.fix_coef_var_optimisation(pcm, bm, coefs)
    return float(max_var)


def optimiser_variance(pcm, bm, obs):
    d = len(obs)
    D = d ** 2
    flatobs = sf.flatten_in_basis(obs, bm)
    x0 = jnp.ones((D,), dtype=jnp.float64) / D
    loss = lambda x: ocf.var_state_optimisation(x, pcm, bm, flatobs)
    loss = jax.jit(loss)
    xs, neg_var = ocf._adam_minimize(loss, x0, steps=OPT_STEPS, lr=OPT_LR)
    return float(-neg_var)


def main():
    sf.STATE_CONSTRAINT = STATE_MODE
    # Keep canonical/state-only and joint optimiser runs on the same Adam budget.
    ocf.ADAM_STEPS_FIX = OPT_STEPS
    ocf.ADAM_LR_FIX = OPT_LR
    povm = build_povm()
    povm_N = pf.tensor_same(povm, N)
    pcm, bm = pf.povm_coef_matrix(povm_N)

    sweep = observable_sweep()
    can_vals = []
    opt_vals = []
    alpha_vals = []
    x_label = "in-plane angle alpha (deg)"

    t0 = time.time()
    for th, ph, alpha in sweep:
        obs = pf.tensor_same(sf.qubit(th, ph), N)
        can_vals.append(canonical_variance_max(pcm, bm, obs))
        opt_vals.append(optimiser_variance(pcm, bm, obs))
        alpha_vals.append(alpha)
        print(f"alpha={float(alpha):.4f}: canonical={can_vals[-1]:.6f}, optimiser={opt_vals[-1]:.6f}")
    print(f"done in {time.time()-t0:.2f}s")

    obs_deg = np.array(alpha_vals) * 180 / np.pi
    can_np = np.array(can_vals)
    opt_np = np.array(opt_vals)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(obs_deg, can_np, marker="o", label="canonical coeffs (Penrose) max over states")
    ax.plot(obs_deg, opt_np, marker="x", linestyle="--", label="optimiser (state+coeffs)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("worst-case variance")
    ax.set_title(f"N={N}, POVM={POVM_AXES}, mode={STATE_MODE}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("canonical_vs_opt.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
