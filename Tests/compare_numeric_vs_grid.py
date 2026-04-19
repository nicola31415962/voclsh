"""
Compare optimiser results against a brute-force grid search (reference) on small, tractable cases.
Default: N=1 qubit, full Pauli POVM, observables in XZ plane. Produces a plot of
variance vs observable angle for:
 - Optimiser (Adam-based worst-case search)
 - Brute-force maximum over a dense state grid (theta_s, phi_s)
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import states_functions as sf
import povm_functions as pf
import opt_variance_coef as ocf

jax.config.update("jax_enable_x64", True)

# Tunables via env vars
N = int(os.getenv("ANALYTIC_N", 2))
DENSITY_OBS = int(os.getenv("ANALYTIC_DENSITY_OBS", 16))   # observable angle resolution
DENSITY_STATE_TH = int(os.getenv("ANALYTIC_DENSITY_STATE_TH", 64))  # state theta grid (product_pure mode)
DENSITY_STATE_PH = int(os.getenv("ANALYTIC_DENSITY_STATE_PH", 64))  # state phi grid (product_pure mode)
SAMPLES_FULL_PURE = int(os.getenv("ANALYTIC_SAMPLES_FULL_PURE", 800))   # Haar pure samples for full mode
SAMPLES_FULL_MIX  = int(os.getenv("ANALYTIC_SAMPLES_FULL_MIX", 400))    # random mixed samples for full mode
STEPS = int(os.getenv("ANALYTIC_STEPS", 2000))
LR = float(os.getenv("ANALYTIC_LR", 5e-3))
POVM_AXES = os.getenv("ANALYTIC_POVM", "full").lower()  # "full" or "xz"
STATE_MODE = os.getenv("STATE_CONSTRAINT", "full")


# --- helpers ---
def build_povm():
    if POVM_AXES == "full":
        return pf.pauli_povm_single()
    if POVM_AXES == "xz":
        return pf.pauli_povm_single("X", "Z")
    raise ValueError("ANALYTIC_POVM must be 'full' or 'xz'")


def optimiser_worst_case(pcm, bm, obs):
    d = len(obs)
    D = d ** 2
    flatobs = sf.flatten_in_basis(obs, bm)
    x0 = jnp.ones((D,), dtype=jnp.float64) / D
    loss = lambda x: ocf.var_state_optimisation(x, pcm, bm, flatobs)
    loss = jax.jit(loss)
    xs, neg_var = ocf._adam_minimize(loss, x0, steps=STEPS, lr=LR)
    return float(-neg_var)


def brute_force_worst_case(pcm, bm, obs):
    flat_obs = sf.flatten_in_basis(obs, bm)
    max_var = -np.inf

    if STATE_MODE.lower() == "product_pure":
        # grid over product pure states |psi(theta,phi)>^{⊗N}
        thetas = jnp.linspace(0, jnp.pi, DENSITY_STATE_TH)
        phis = jnp.linspace(0, 2 * jnp.pi, DENSITY_STATE_PH)
        for th in thetas:
            for ph in phis:
                rho_1 = sf.qubit(th, ph)
                rho_N = pf.tensor_same(rho_1, N)
                rho_flat = sf.flatten_in_basis(rho_N, bm)
                coefs = ocf.opt_coef_state(pcm, flat_obs, rho_flat)
                var = jnp.real(ocf.variance_coef_state(rho_flat, pcm, coefs))
                max_var = max(max_var, float(var))
    else:
        # Monte Carlo over random *entangled* pure and mixed states (full constraint)
        key = jax.random.PRNGKey(0)
        d = obs.shape[0]

        # pure states
        for _ in range(SAMPLES_FULL_PURE):
            key, sub = jax.random.split(key)
            psi = jax.random.normal(sub, (d,)) + 1j * jax.random.normal(sub, (d,))
            psi = psi / jnp.linalg.norm(psi)
            rho = jnp.outer(psi, jnp.conj(psi))
            rho_flat = sf.flatten_in_basis(rho, bm)
            coefs = ocf.opt_coef_state(pcm, flat_obs, rho_flat)
            var = jnp.real(ocf.variance_coef_state(rho_flat, pcm, coefs))
            max_var = max(max_var, float(var))

        # mixed states via random Ginibre / Wishart
        for _ in range(SAMPLES_FULL_MIX):
            key, sub = jax.random.split(key)
            G = jax.random.normal(sub, (d, d)) + 1j * jax.random.normal(sub, (d, d))
            rho = G @ jnp.conj(G.T)
            rho = rho / jnp.trace(rho)
            rho_flat = sf.flatten_in_basis(rho, bm)
            coefs = ocf.opt_coef_state(pcm, flat_obs, rho_flat)
            var = jnp.real(ocf.variance_coef_state(rho_flat, pcm, coefs))
            max_var = max(max_var, float(var))

        # include maximally mixed as a cheap corner case
        rho_mm = jnp.eye(d, dtype=jnp.complex128) / d
        rho_flat = sf.flatten_in_basis(rho_mm, bm)
        coefs = ocf.opt_coef_state(pcm, flat_obs, rho_flat)
        var = jnp.real(ocf.variance_coef_state(rho_flat, pcm, coefs))
        max_var = max(max_var, float(var))
    return max_var


def main():
    sf.STATE_CONSTRAINT = STATE_MODE
    povm = build_povm()
    povm_N = pf.tensor_same(povm, N)
    pcm, bm = pf.povm_coef_matrix(povm_N)

    obs_thetas = jnp.pi * jnp.arange(DENSITY_OBS) / (4 * DENSITY_OBS)
    optimiser_vals = []
    brute_vals = []

    t0 = time.time()
    for th in obs_thetas:
        obs = pf.tensor_same(sf.qubit(th, 0.0), N)
        optimiser_vals.append(optimiser_worst_case(pcm, bm, obs))
        brute_vals.append(brute_force_worst_case(pcm, bm, obs))
        print(f"theta={float(th):.4f}: opt={optimiser_vals[-1]:.6f}, brute={brute_vals[-1]:.6f}")
    print(f"done in {time.time()-t0:.2f}s")

    obs_deg = np.array(obs_thetas) * 180 / np.pi
    opt_np = np.array(optimiser_vals)
    brute_np = np.array(brute_vals)
    diff = opt_np - brute_np
    print(f"max abs diff: {np.max(np.abs(diff)):.3e}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(obs_deg, brute_np, marker='o', label='brute-force (grid)')
    ax.plot(obs_deg, opt_np, marker='x', linestyle='--', label='optimiser')
    ax.set_xlabel('observable theta (deg, XZ plane)')
    ax.set_ylabel('worst-case variance')
    ax.set_title(f'N={N}, POVM={POVM_AXES}, mode={STATE_MODE}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('analytic_compare.png', dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
