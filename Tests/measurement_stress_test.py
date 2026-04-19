# Measurement choice stress-test: compare POVM choices for a fixed observable family.
# Varies POVM (Pauli planes vs full Pauli) and logs min/avg variance vs N.
# Uses the current STATE_CONSTRAINT (full or product_pure) from the environment.

import os
import time
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import povm_functions as pf
import states_functions as sf
import opt_variance_coef as ocf

jax.config.update("jax_enable_x64", True)

# configuration
STATE_MODE = os.getenv("STATE_CONSTRAINT", "full")
N_MIN, N_MAX = 1, 4
DENSITY = 8  # resolution for the sweep (higher → finer sampling)
# faster but stable-ish optimiser settings
STABLE_STEPS = 400
STABLE_LR = 2e-3


def build_povms() -> List[Tuple[str, Callable[[], jnp.ndarray]]]:
    """Return (label, factory) for each POVM to test."""
    return [
        ("XZ plane", lambda: pf.pauli_povm_single("X", "Z")),
        ("XY plane", lambda: pf.pauli_povm_single("X", "Y")),
        ("YZ plane", lambda: pf.pauli_povm_single("Y", "Z")),
        ("Full Pauli", lambda: pf.pauli_povm_single()),  # all 6 effects
    ]


def observables_for_povm(label: str):
    """
    Generate a list of (theta, phi) pairs with the Bloch vector lying in the POVM's span,
    so comparisons are fair (no projection loss).
    """
    if "XZ" in label:
        # sweep polar angle in XZ plane (phi=0)
        thetas = jnp.pi * jnp.arange(DENSITY) / (4 * DENSITY)
        return [(float(th), 0.0) for th in thetas]
    if "YZ" in label:
        # sweep polar angle in YZ plane (phi=pi/2)
        thetas = jnp.pi * jnp.arange(DENSITY) / (4 * DENSITY)
        return [(float(th), float(jnp.pi / 2)) for th in thetas]
    if "XY" in label:
        # stay in equator (theta=pi/2) and sweep azimuth phi in XY plane
        phis = jnp.pi * jnp.arange(DENSITY) / (2 * DENSITY)
        return [(float(jnp.pi / 2), float(ph)) for ph in phis]
    # full Pauli: reuse XZ sweep as a representative observable family
    thetas = jnp.pi * jnp.arange(DENSITY) / (4 * DENSITY)
    return [(float(th), 0.0) for th in thetas]


def solve_single_theta(theta, pcm, bm, N):
    singleobs = sf.qubit(theta[0], theta[1])
    obs = pf.tensor_same(singleobs, N)
    var, rho, oc = ocf.variance_optimisation(pcm, bm, obs)
    return var


def sweep_thetas(theta_array, pcm, bm, N):
    batched = jax.jit(jax.vmap(lambda th: solve_single_theta(th, pcm, bm, N)))
    return batched(theta_array)


def run_for_povm(label: str, povm_factory: Callable[[], jnp.ndarray]):
    povm = povm_factory()
    results = []
    for N in range(N_MIN, N_MAX + 1):
        t0 = time.time()
        povm_N = pf.tensor_same(povm, N)
        pcm, bm = pf.povm_coef_matrix(povm_N)
        # build angle list aligned with this POVM's span
        angle_list = observables_for_povm(label)
        angle_arr = jnp.array(angle_list)
        vars_jax = sweep_thetas(angle_arr, pcm, bm, N)
        vars_np = np.array(vars_jax)
        elapsed = time.time() - t0
        results.append(
            {
                "N": N,
                "min_var": float(np.min(vars_np)),
                "avg_var": float(np.mean(vars_np)),
                "time_s": elapsed,
            }
        )
    return results


def plot_results(collected):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].set_title(f"Min variance vs N (mode={STATE_MODE})")
    ax[1].set_title(f"Avg variance vs N (mode={STATE_MODE})")
    ax[0].set_xlabel("N qubits")
    ax[1].set_xlabel("N qubits")
    ax[0].set_ylabel("min Var")
    ax[1].set_ylabel("avg Var")

    markers = ["o", "s", "^", "D", "v", "X", "P", "*"]
    for idx, (label, res) in enumerate(collected):
        Ns = [r["N"] for r in res]
        mins = [r["min_var"] for r in res]
        avgs = [r["avg_var"] for r in res]
        marker = markers[idx % len(markers)]
        ax[0].plot(Ns, mins, marker=marker, label=label)
        ax[1].plot(Ns, avgs, marker=marker, label=label)

    for a in ax:
        a.legend(fontsize=8)
        a.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"measurement_stress_{STATE_MODE}.png", dpi=200)
    plt.show()


def main():
    os.environ["STATE_CONSTRAINT"] = STATE_MODE
    sf.STATE_CONSTRAINT = STATE_MODE
    # override optimiser settings for stability
    ocf.ADAM_STEPS_MAIN = STABLE_STEPS
    ocf.ADAM_LR_MAIN = STABLE_LR
    ocf.ADAM_STEPS_FIX = STABLE_STEPS
    ocf.ADAM_LR_FIX = STABLE_LR
    print(f"STATE_CONSTRAINT={STATE_MODE}")

    collected = []
    povms = build_povms()
    for label, factory in povms:
        print(f"\nPOVM: {label}")
        res = run_for_povm(label, factory)
        for r in res:
            print(
                f"  N={r['N']}: min={r['min_var']:.6f}, avg={r['avg_var']:.6f}, time={r['time_s']:.2f}s"
            )
        collected.append((label, res))

    plot_results(collected)


if __name__ == "__main__":
    main()
