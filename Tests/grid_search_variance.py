# Grid search focused on a single POVM (e.g., XZ plane) while sweeping
# state constraint mode and basic Adam optimiser hyperparameters.
# This is intended as a stress test / validation pass: does changing
# optimiser settings or locality constraints move the worst-case variance?

import os
import time
from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import states_functions as sf
import povm_functions as pf
import opt_variance_coef as ocf

jax.config.update("jax_enable_x64", True)

# ----- Grid configuration -----
# fixed POVM plane for the specific problem
POVM_AXES = ("X", "Z")

# sweep state constraints
STATE_MODES: Sequence[str] = ["full", "product_pure"]

# sweep a few optimiser settings (steps, lr)
OPT_SETTINGS: Sequence[Dict[str, float]] = [
    {"name": "fast", "steps": 500, "lr": 1e-2},
    {"name": "default", "steps": 2000, "lr": 5e-3},
    {"name": "stable", "steps": 4000, "lr": 2e-3},
    {"name": "ultra_fast", "steps": 200, "lr": 2e-2},  # push speed, check robustness
    {"name": "risky", "steps": 100, "lr": 5e-2},       # very aggressive, may expose instabilities
    {"name": "extreme", "steps": 50, "lr": 1e-1},      # likely unstable; for stress-testing only
    {"name": "burst", "steps": 30, "lr": 2e-1},        # minimal iterations, huge lr
]

# theta sampling resolution (keep fixed; higher values mainly refine sampling)
DENSITY = 2

# qubit range (not a hyperparameter, but we log scaling)
N_MIN, N_MAX = 1, 3


def optimise_variance(pcm, bm, observable, steps: int, lr: float):
    """Variant of ocf.variance_optimisation with tunable Adam settings."""
    d = len(observable)
    D = d ** 2
    flatobs = sf.flatten_in_basis(observable, bm)
    x0 = jnp.ones((D,), dtype=jnp.float64) / D

    loss = lambda x: ocf.var_state_optimisation(x, pcm, bm, flatobs)
    loss = jax.jit(loss)
    xs, neg_var = ocf._adam_minimize(loss, x0, steps=int(steps), lr=float(lr))

    flat_rho = sf.flat_state_from_mat(xs, bm)
    rho = (bm @ flat_rho).reshape((d, d))
    var = -neg_var
    return float(jnp.real(var)), rho


def run_config(state_mode: str, opt_cfg: Dict[str, float]):
    os.environ["STATE_CONSTRAINT"] = state_mode
    sf.STATE_CONSTRAINT = state_mode

    povm = pf.pauli_povm_single(*POVM_AXES)
    results = []
    for N in range(N_MIN, N_MAX + 1):
        t0 = time.time()
        povm_N = pf.tensor_same(povm, N)
        pcm, bm = pf.povm_coef_matrix(povm_N)

        thetas = jnp.pi * jnp.arange(DENSITY) / (4 * DENSITY)
        vars_list = []
        for th in thetas:
            singleobs = sf.qubit(th, 0.0)
            obs = pf.tensor_same(singleobs, N)
            var, rho = optimise_variance(pcm, bm, obs, opt_cfg["steps"], opt_cfg["lr"])
            vars_list.append(var)

        vars_np = np.array(vars_list)
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


def main():
    print(f"Grid over modes={STATE_MODES}, optimiser={[(c['name'], c['steps'], c['lr']) for c in OPT_SETTINGS]}, povm={POVM_AXES}, density={DENSITY}")
    collected = []
    for mode in STATE_MODES:
        for opt_cfg in OPT_SETTINGS:
            label = f"mode={mode}, opt={opt_cfg['name']}(steps={opt_cfg['steps']}, lr={opt_cfg['lr']})"
            print(f"\nConfig: {label}")
            res = run_config(mode, opt_cfg)
            for r in res:
                print(
                    f"  N={r['N']}: min={r['min_var']:.6f}, avg={r['avg_var']:.6f}, time={r['time_s']:.2f}s"
                )
            collected.append((label, mode, opt_cfg["name"], res))
    print("\nDone.")

    # --- simple visual comparison ---
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].set_title("Min variance vs N")
    ax[1].set_title("Avg variance vs N")
    ax[0].set_xlabel("N qubits")
    ax[1].set_xlabel("N qubits")
    ax[0].set_ylabel("min Var")
    ax[1].set_ylabel("avg Var")

    markers = ["o", "s", "^", "D", "v", "X", "P", "*"]
    for idx, (label, mode, opt_name, res) in enumerate(collected):
        x_jitter = 0.02 * (idx - len(collected) / 2)  # separate overlapping series horizontally
        y_jitter = 0.002 * (idx - len(collected) / 2)  # tiny vertical jitter to make overlapping lines visible
        Ns = [r["N"] + x_jitter for r in res]
        mins = [r["min_var"] + y_jitter for r in res]
        avgs = [r["avg_var"] + y_jitter for r in res]
        style = "-" if mode == "full" else "--"
        marker = markers[idx % len(markers)]
        ax[0].plot(Ns, mins, linestyle=style, marker=marker, label=f"{opt_name} ({mode})")
        ax[1].plot(Ns, avgs, linestyle=style, marker=marker, label=f"{opt_name} ({mode})")

    for a in ax:
        a.legend(fontsize=8)
        a.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
