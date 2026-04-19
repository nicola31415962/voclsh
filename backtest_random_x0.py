"""
Backtest optimiser stability vs random initial points x0.
Runs R repeats for a fixed configuration (N=3 qubits, XZ-plane POVM,
state constraint taken from env) and plots min/avg variance per run.
"""
import os
import time
from typing import Sequence, Dict

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import states_functions as sf
import povm_functions as pf
import opt_variance_coef as ocf

jax.config.update("jax_enable_x64", True)

# Config (edit here or via CLI env)
R = int(os.getenv("BACKTEST_RUNS", 10))
STEPS = int(os.getenv("BACKTEST_STEPS", 2000))
LR = float(os.getenv("BACKTEST_LR", 5e-3))
NOISE = float(os.getenv("BACKTEST_NOISE", 0.05))  # scale of x0 jitter
DENSITY = int(os.getenv("BACKTEST_DENSITY", 2))   # theta resolution
N = int(os.getenv("BACKTEST_N", 3))
POVM_AXES = ("X", "Z")


def optimise_with_random_x0(pcm, bm, observable, key, steps: int, lr: float, noise: float):
    d = len(observable)
    D = d ** 2
    flatobs = sf.flatten_in_basis(observable, bm)
    base = jnp.ones((D,), dtype=jnp.float64) / D
    noise_vec = noise * jax.random.normal(key, (D,))
    x0 = base + noise_vec

    loss = lambda x: ocf.var_state_optimisation(x, pcm, bm, flatobs)
    loss = jax.jit(loss)
    xs, neg_var = ocf._adam_minimize(loss, x0, steps=int(steps), lr=float(lr))

    flat_rho = sf.flat_state_from_mat(xs, bm)
    rho = (bm @ flat_rho).reshape((d, d))
    var = -neg_var
    return float(jnp.real(var)), rho


def run_single_backtest(key) -> Dict[str, float]:
    povm = pf.pauli_povm_single(*POVM_AXES)
    povm_N = pf.tensor_same(povm, N)
    pcm, bm = pf.povm_coef_matrix(povm_N)

    thetas = jnp.pi * jnp.arange(DENSITY) / (4 * DENSITY)

    vars_list = []
    subkeys = jax.random.split(key, len(thetas))
    for th, k in zip(thetas, subkeys):
        singleobs = sf.qubit(th, 0.0)
        obs = pf.tensor_same(singleobs, N)
        var, _ = optimise_with_random_x0(pcm, bm, obs, k, STEPS, LR, NOISE)
        vars_list.append(var)

    vars_np = np.array(vars_list)
    return {
        "min_var": float(np.min(vars_np)),
        "avg_var": float(np.mean(vars_np)),
        "vars": vars_np,
    }


def main():
    # propagate env constraint to module
    sf.STATE_CONSTRAINT = os.getenv("STATE_CONSTRAINT", "full")
    print(f"STATE_CONSTRAINT={sf.STATE_CONSTRAINT}, N={N}, runs={R}, steps={STEPS}, lr={LR}, noise={NOISE}")

    base_key = jax.random.PRNGKey(int(time.time()))
    run_keys = jax.random.split(base_key, R)

    mins, avgs = [], []
    all_vars = []
    t0 = time.time()
    for i, k in enumerate(run_keys):
        res = run_single_backtest(k)
        mins.append(res["min_var"])
        avgs.append(res["avg_var"])
        all_vars.append(res["vars"])
        print(f"run {i:02d}: min={res['min_var']:.6f}, avg={res['avg_var']:.6f}")
    elapsed = time.time() - t0
    print(f"done in {elapsed:.2f}s")

    runs = np.arange(R)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(runs, mins, marker="o", label="min variance across thetas")
    ax.plot(runs, avgs, marker="x", linestyle="--", label="avg variance across thetas")
    ax.set_xlabel("run index (different x0)")
    ax.set_ylabel("variance")
    ax.set_title(f"Backtest (N={N}, POVM={POVM_AXES}, density={DENSITY})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("backtest_random_x0.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
