# JAX mirror of plane_proj_XZ.py (kept variable names/flow)
# Minor changes:
# - Uses the JAX modules
# - Removes joblib parallel for simplicity (can reintroduce with pmap/vmap later)
#
# This is the experiment script for the XZ-plane POVM problem. It is the file a student would
# copy and adapt for a new problem: define a new POVM (single_povm), a new family of observables
# (the singleobs / obs construction in the loops), and adjust N_min/N_max and the output directory.
# The three imported modules (states_functions, povm_functions, opt_variance_coef) are the core
# engine and should not need to be modified.
import warnings
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")


import os
import time
import numpy as np
import matplotlib.pyplot as plt

# JAX
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# homebrewed modules for the actual variance optimisation and handling of different functions
# toggle: set to 1 for product-pure, 0 for full
# USE_PRODUCT_PURE selects the state parameterisation used during optimisation (see
# states_functions.flat_state_from_mat). Set to 1 to restrict to product pure states (faster,
# but misses entangled worst-case states); 0 uses the full density matrix parameterisation.
USE_PRODUCT_PURE = 0

# constrain states unless overridden by env
if "STATE_CONSTRAINT" not in os.environ:
    os.environ["STATE_CONSTRAINT"] = "product_pure" if USE_PRODUCT_PURE else "full"

import states_functions as sf
# ensure module-level toggle matches env even in long-lived sessions
sf.STATE_CONSTRAINT = os.environ["STATE_CONSTRAINT"]
import povm_functions as pf
import opt_variance_coef as ocf

# global variables
# single_povm: the single-qubit POVM for the XZ plane — the 4 projectors onto the ±1 eigenstates
# of X and Z, each weighted by 1/2. For a different problem, replace this line with a call to
# a function that returns your POVM as an (n, d, d) array and passes it to pf.povm_coef_matrix.
single_povm = pf.pauli_povm_single('X','Z') # single qubit plane POVM

# density: number of theta values sampled per qubit count N. The angles are theta_j = j*pi/(4*density)
# for j in range(density), covering a quarter of the Bloch sphere equator (XZ plane symmetry reduces
# the full [0, pi/2] range to [0, pi/4]). Increase density for a finer angular resolution.
density = 2  # number of different projectors considered

N_min = 1     # minimal dimension considered
N_max = 5     # maximal dimension considered

# saving directory (created if non-existing)
directory= 'plane_proj_XZ_jax'
os.makedirs(directory, exist_ok=True)

# ---- batched solver wrapper (CPU / 1-GPU / multi-GPU) ----

def solve_single_theta(theta, pcm, bm, N):
    # Runs variance_optimisation for a single observable at polar angle theta in the XZ plane
    # (phi=0 fixes the observable to the XZ plane). The single-qubit observable is tensored N
    # times to get the N-qubit target observable. Returns only the variance (scalar float),
    # discarding the optimal state and coefficients since we only need the variance curve.
    phi = 0.0
    singleobs = sf.qubit(theta, phi)
    obs = pf.tensor_same(singleobs, N)
    var, rho, oc = ocf.variance_optimisation(pcm, bm, obs)
    return var

def solve_thetas(theta_array, pcm, bm, N):
    """
    theta_array: (B,) values to sweep
    pcm, bm: POVM coeff matrix and basis
    N: number of qubits
    """
    # Vectorises solve_single_theta over an array of theta values. On a single CPU or GPU, jit+vmap
    # compiles one batched call that avoids Python loop overhead and is significantly faster than
    # calling solve_single_theta in a Python for-loop. On multiple GPUs, the batch is split evenly
    # across devices with zero-padding to handle non-divisible sizes, and results are stitched back
    # together. The device selection is automatic — no manual configuration needed.

    # vmap for single device (CPU or single GPU)
    def single_device(theta_array):
        batched = jax.jit(
            jax.vmap(lambda th: solve_single_theta(th, pcm, bm, N))
        )
        return batched(theta_array)

    # pmap for multi-GPU cluster
    def multi_device(theta_array, gpu_devices):
        num_devices = len(gpu_devices)
        B = theta_array.shape[0]
        per_dev = (B + num_devices - 1) // num_devices
        total = per_dev * num_devices

        theta_pad = jnp.pad(theta_array, (0, total - B))
        theta_shard = theta_pad.reshape(num_devices, per_dev)

        @jax.pmap
        def per_dev_solve(theta_local):
            return jax.vmap(lambda th: solve_single_theta(th, pcm, bm, N))(theta_local)

        vars_sh = per_dev_solve(theta_shard)

        # de-shard and trim padding
        vars_flat = vars_sh.reshape(total)[:B]
        return vars_flat

    # ---- device selection (safe) ----
    all_devices = jax.devices()
    gpu_devices = [d for d in all_devices if d.platform == "gpu"]

    if len(gpu_devices) > 1:
        # multi-GPU
        return multi_device(theta_array, gpu_devices)
    else:
        # CPU or single GPU → just use jit+vmap
        return single_device(theta_array)


# -------

# --- helper function mirroring your structure ---
def compute_var_for_theta(j, density, N, pcm, bm):
    # Scalar fallback version of the per-theta computation (same logic as solve_single_theta,
    # but takes j and density as arguments and computes theta internally). Not used in the main
    # loop (which calls the faster solve_thetas), but kept as a readable reference for the
    # structure of a single computation: build observable → call variance_optimisation → return var.
    theta = j*np.pi/(4*density)
    phi = 0
    singleobs = sf.qubit(theta, phi)
    obs = pf.tensor_same(singleobs, N)
    var, rho, oc = ocf.variance_optimisation(pcm, bm, obs)
    return float(var)

# --- main loop ---
# Iterates over qubit counts N from 1 to N_max. For each N:
#   1. Builds the N-qubit POVM as tensor_same(single_povm, N) and computes pcm, bm via povm_coef_matrix.
#   2. Computes the canonical estimator variance as a benchmark: the canonical coefficients are the
#      rows of pinv(pcm), and fix_coef_var_optimisation finds the worst-case state for those fixed
#      coefficients. This gives an upper bound on the variance achievable by the canonical estimator.
#   3. Computes the optimal variance curve over the theta sweep using solve_thetas (vectorised over
#      the batch of angles). This is the tightest upper bound on the variance for any estimator.
#   4. Saves results to .npy files for later analysis or plotting.
# To adapt this for a different problem: replace single_povm with your POVM, replace sf.qubit and
# pf.tensor_same with your own observable construction, and adjust the theta sweep as needed.
all_vec = []
all_can = []
min_vec = []
avg_vec = []
times = []

for N in range(1, N_max+1):
    time_start = time.time()
    filename = directory + f'_{N}'

    # tensor POVM definition
    povm = pf.tensor_same(single_povm, N)
    pcm, bm = pf.povm_coef_matrix(povm)
    can_em = jnp.linalg.pinv(pcm)  # canonical inverse

    # canonical inversion
    single_obs = sf.qubit(0, 0)
    obs = pf.tensor_same(single_obs, N)
    cc = can_em.T @ jnp.conj(sf.flatten_in_basis(obs, bm))
    var_can, rho_can = ocf.fix_coef_var_optimisation(pcm, bm, cc)
    np.save(f'{directory}/{filename}_can', np.array(var_can))

    # optimal upper bound on variance (simple loop over theta; could vmap)
    thetas = jnp.pi * jnp.arange(density) / (4 * density)
    var_vec = solve_thetas(thetas, pcm, bm, N)
    var_vec = np.array(var_vec)   # convert to NumPy for saving

    np.save(f'{directory}/{filename}_opt', var_vec)

    all_vec.append(var_vec)
    all_can.append(float(var_can))
    min_vec.append(float(np.min(var_vec)))
    avg_vec.append(float(np.mean(var_vec)))

    time_N = time.time()
    print(N, time_N-time_start)
    times.append(time_N-time_start)

np.save(f'{directory}/time_log', np.array(times))

# --- Plots (clean legends & labels) ---
# Three-panel figure: (left) optimal and canonical variance vs theta for each N; (middle) ratio
# of optimal to canonical, showing how much the optimised estimator improves on the canonical one;
# (right) scaling of the minimum and average optimal variance with qubit number N, alongside the
# canonical scaling, to assess how the advantage (or disadvantage) grows with system size.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

color_lite = [
    'xkcd:azure','xkcd:kelly green','xkcd:bright orange','xkcd:red',
    'xkcd:lavender','xkcd:golden yellow','xkcd:maroon','xkcd:light grey',
    'xkcd:robin egg blue','xkcd:medium brown'
]
color_lite_dark = [
    'xkcd:royal blue','xkcd:forest green','xkcd:burnt orange','xkcd:scarlet',
    'xkcd:violet','xkcd:mustard','xkcd:steel grey','xkcd:bright sky blue','xkcd:earth'
]

thetas = np.pi * np.arange(density) / (2 * density)

goldenratio = 1.618
imagewidth = 7
nrow, ncol = 1, 3
fig, ax = plt.subplots(nrow, ncol, figsize=[ncol*imagewidth, nrow*imagewidth/goldenratio])

# --- Titles & labels ---
ax[0].set_title('Optimal variance')
ax[1].set_title('Ratio with canonical case')
ax[2].set_title('Scaling with number of qubits')

ax[0].set_ylabel(r'Var')
ax[1].set_ylabel(r'Var$_{\mathrm{opt}}$/Var$_{\mathrm{can}}$')
ax[0].set_xlabel(r'$\theta$')
ax[1].set_xlabel(r'$\theta$')
ax[2].set_ylabel('min Var')
ax[2].set_xlabel('# qubits')

# --- Panels 0 & 1: per-qubit curves ---
for N in range(N_max):
    label_q = f'{N+1} qubit' if N == 0 else f'{N+1} qubits'

    # Panel 0: optimal (solid) + canonical (dashed)
    ax[0].plot(thetas, all_vec[N], c=color_lite[N], label=label_q)
    ax[0].hlines(all_can[N], 0, np.pi/2, color=color_lite_dark[N], linestyle='--')

    # Panel 1: standard ratio (θ-based)
    ax[1].plot(thetas, all_vec[N] / all_can[N], c=color_lite[N], label=label_q)

# --- Panel 2: scaling summary ---
xN = np.arange(1, N_max+1)
ax[2].plot(xN, min_vec, marker='o', label='best optimal scaling')
ax[2].plot(xN, avg_vec, marker='x', linestyle='--', label='average optimal scaling')
ax[2].plot(xN, all_can, marker='^', linestyle=':', label='canonical scaling')

# --- Legends (3 separate boxes below each subplot) ---
# Left panel: line style box
style_handles = [
    Line2D([0],[0], color='black', lw=2, linestyle='-',  label='optimal (solid)'),
    Line2D([0],[0], color='black', lw=2, linestyle='--', label='canonical (dashed)'),
]
ax[0].legend(
    handles=style_handles, fontsize=9, frameon=True, fancybox=True, framealpha=0.9,
    loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=1, title='line styles'
)

# Middle panel: qubit-count legend
ax[1].legend(
    fontsize=9, frameon=True, fancybox=True, framealpha=0.9,
    loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=1, title='qubits'
)

# Right panel: scaling legend
ax[2].legend(
    fontsize=9, frameon=True, fancybox=True, framealpha=0.9,
    loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=1
)

# --- Layout adjustment ---
fig.subplots_adjust(bottom=0.33)
plt.savefig(f'{directory}/plane_proj.pdf', transparent=True, bbox_inches='tight')
plt.show()
