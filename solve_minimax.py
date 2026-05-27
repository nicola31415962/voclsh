# sdp/solve_minimax.py
#
# Full minimax SDP for the worst-case variance of a POVM estimator:
#
#   V* = max_rho  V(rho)  =  max_rho  min_c  Var(c, rho)
#
# Derivation sketch
# -----------------
# For fixed rho, the BLUE gives:
#   V(rho) = shadow_norm(rho) - [Tr(O rho)]^2
#           = b^T W(rho)^{-1} b  -  e(rho)^2
#
# where  W(rho) = P^T D(rho)^{-1} P,  e(rho) = Tr(O rho),
#        b = flatten_in_basis(O, bm)  (nz-vector),
#        P = pcm  (n x nz POVM coefficient matrix),
#        D(rho) = diag(p_1(rho),...,p_n(rho)).
#
# Variational identity (Legendre):
#   b^T W^{-1} b = max_y { 2 b^T y  -  y^T W y }
#                = max_y { 2 b^T y  -  sum_i (Py)_i^2 / p_i }
#
# Schur complement: (Py)_i^2 / p_i <= t_i  <=>  [[t_i, (Py)_i],[(Py)_i, p_i]] >=0
# Schur complement: e^2 <= s              <=>  [[s, e],[e, 1]] >= 0
#
# Full minimax SDP (maximise over rho, y, t_1..n, s):
#
#   max   2 b^T y  -  sum_i t_i  -  s
#   s.t.  [[t_i, (Py)_i], [(Py)_i, p_i(rho)]] >= 0   for i=1..n
#         [[s, e(rho)], [e(rho), 1]]            >= 0
#         rho >= 0,  Tr(rho) = 1
#
# All constraints are linear in (rho, y, t, s) => valid SDP.
#
# Usage:
#   python sdp/solve_minimax.py --N 1 --theta 0.0
#   python sdp/solve_minimax.py --N 2 --theta 0.785

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cvxpy as cp

import jax
jax.config.update("jax_enable_x64", True)

os.environ.setdefault("STATE_CONSTRAINT", "full")

import povm_functions as pf
import states_functions as sf
import opt_variance_coef as ocf

# ---- arguments ----
parser = argparse.ArgumentParser()
parser.add_argument('--N',     type=int,   default=1)
parser.add_argument('--theta', type=float, default=0.0)
parser.add_argument('--no_adam', action='store_true',
                    help='skip Adam optimiser, compare SDP only against analytical lower bound')
args = parser.parse_args()

N     = args.N
theta = args.theta
phi   = 0.0

print(f"Full minimax SDP  |  N={N}, theta={theta:.4f}")
print("=" * 60)

# ---- build POVM and observable ----
single_povm = pf.pauli_povm_single('X', 'Z')
povm        = pf.tensor_same(single_povm, N)   # (n, d, d)
pcm, bm     = pf.povm_coef_matrix(povm)

single_obs  = sf.qubit(theta, phi)
obs         = pf.tensor_same(single_obs, N)    # (d, d)

d   = obs.shape[0]
n   = povm.shape[0]
D   = d * d
nz  = pcm.shape[1]

print(f"  d={d}, n_effects={n}, subspace_dim={nz}")

# real versions (imaginary parts are numerical noise for XZ POVM)
pcm_np = np.real(np.array(pcm))           # (n, nz)
bm_np  = np.real(np.array(bm))            # (D, nz)
obs_np = np.array(obs, dtype=complex)     # (d, d)
obs_flat = np.real(np.array(sf.flatten_in_basis(obs, bm)))  # (nz,)

povm_np = [np.array(povm[i], dtype=complex) for i in range(n)]

# ---- [1] Adam alternating optimiser (reference) ----
if not args.no_adam:
    print("\n[1] Adam alternating optimiser...")
    var_adam, rho_adam, c_adam = ocf.variance_optimisation(pcm, bm, obs)
    var_adam = float(np.real(var_adam))
    rho_adam = np.array(rho_adam, dtype=complex)
    print(f"    V* (Adam)   = {var_adam:.8f}")
else:
    var_adam = None
    print("\n[1] Adam optimiser skipped (--no_adam)")

# ---- [2] Full minimax SDP ----
print("\n[2] Full minimax SDP...")

rho_var = cp.Variable((d, d), hermitian=True, name='rho')
y_var   = cp.Variable(nz,  name='y')
t_vars  = [cp.Variable(name=f't{i}') for i in range(n)]
s_var   = cp.Variable(name='s')

# p_i(rho) = Tr(E_i rho)  [real, affine in rho_var]
p_exprs = [cp.real(cp.trace(povm_np[i] @ rho_var)) for i in range(n)]

# (Py)_i = pcm[i,:] @ y_var  [affine in y_var]
py_exprs = [float(pcm_np[i, :]) @ y_var if nz == 1
            else pcm_np[i, :] @ y_var
            for i in range(n)]

# e(rho) = Tr(O rho)  [real, affine in rho_var]
e_expr = cp.real(cp.trace(obs_np @ rho_var))

# objective: 2 b^T y - sum_i t_i - s
obj = cp.Maximize(2.0 * obs_flat @ y_var - cp.sum(cp.hstack(t_vars)) - s_var)

constraints = [
    rho_var >> 0,
    cp.real(cp.trace(rho_var)) == 1,
]

# 2x2 PSD blocks for each effect (Schur: t_i p_i >= (Py)_i^2)
for i in range(n):
    t_i  = cp.reshape(t_vars[i], (1, 1))
    py_i = cp.reshape(py_exprs[i], (1, 1))
    p_i  = cp.reshape(p_exprs[i], (1, 1))
    block = cp.bmat([[t_i, py_i], [py_i, p_i]])
    constraints.append(block >> 0)

# 2x2 PSD block for squared expectation value (Schur: s >= e(rho)^2)
e_m  = cp.reshape(e_expr, (1, 1))
s_m  = cp.reshape(s_var,  (1, 1))
one  = np.ones((1, 1))
block_s = cp.bmat([[s_m, e_m], [e_m, one]])
constraints.append(block_s >> 0)

prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.CLARABEL)

if prob.status not in ("optimal", "optimal_inaccurate"):
    print(f"    SDP solver status: {prob.status}")
    sys.exit(1)

var_sdp  = float(prob.value)
rho_sdp  = rho_var.value
y_sdp    = y_var.value
s_sdp    = float(s_var.value)
e_sdp    = float(np.real(np.trace(obs_np @ rho_sdp)))

print(f"    V* (SDP)    = {var_sdp:.8f}  [{prob.status}]")

# manual verification at SDP solution
shadow_sdp = float(2.0 * obs_flat @ y_sdp - sum(float(t_vars[i].value) for i in range(n)))
print(f"    shadow norm = {shadow_sdp:.8f}")
print(f"    e(rho*)^2   = {e_sdp**2:.8f}  (s = {s_sdp:.8f})")
print(f"    2b^Ty - Σt - s  = {shadow_sdp - s_sdp:.8f}  (should match V*)")

# ---- [3] V(rho_sdp) via direct BLUE ----
print("\n[3] V(rho_sdp*) by direct BLUE formula...")
p_sdp    = np.array([float(np.real(np.trace(povm_np[i] @ rho_sdp))) for i in range(n)])
p_sdp    = np.clip(p_sdp, 1e-12, None)
p_sdp   /= p_sdp.sum()
D_inv    = np.diag(1.0 / p_sdp)
W        = pcm_np.T @ D_inv @ pcm_np           # (nz, nz)
W_inv    = np.linalg.inv(W)
shadow_direct = float(obs_flat @ W_inv @ obs_flat)
e_direct      = float(np.real(np.trace(obs_np @ rho_sdp)))
V_direct      = shadow_direct - e_direct**2
print(f"    shadow norm (BLUE) = {shadow_direct:.8f}")
print(f"    e(rho*)^2          = {e_direct**2:.8f}")
print(f"    V(rho_sdp*)        = {V_direct:.8f}")

# ---- [4] Summary ----
print("\n" + "=" * 60)
print("Summary")
if var_adam is not None:
    print(f"  V* (Adam)       = {var_adam:.8f}")
print(f"  V* (SDP)        = {var_sdp:.8f}")
print(f"  V(rho_sdp BLUE) = {V_direct:.8f}  (sanity: should match SDP)")
if var_adam is not None:
    gap = abs(var_sdp - var_adam)
    print(f"  |SDP - Adam|    = {gap:.2e}")
    tol = 1e-3
    if gap < tol:
        print(f"\n  MINIMAX SDP CONFIRMED  (gap < {tol})")
    else:
        print(f"\n  WARNING: gap = {gap:.2e} > {tol}")
        print("  Either Adam has not converged or the SDP formulation needs review.")

# ---- [5] SDP worst-case state info ----
print("\n[4] SDP worst-case state rho*:")
eigs = np.sort(np.real(np.linalg.eigvalsh(rho_sdp)))[::-1]
print(f"    eigenvalues = {np.round(eigs, 6)}")
rank = int(np.sum(eigs > 1e-4))
print(f"    rank = {rank}  ({'pure' if rank == 1 else 'mixed'})")
print(f"    Tr(O rho*) = {e_direct:.6f}")
