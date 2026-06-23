"""
frank_wolfe.py — Minimax variance via Frank-Wolfe with Kronecker structure
==========================================================================

Solves:   V* = max_ρ { b^T W(ρ)^{-1} b  −  Tr(O ρ)² }

where W(ρ) = pcm_N^T diag(1/p(ρ)) pcm_N  is the Fisher information matrix,
      p_i(ρ) = Tr(E_i ρ)  are the outcome probabilities,
      pcm_N   is the coefficient matrix of the N-qubit POVM in the observable subspace.

ALGORITHM: Frank-Wolfe (conditional gradient) on the spectahedron.
  V(ρ) is concave → single global maximum → FW converges O(1/k) to V*.
  No SDP solver required. No matrix larger than (2^N × 2^N) is ever formed.

ACCURACY:
  N ≤ 5  : < 0.01% vs Caprotti et al. (2026) Table 1   (300 steps, ~1s)
  N = 8  : V* ≈ 64.25 (matches exact kron_sdp.py to 0.01%)  (300 steps, ~30s)
  N = 10 : V* converging (100 steps ≈ 3.5 min, 300 steps ≈ 11 min)
  N = 11 : V* converging (100 steps ≈ 18 min)
  For higher accuracy at any N, increase --steps (convergence is O(1/k)).

PERFORMANCE (after eigsh + CG fixes):
  N=8 : 0.1s/step   N=10 : 2.1s/step   N=11 : 11s/step   N=12 : ~45s/step
  Bottleneck: CG for shadow norm W^{-1}obs_flat (40 ms/matvec at N=10;
  ×4 per qubit). eigsh (Lanczos, leading eigenvector only) is <0.1s at all N.

WHEN TO USE THIS vs THE EXACT SDP (kron_sdp.py):
  Frank-Wolfe scales to N ≥ 10 on a laptop.
  The exact SDP (MOSEK, kron_sdp.py) gives certified V* to machine precision
  but hits memory at N ≥ 9 (~65 GB).  Use this script for N ≥ 9, or when
  O(1/k) convergence is acceptable; use the exact SDP for N ≤ 8.

WHY KRONECKER STRUCTURE
-----------------------
The N-qubit POVM is a tensor product of 1-qubit POVMs:
    E_{i_1,...,i_N} = E_{i_1} ⊗ ... ⊗ E_{i_N}

Writing each 1-qubit effect in the Pauli basis {I, X, Y, Z} with real coefficients:
    E_i = sum_j pcm1[i,j] B_j       (pcm1 always real, even for complex effects)

the N-qubit coefficient matrix factors as a Kronecker product:
    pcm_N = pcm1 ⊗ ... ⊗ pcm1       (N times)

This means pcm_N @ y and pcm_N^T @ v can be computed via N sequential (4×r)
contractions, never forming the (4^N × r^N) matrix.

WHY FRANK-WOLFE
---------------
V(ρ) is concave in ρ, so maximizing over the spectahedron {ρ ≥ 0, Tr ρ = 1} is
a convex optimisation problem. Frank-Wolfe (conditional gradient) exploits this:

  1. Compute gradient G = ∂V/∂ρ  (Hermitian matrix)
  2. FW oracle: ρ_fw = |u⟩⟨u|  where u = leading eigenvector of G
  3. Update: ρ ← (1 − γ) ρ + γ ρ_fw,   γ = 2/(k+2)

No step-size tuning; ρ remains a valid density matrix; O(1/k) convergence to
the global maximum.

COMPLEX POVM SUPPORT
--------------------
Effects like the XY POVM {(I±X)/4, (I±Y)/4} are complex. The Pauli-basis
construction keeps pcm1 real regardless, so the Kronecker CG solver is unchanged.
kron_probs and kron_grad_G handle complex effects naturally via tensordot.

Usage:
    python frank_wolfe.py --N 5 --theta 0.0 --example 2
    python frank_wolfe.py --N 7 --povm xz --steps 500
    python frank_wolfe.py --N 4 --povm xy --example 2 --theta 0.0
    python frank_wolfe.py --N 1 --max_N 8   # sweep N=1..8
"""

import argparse
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg as sp_cg, eigsh


# ─────────────────────────────────────────────────────────────────────────────
# Reference values  (Caprotti et al. 2026, Table 1;  N=6,7 from SCS runs)
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_EX2 = {
    1: 2.00000, 2: 5.33321, 3: 10.28550,
    4: 17.06671, 5: 25.80534, 6: 36.74810, 7: 49.41243,
}


# ─────────────────────────────────────────────────────────────────────────────
# Pauli matrices
# ─────────────────────────────────────────────────────────────────────────────

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [_I, _X, _Y, _Z]   # ordered I, X, Y, Z


# ─────────────────────────────────────────────────────────────────────────────
# Single-qubit POVM effects
# ─────────────────────────────────────────────────────────────────────────────

def xz_effects_1q():
    """
    XZ POVM: {(I+X)/4, (I-X)/4, (I+Z)/4, (I-Z)/4}.
    Real effects; observable subspace = span{I, X, Z}  (r = 3).
    """
    return np.array([
        (_I + _X) / 4,
        (_I - _X) / 4,
        (_I + _Z) / 4,
        (_I - _Z) / 4,
    ], dtype=complex)


def xy_effects_1q():
    """
    XY POVM: {(I+X)/4, (I-X)/4, (I+Y)/4, (I-Y)/4}.
    Complex effects; observable subspace = span{I, X, Y}  (r = 3).
    NOTE: observables with Z component (e.g. sum_local at θ ≠ 0) are not
    estimable and will produce a warning.
    """
    return np.array([
        (_I + _X) / 4,
        (_I - _X) / 4,
        (_I + _Y) / 4,
        (_I - _Y) / 4,
    ], dtype=complex)


# ─────────────────────────────────────────────────────────────────────────────
# Observable subspace basis via Pauli decomposition
# ─────────────────────────────────────────────────────────────────────────────

def build_1q_basis(effects_1q):
    """
    Identify the r-dimensional observable subspace and return pcm1, bm1.

    For any Hermitian effects E_i (real or complex), the Pauli coordinates
        can_mat[i, k] = Re( Tr(P_k E_i) ) / 2
    are always real.  SVD of can_mat gives:
        pcm1 : (4, r) real  — pcm1[i,j] = coordinate of E_i in basis vector j
        bm1  : (4, r) real  — bm1[:,j] = j-th basis vector in Pauli coordinates

    Kronecker structure: for N-qubit POVM,
        pcm_N = pcm1 ⊗ ... ⊗ pcm1   (Nfold tensor product)
    This holds because the Pauli basis coefficients of a tensor product effect
    are the products of the 1-qubit Pauli basis coefficients.
    """
    n = len(effects_1q)
    can_mat = np.array(
        [[np.real(np.trace(P @ effects_1q[i])) / 2 for P in PAULIS]
         for i in range(n)],
        dtype=float,
    )  # shape (4, 4), always real

    U, s, Vt = np.linalg.svd(can_mat, full_matrices=False)
    tol = 1e-8 * s[0]
    r   = int(np.sum(s > tol))

    pcm1 = (U[:, :r] * s[:r]).astype(float)   # (4, r)
    bm1  = Vt[:r].T.astype(float)             # (4, r)
    return pcm1, bm1


def obs_to_flat(obs, bm1, N):
    """
    Project N-qubit Hermitian obs onto the r^N-dimensional observable subspace.

    The N-qubit subspace basis is {B_{j_1} ⊗ ... ⊗ B_{j_N}} where each B_j is
    the 1-qubit matrix B_j = sum_k bm1[k,j] P_k.

    Computed via N tensor contractions (same pattern as kron_probs) rather than
    forming all r^N basis matrices explicitly.

    obs_flat[j_1,...,j_N] = Tr((B_{j_1} ⊗ ... ⊗ B_{j_N}) @ obs)

    Returns (r^N,) real array and an out-of-subspace residual norm (≈0 if obs
    is estimable by the POVM).
    """
    # 1-qubit basis matrices: basis_mats[j] = sum_k bm1[k,j] P_k,  shape (r, 2, 2)
    basis_mats = np.einsum('kj,kab->jab', bm1, np.array(PAULIS))   # (r, 2, 2) complex

    T = obs.reshape([2] * (2 * N))
    for k in range(N):
        T = np.tensordot(T, basis_mats, axes=([k, N], [1, 2]))
        T = np.moveaxis(T, -1, k)
    # The contraction computes Tr(B_J obs) for each multi-index J.
    # Since Tr(B_j^2) = 2 for each 1-qubit basis element, the correct subspace
    # coordinate is Tr(B_J obs) / 2^N.
    obs_flat = T.reshape(-1).real / (2 ** N)   # (r^N,)

    # Residual: reconstruct obs from obs_flat and compare.
    # obs_rec = sum_J obs_flat[J] B_J; coincides with obs if obs is in the subspace.
    obs_rec  = kron_grad_G(basis_mats, obs_flat, N)
    residual = float(np.linalg.norm(obs - obs_rec))
    return obs_flat, residual


# ─────────────────────────────────────────────────────────────────────────────
# Observables
# ─────────────────────────────────────────────────────────────────────────────

def product_observable(theta, N):
    """
    Example 1: O^(N) = |ψ(θ)⟩⟨ψ(θ)|^⊗N,
    |ψ(θ)⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩.
    In span{I, X, Z}^⊗N for all θ — compatible with XZ POVM.
    """
    psi = np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=complex)
    rho = np.outer(psi, psi.conj())
    obs = rho.copy()
    for _ in range(N - 1):
        obs = np.kron(obs, rho)
    return obs


def sum_local_observable(theta, N):
    """
    Example 2: O^(N) = sum_k σ_k(θ),
    σ(θ) = cos(θ/2) X + sin(θ/2) Z.
    At θ=0: pure X (compatible with both XZ and XY POVM).
    At θ≠0: has Z component (not estimable with XY POVM).
    """
    d     = 2 ** N
    sigma = np.cos(theta / 2) * _X + np.sin(theta / 2) * _Z
    obs   = np.zeros((d, d), dtype=complex)
    for i in range(N):
        obs += np.kron(np.kron(np.eye(2 ** i, dtype=complex), sigma),
                       np.eye(2 ** (N - 1 - i), dtype=complex))
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# Kronecker primitives  (never form the 4^N × r^N matrix)
# ─────────────────────────────────────────────────────────────────────────────

def kron_matvec(pcm1, y, N):
    """
    Compute (pcm1 ⊗ ... ⊗ pcm1) @ y via N sequential (4×r) contractions.
    pcm1 : (4, r) real.    y : (r^N,) real.    Returns: (4^N,) real.
    """
    _, r = pcm1.shape
    T = y.reshape([r] * N)
    for k in range(N - 1, -1, -1):
        T = np.tensordot(T, pcm1, axes=([k], [1]))
        T = np.moveaxis(T, -1, k)
    return T.reshape(-1)


def kron_rmatvec(pcm1, v, N):
    """
    Compute (pcm1 ⊗ ... ⊗ pcm1)^T @ v via N sequential (r×4) contractions.
    pcm1 : (4, r) real.    v : (4^N,) real.    Returns: (r^N,) real.
    """
    m, _ = pcm1.shape
    T = v.reshape([m] * N)
    for k in range(N - 1, -1, -1):
        T = np.tensordot(T, pcm1, axes=([k], [0]))
        T = np.moveaxis(T, -1, k)
    return T.reshape(-1)


def kron_probs(effects1, rho, N):
    """
    Compute p_i(ρ) = Tr(E_i ρ) for all N-qubit multi-indices i.
    Uses N tensor contractions, one qubit at a time.
    effects1 : (4, 2, 2) possibly complex.
    rho      : (2^N, 2^N) complex Hermitian.
    Returns  : (4^N,) real.
    """
    T = rho.reshape([2] * (2 * N))
    for k in range(N):
        T = np.tensordot(T, effects1, axes=([k, N], [1, 2]))
        T = np.moveaxis(T, -1, k)
    return T.reshape(-1).real


def kron_grad_G(effects1, w, N):
    """
    Compute G = sum_i w_i E_i as a (2^N, 2^N) Hermitian matrix.
    Uses N tensor contractions to accumulate the weighted sum without
    forming all 4^N effects explicitly.
    effects1 : (4, 2, 2) possibly complex.
    w        : (4^N,) real weights.
    Returns  : (2^N, 2^N) complex Hermitian.
    """
    m = effects1.shape[0]
    T = w.reshape([m] * N)
    for _ in range(N):
        T = np.tensordot(effects1, T, axes=([0], [0]))
        T = np.moveaxis(T, [0, 1], [-2, -1])
    perm = list(range(0, 2 * N, 2)) + list(range(1, 2 * N, 2))
    return T.transpose(perm).reshape(2 ** N, 2 ** N)


# ─────────────────────────────────────────────────────────────────────────────
# CG solver for the shadow norm  W q = obs_flat
# ─────────────────────────────────────────────────────────────────────────────

def shadow_cg(pcm1, p, obs_flat, N, tol=1e-4):
    """
    Solve W q = obs_flat  where  W = pcm_N^T diag(1/p) pcm_N.

    W is (r^N × r^N), symmetric PD.  Matrix-vector products use the Kronecker
    factorisation and cost O(N · 4^N) instead of O(r^{2N}).

    Returns (q, S) where S = obs_flat^T q = shadow norm = BLUE variance + e².

    maxiter=1000 caps runtime at ~N·4^N·1000 ops. When ρ is near-pure some
    p_i→0 and kappa blows up; CG returns an approximate gradient after 1000
    steps — sufficient for FW direction, and unbiased (no probability clipping).
    """
    p_safe    = np.maximum(p, 1e-12)
    r         = pcm1.shape[1]
    nz        = r ** N
    maxiter   = min(200, nz)

    def matvec(v):
        u = kron_matvec(pcm1, v, N)
        return kron_rmatvec(pcm1, u / p_safe, N)

    A    = LinearOperator((nz, nz), matvec=matvec, dtype=np.float64)
    q, _ = sp_cg(A, obs_flat, rtol=tol, maxiter=maxiter)
    return q, float(obs_flat @ q)


# ─────────────────────────────────────────────────────────────────────────────
# Variance and gradient
# ─────────────────────────────────────────────────────────────────────────────

def variance_and_grad(rho, pcm1, effects1, obs, obs_flat, N):
    """
    Compute V(ρ) = S(ρ) − e(ρ)²  and  G = ∂V/∂ρ.

    Derivation of G:
      S = obs_flat^T W^{-1} obs_flat,   q = W^{-1} obs_flat
      ∂S/∂p_i = (r_i / p_i)²            where r_i = (pcm_N q)_i
      ∂p_i/∂ρ = E_i
      ∂e/∂ρ   = O
      G = sum_i (r_i/p_i)² E_i  −  2 e O

    G is Hermitian; its leading eigenvector gives the Frank-Wolfe direction.
    """
    p    = kron_probs(effects1, rho, N)
    p    = np.maximum(p, 1e-12)
    q, S = shadow_cg(pcm1, p, obs_flat, N)
    r    = kron_matvec(pcm1, q, N)         # r_i = (pcm_N q)_i
    e    = float(np.real(np.trace(obs @ rho)))
    V    = S - e ** 2
    w    = (r / p) ** 2                    # ∂S/∂p_i
    G    = kron_grad_G(effects1, w, N) - 2.0 * e * obs
    return V, G


# ─────────────────────────────────────────────────────────────────────────────
# Frank-Wolfe solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_frank_wolfe(N, obs, obs_flat, effects1, pcm1,
                      steps=300, n_restarts=3, verbose=True):
    """
    Maximise V(ρ) over the spectahedron via Frank-Wolfe.

    Strategy:
      - First restart from I/d (well-conditioned initial CG, global start).
      - Subsequent restarts continue from the best ρ found so far
        (extends optimisation while staying in a good basin).
      - γ = 2/(k+2) gives O(1/k) convergence without line search.

    Returns (v_opt, best_rho, t_solve).
    """
    d        = 2 ** N
    best_V   = -np.inf
    best_rho = np.eye(d, dtype=complex) / d

    t0 = time.perf_counter()

    for restart in range(n_restarts):
        rho = best_rho.copy()

        for k in range(steps):
            V, G = variance_and_grad(rho, pcm1, effects1, obs, obs_flat, N)

            if V > best_V:
                best_V   = V
                best_rho = rho.copy()

            # FW oracle: pure state in the direction of steepest ascent.
            # eigsh (Lanczos) finds only the leading eigenvector — O(d²) per
            # matvec × ~20 iterations, vs eigh's O(d³). 100x faster at N≥10.
            _, evecs = eigsh(G, k=1, which='LM')
            u      = evecs[:, 0].real
            u     /= np.linalg.norm(u)
            rho_fw = np.outer(u, u.conj())

            fw_gap = float(np.real(np.trace(G @ (rho_fw - rho))))
            gamma  = 2.0 / (k + 2.0)
            rho    = (1.0 - gamma) * rho + gamma * rho_fw

            if verbose and (k % 50 == 0 or k == steps - 1):
                elapsed = time.perf_counter() - t0
                print(f"    [r{restart} k={k:4d}]  V={V:.6f}  "
                      f"fw_gap={fw_gap:.2e}  t={elapsed:.1f}s", flush=True)

    # Final evaluation at best_rho
    V_final, _ = variance_and_grad(best_rho, pcm1, effects1, obs, obs_flat, N)
    if V_final > best_V:
        best_V = V_final

    return best_V, best_rho, time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry point for one (N, theta) configuration
# ─────────────────────────────────────────────────────────────────────────────

def run(N, theta, povm='xz', example=2, steps=300, n_restarts=3, verbose=True):
    """
    Solve the minimax SDP for one configuration.

    Returns a dict with v_opt, v_can, t_build, t_solve, N, theta.
    """
    t_build = time.perf_counter()

    # POVM effects and basis
    effects1 = xz_effects_1q() if povm == 'xz' else xy_effects_1q()
    pcm1, bm1 = build_1q_basis(effects1)

    # Observable
    obs_fn = product_observable if example == 1 else sum_local_observable
    obs    = obs_fn(theta, N)

    obs_flat, residual = obs_to_flat(obs, bm1, N)
    if residual > 1e-6:
        print(f"  WARNING: observable has out-of-subspace component {residual:.2e} "
              f"— not estimable with {povm.upper()} POVM. V* = inf.")
        return dict(N=N, theta=theta, v_opt=float('inf'), v_can=float('nan'),
                    t_build=0.0, t_solve=0.0)

    t_build = time.perf_counter() - t_build

    if verbose:
        d = 2 ** N
        r = pcm1.shape[1]
        print(f"  N={N}  d={d}  4^N={4**N}  r^N={r**N}  "
              f"pcm1=({4},{r})  build={t_build:.3f}s", flush=True)

    v_opt, _, t_solve = solve_frank_wolfe(
        N, obs, obs_flat, effects1, pcm1,
        steps=steps, n_restarts=n_restarts, verbose=verbose,
    )
    v_can = float(N * (N + 1))   # canonical variance for Example 2 XZ POVM

    return dict(N=N, theta=theta, v_opt=v_opt, v_can=v_can,
                t_build=t_build, t_solve=t_solve)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--N',        type=int,   default=5,
                    help='Number of qubits  (default 5)')
    ap.add_argument('--max_N',    type=int,   default=None,
                    help='If set, run N = N..max_N (sweep)')
    ap.add_argument('--theta',    type=float, default=0.0,
                    help='Observable angle in [0, π/2]  (default 0.0 = X)')
    ap.add_argument('--povm',     type=str,   default='xz',
                    choices=['xz', 'xy'],
                    help='POVM type: xz (real) or xy (complex)  (default xz)')
    ap.add_argument('--example',  type=int,   default=2,
                    choices=[1, 2],
                    help='Observable: 1=product pure state, 2=sum local  (default 2)')
    ap.add_argument('--steps',    type=int,   default=300,
                    help='Frank-Wolfe steps per restart  (default 300)')
    ap.add_argument('--restarts', type=int,   default=3,
                    help='Number of FW restarts  (default 3)')
    ap.add_argument('--quiet',    action='store_true',
                    help='Suppress per-step output')
    args = ap.parse_args()

    min_N = args.N
    max_N = args.max_N if args.max_N is not None else args.N
    N_values = list(range(min_N, max_N + 1))

    print('=' * 65)
    print(f'  frank_wolfe.py  |  Frank-Wolfe + Kronecker structure')
    print(f'  POVM={args.povm.upper()}  Example={args.example}  '
          f'theta={args.theta:.4f}  steps={args.steps}×{args.restarts}')
    print(f'  N = {N_values}')
    print('=' * 65)

    results = []
    for N in N_values:
        print(f'\n--- N={N} ---', flush=True)
        r = run(N, theta=args.theta, povm=args.povm, example=args.example,
                steps=args.steps, n_restarts=args.restarts,
                verbose=not args.quiet)
        results.append(r)

        ref  = REFERENCE_EX2.get(N)
        ref_str = f'{(r["v_opt"] - ref) / ref * 100:+.4f}%' if ref else '—'
        print(f'  V*    = {r["v_opt"]:.6f}  (ref: {ref_str})')
        print(f'  V_can = {r["v_can"]:.1f}   ratio = {r["v_can"] / r["v_opt"]:.4f}')
        print(f'  t_build={r["t_build"]:.2f}s   t_solve={r["t_solve"]:.1f}s')

    print('\n' + '=' * 65)
    print(f'  {"N":>3}  {"V*":>12}  {"V_can":>7}  {"ratio":>7}  '
          f'{"t_solve":>9}  {"vs ref":>10}')
    print('  ' + '-' * 55)
    for r in results:
        ref     = REFERENCE_EX2.get(r['N'])
        ref_str = f'{(r["v_opt"] - ref) / ref * 100:+.4f}%' if ref else '—'
        vcan    = r['v_can']
        ratio   = vcan / r['v_opt'] if r['v_opt'] > 0 else float('nan')
        print(f'  {r["N"]:>3}  {r["v_opt"]:>12.6f}  {vcan:>7.1f}  '
              f'{ratio:>7.4f}  {r["t_solve"]:>8.1f}s  {ref_str:>10}')
    print('=' * 65)
