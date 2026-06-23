"""
kron_sdp.py — Exact minimax variance SDP via MOSEK Fusion + Kronecker basis
=============================================================================

Solves to interior-point precision:
    V* = max_rho { b^T W(rho)^{-1} b  -  Tr(O rho)^2 }

KEY TRICK — avoid the (4^N, d, d) effects array
------------------------------------------------
Introduce r^N auxiliary variables  q_a = Tr(B_a rho)  (a = 0..r^N-1)
where B_a = B_{a_1} x...x B_{a_N} are the observable-subspace basis matrices.

Then p_i(rho) = pcm_N[i,:] @ q  and  (Py)_i = pcm_N[i,:] @ y
are pure scalar linear expressions. MOSEK only needs the r^N basis matrices
(each d x d), never the 4^N POVM effects.

Data sizes (XZ POVM, r=3):
  N=5 : r^5=243   basis matrices (32x32)    ~6 MB   -> trivial
  N=6 : r^6=729   basis matrices (64x64)    ~24 MB  -> fast
  N=7 : r^7=2187  basis matrices (128x128)  ~286 MB -> minutes
  N=8 : r^8=6561  basis matrices (256x256)  ~3.4 GB -> tight but feasible

SDP formulation (MOSEK Fusion)
-------------------------------
Variables:  rho (d x d PSD),  q (r^N),  y (r^N),  t (4^N >= 0),  s >= 0,  e
Objective:  max  2 b^T y  -  sum(t)  -  s
Constraints:
  Tr(rho) = 1
  q_a = Tr(B_a rho)                          for a = 0..r^N-1
  e   = Tr(O rho)
  (t_i, p_i/2, py_i) in RotatedQCone^3      for i = 0..4^N-1
    where p_i = pcm_N[i,:] @ q,  py_i = pcm_N[i,:] @ y
  (s, 1/2, e) in RotatedQCone^3

Note on the factor 1/2: MOSEK's rotated cone is 2*u*v >= w^2.
Scaling the second argument by 1/2 gives 2*u*(v/2) = u*v >= w^2.

Note on rho: for XZ POVM and real observables, the optimal rho is real
symmetric. We use a real d x d PSD variable directly. XY POVM requires
a complex embedding (not yet implemented).

Usage:
    python kron_sdp.py --N 5
    python kron_sdp.py --N 4 --max_N 7
    python kron_sdp.py --N 7 --povm xz --example 2
"""

import argparse
import itertools
import time

import numpy as np
import mosek.fusion as mf


# ─────────────────────────────────────────────────────────────────────────────
# Reference values (Caprotti et al. 2026, Table 1)
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
PAULIS = [_I, _X, _Y, _Z]


# ─────────────────────────────────────────────────────────────────────────────
# Single-qubit POVM effects
# ─────────────────────────────────────────────────────────────────────────────

def xz_effects_1q():
    return np.array([(_I + _X) / 4, (_I - _X) / 4,
                     (_I + _Z) / 4, (_I - _Z) / 4], dtype=complex)


def xy_effects_1q():
    return np.array([(_I + _X) / 4, (_I - _X) / 4,
                     (_I + _Y) / 4, (_I - _Y) / 4], dtype=complex)


# ─────────────────────────────────────────────────────────────────────────────
# Basis construction
# ─────────────────────────────────────────────────────────────────────────────

def build_1q_basis(effects_1q):
    """SVD of Pauli-decomposition → pcm1 (4, r) and bm1 (4, r), both real."""
    n = len(effects_1q)
    can_mat = np.array(
        [[np.real(np.trace(P @ effects_1q[i])) / 2 for P in PAULIS]
         for i in range(n)],
        dtype=float,
    )
    U, s, Vt = np.linalg.svd(can_mat, full_matrices=False)
    r    = int(np.sum(s > 1e-8 * s[0]))
    pcm1 = (U[:, :r] * s[:r]).astype(float)   # (4, r)
    bm1  = Vt[:r].T.astype(float)             # (4, r)
    return pcm1, bm1


def build_kron_pcm(pcm1, N):
    """pcm_N = pcm1 ⊗ ... ⊗ pcm1 (N times), shape (4^N, r^N)."""
    pcm = pcm1.copy()
    for _ in range(N - 1):
        pcm = np.kron(pcm, pcm1)
    return pcm


def basis_matrix_1q(bm1, j):
    """j-th 1-qubit basis matrix: B_j = sum_k bm1[k,j] * P_k  (2x2, real for XZ)."""
    return sum(float(bm1[k, j]) * PAULIS[k] for k in range(4))


def basis_matrix_Nq(bm1, alpha_tuple):
    """
    N-qubit basis matrix B_alpha = B_{a_1} x ... x B_{a_N}.
    Built via sequential Kronecker — no large array ever stored.
    Returns (d, d) complex Hermitian (purely real for XZ POVM).
    """
    B = basis_matrix_1q(bm1, alpha_tuple[0])
    for a in alpha_tuple[1:]:
        B = np.kron(B, basis_matrix_1q(bm1, a))
    return B


# ─────────────────────────────────────────────────────────────────────────────
# Observables
# ─────────────────────────────────────────────────────────────────────────────

def product_observable(theta, N):
    """Example 1: O = |psi(theta)><psi(theta)|^{otimes N}."""
    psi = np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=complex)
    rho1 = np.outer(psi, psi.conj())
    obs  = rho1.copy()
    for _ in range(N - 1):
        obs = np.kron(obs, rho1)
    return obs


def sum_local_observable(theta, N):
    """Example 2: O = sum_k sigma_k(theta),  sigma = cos(t/2) X + sin(t/2) Z."""
    d     = 2 ** N
    sigma = np.cos(theta / 2) * _X + np.sin(theta / 2) * _Z
    obs   = np.zeros((d, d), dtype=complex)
    for i in range(N):
        obs += np.kron(np.kron(np.eye(2 ** i, dtype=complex), sigma),
                       np.eye(2 ** (N - i - 1), dtype=complex))
    return obs


def obs_to_flat(obs, bm1, N):
    """Project obs onto r^N observable subspace via N tensor contractions."""
    basis_mats = np.einsum('kj,kab->jab', bm1, np.array(PAULIS))
    T = obs.reshape([2] * (2 * N))
    for k in range(N):
        T = np.tensordot(T, basis_mats, axes=([k, N], [1, 2]))
        T = np.moveaxis(T, -1, k)
    return T.reshape(-1).real / (2 ** N)


# ─────────────────────────────────────────────────────────────────────────────
# Core solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_kron_sdp(N, obs, obs_flat, bm1, pcm1, verbose=False):
    """
    Build and solve the minimax SDP with MOSEK Fusion.

    Returns (v_opt, status, t_build, t_solve).
    """
    d  = 2 ** N
    r  = pcm1.shape[1]
    n  = 4 ** N      # number of effects
    nz = r ** N      # subspace / basis dimension

    # Build pcm_N: (4^N, r^N) — needed for p_i = pcm_N @ q, py_i = pcm_N @ y
    # All r^N multi-indices for basis enumeration
    alpha_list = list(itertools.product(range(r), repeat=N))

    t0 = time.perf_counter()

    with mf.Model("kron_sdp") as M:
        M.setSolverParam("log", 1 if verbose else 0)

        # ── Decision variables ────────────────────────────────────────────────
        rho = M.variable(mf.Domain.inPSDCone(d))              # d×d real PSD
        q   = M.variable(nz)                                   # basis coords of rho
        y   = M.variable(nz)                                   # Legendre dual
        t   = M.variable(n, mf.Domain.greaterThan(0.0))       # Schur slacks
        s   = M.variable(mf.Domain.greaterThan(0.0))          # slack for e^2
        e   = M.variable()                                     # Tr(O rho)

        # ── Objective: max  2 b^T y - sum(t) - s ─────────────────────────────
        M.objective(
            mf.ObjectiveSense.Maximize,
            mf.Expr.sub(
                mf.Expr.sub(
                    mf.Expr.dot(2.0 * obs_flat, y),
                    mf.Expr.sum(t),
                ),
                s,
            ),
        )

        # ── Tr(rho) = 1 ──────────────────────────────────────────────────────
        M.constraint("trace",
                     mf.Expr.sum(rho.diag()),
                     mf.Domain.equalsTo(1.0))

        # ── q_a = Tr(B_a rho)  for each basis multi-index a ──────────────────
        # B_a = B_{a_1} x...x B_{a_N}  built on the fly (real for XZ POVM).
        # Expr.dot(A, X) computes <A, X>_F = Tr(A^T X) = Tr(A X) for symmetric A, X.
        for a_idx, alpha in enumerate(alpha_list):
            B_a = np.real(basis_matrix_Nq(bm1, alpha))   # (d, d) real symmetric
            M.constraint(
                f"q{a_idx}",
                mf.Expr.sub(q.index(a_idx),
                            mf.Expr.dot(B_a, rho)),
                mf.Domain.equalsTo(0.0),
            )

        # ── e = Tr(O rho) ─────────────────────────────────────────────────────
        M.constraint(
            "e_def",
            mf.Expr.sub(e, mf.Expr.dot(np.real(obs), rho)),
            mf.Domain.equalsTo(0.0),
        )

        # ── Rotated SOC per effect: t_i * p_i >= py_i^2 ──────────────────────
        # MOSEK rotated cone: 2*u*v >= w^2.
        # Use (t_i, p_i/2, py_i) -> 2 * t_i * (p_i/2) = t_i * p_i >= py_i^2.
        #
        # To avoid materialising pcm_N (4^N x r^N, up to 3 GB at N=8),
        # compute each row of pcm_N on the fly via N sequential np.kron calls
        # and add the rotated cone constraints in batches.
        BATCH = 2048   # tune: larger = faster Python loop, more peak RAM
        effect_multi = list(itertools.product(range(4), repeat=N))

        for b_start in range(0, n, BATCH):
            b_end   = min(b_start + BATCH, n)
            b_size  = b_end - b_start
            # Build pcm rows for this batch: (b_size, r^N)
            pcm_batch = np.empty((b_size, nz), dtype=float)
            for local_i, i_multi in enumerate(effect_multi[b_start:b_end]):
                row = pcm1[i_multi[0], :].copy()
                for k in range(1, N):
                    row = np.kron(row, pcm1[i_multi[k], :])
                pcm_batch[local_i] = row

            pcm_M_b   = mf.Matrix.dense(pcm_batch)
            p_b       = mf.Expr.mul(pcm_M_b, q)               # (b_size,)
            py_b      = mf.Expr.mul(pcm_M_b, y)               # (b_size,)
            p_half_b  = mf.Expr.mul(0.5, p_b)

            t_b = mf.Expr.reshape(t.slice(b_start, b_end), b_size, 1)
            M.constraint(
                f"schur_effects_{b_start}",
                mf.Expr.hstack(
                    t_b,
                    mf.Expr.reshape(p_half_b,  b_size, 1),
                    mf.Expr.reshape(py_b,      b_size, 1),
                ),
                mf.Domain.inRotatedQCone(b_size, 3),
            )

        # ── Rotated SOC: s >= e^2 ─────────────────────────────────────────────
        # Use (s, 1/2, e) -> 2 * s * (1/2) = s >= e^2.
        M.constraint(
            "schur_e",
            mf.Expr.vstack(s, mf.Expr.constTerm(0.5), e),
            mf.Domain.inRotatedQCone(),
        )

        # ── Solve ─────────────────────────────────────────────────────────────
        t_build = time.perf_counter() - t0
        ts = time.perf_counter()
        M.solve()
        t_solve = time.perf_counter() - ts

        status = str(M.getProblemStatus())
        try:
            v_opt = float(M.primalObjValue())
        except Exception:
            v_opt = float('nan')

    return v_opt, status, t_build, t_solve


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(N, theta=0.0, povm='xz', example=2, verbose=False):
    effects_1q = xz_effects_1q() if povm == 'xz' else xy_effects_1q()
    pcm1, bm1  = build_1q_basis(effects_1q)
    r  = pcm1.shape[1]
    d  = 2 ** N
    nz = r ** N
    n  = 4 ** N

    obs_fn   = product_observable if example == 1 else sum_local_observable
    obs      = obs_fn(theta, N)
    obs_flat = obs_to_flat(obs, bm1, N)

    if povm == 'xy' and example == 2 and abs(theta) > 1e-6:
        print("  WARNING: XY POVM + sum_local at theta!=0 has Z component — not estimable.")

    mem_basis_gb = nz * d * d * 8 / 1e9
    mem_pcm_gb   = n * nz * 8 / 1e9
    print(f"  N={N}  d={d}  4^N={n}  r^N={nz}  "
          f"basis_data={mem_basis_gb:.2f}GB  pcm_data={mem_pcm_gb:.2f}GB",
          flush=True)
    if mem_basis_gb + mem_pcm_gb > 16.0:
        print("  WARNING: total data > 16 GB — may OOM.")

    v_opt, status, t_build, t_solve = solve_kron_sdp(
        N, obs, obs_flat, bm1, pcm1, verbose=verbose)

    return dict(N=N, theta=theta, v_opt=v_opt, status=status,
                t_build=t_build, t_solve=t_solve)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--N',       type=int,   default=5)
    ap.add_argument('--max_N',   type=int,   default=None)
    ap.add_argument('--theta',   type=float, default=0.0)
    ap.add_argument('--povm',    type=str,   default='xz', choices=['xz', 'xy'])
    ap.add_argument('--example', type=int,   default=2,    choices=[1, 2])
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    min_N = args.N
    max_N = args.max_N if args.max_N is not None else args.N

    print('=' * 65)
    print('  kron_sdp.py  |  Exact SDP + Kronecker basis  |  MOSEK Fusion')
    print(f'  POVM={args.povm.upper()}  Example={args.example}  theta={args.theta:.4f}')
    print(f'  N = {list(range(min_N, max_N + 1))}')
    print('=' * 65)

    results = []
    for N in range(min_N, max_N + 1):
        print(f'\n--- N={N} ---', flush=True)
        try:
            r = run(N, theta=args.theta, povm=args.povm,
                    example=args.example, verbose=args.verbose)
        except Exception as exc:
            print(f'  ERROR: {exc}')
            break

        ref     = REFERENCE_EX2.get(N)
        ref_str = f'{(r["v_opt"] - ref) / ref * 100:+.5f}%' if ref else '—'
        print(f'  V*     = {r["v_opt"]:.8f}  [{r["status"]}]  (ref: {ref_str})')
        print(f'  t_build={r["t_build"]:.2f}s   t_solve={r["t_solve"]:.2f}s')
        results.append(r)

    if len(results) > 1:
        print('\n' + '=' * 65)
        print(f'  {"N":>3}  {"V*":>14}  {"t_build":>9}  {"t_solve":>9}  {"vs ref":>10}')
        print('  ' + '-' * 55)
        for r in results:
            ref     = REFERENCE_EX2.get(r['N'])
            ref_str = f'{(r["v_opt"] - ref) / ref * 100:+.5f}%' if ref else '—'
            print(f'  {r["N"]:>3}  {r["v_opt"]:>14.8f}  '
                  f'{r["t_build"]:>8.2f}s  {r["t_solve"]:>8.2f}s  {ref_str:>10}')
        print('=' * 65)
