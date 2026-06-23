"""
mosek_sdp.py — Exact minimax variance SDP solved with MOSEK
============================================================

Solves to machine precision:   V* = max_ρ { b^T W(ρ)^{-1} b  −  Tr(O ρ)² }

This is the exact SDP from Caprotti, Morris & Dakic (2026), formulated via the
Legendre variational identity and Schur complements, solved by MOSEK via CVXPY.

REQUIRES: a MOSEK licence in the standard location (~/.mosek/ or $MOSEKLM_LICENSE_FILE).
Academic licences are free at https://www.mosek.com/products/academic-licenses/

SDP FORMULATION
---------------
The minimax variance admits the semidefinite programme:

    max_{ρ, y, {t_i}, s}   2 b^T y  −  Σ_i t_i  −  s

    s.t.  ⎡ t_i    (Py)_i ⎤ ⪰ 0     for i = 1, …, 4^N   [one 2×2 block per effect]
          ⎣(Py)_i   p_i(ρ)⎦

          ⎡ s      e(ρ) ⎤ ⪰ 0                             [one 2×2 block for e²]
          ⎣e(ρ)     1   ⎦

          ρ ⪰ 0,   Tr(ρ) = 1

where p_i(ρ) = Tr(E_i ρ),  e(ρ) = Tr(O ρ),  (Py)_i = pcm[i,:] @ y.

At optimality t_i = (Py)_i² / p_i and s = e² so the objective equals V* exactly.
The SDP has 4^N + 1 linear matrix inequality constraints (each 2×2) plus the
density matrix constraint (d × d = 2^N × 2^N).

EFFICIENCY vs minimax_sdp.py
------------------------------
This script builds the POVM coefficient matrix pcm (4^N × 3^N) via Kronecker
products of the 1-qubit pcm1 (4 × 3) instead of SVD of the full 4^N × 4^N
vectorised effects matrix.  This avoids:
  - Storing (4^N × 4^N) canonicalisation matrix
  - Computing SVD on a 4^N-dimensional space
Both improvements are necessary to reach N=6 in reasonable time.

ACCURACY
--------
MOSEK is an interior-point solver; it finds V* to eps ~ 1e-8 by default.
For N=1..5 this is effectively machine precision.

RUNTIME ESTIMATES (MOSEK, MacBook)
------------------------------------
N=1  : < 0.1 s
N=2  : < 0.1 s
N=3  : ~ 0.3 s
N=4  : ~ 2 s
N=5  : ~ 30 s
N=6  : ~ 5 min   (4096 Schur blocks; 64×64 density matrix)
N≥7  : effects array > 2 GB — use frank_wolfe.py instead

COMPARISON WITH frank_wolfe.py
-------------------------------
  mosek_sdp.py   : exact (1e-8 precision),  N ≤ 5 fast,  N=6 ~5 min
  frank_wolfe.py : ~0.05% at N=7 in 23s,   ~0% at N≤5

Usage:
    python mosek_sdp.py --N 4 --theta 0.0 --example 2
    python mosek_sdp.py --N 5 --theta 1.5708 --example 1 --povm xz
    python mosek_sdp.py --N 3 --povm xy --theta 0.0 --example 2
"""

import argparse
import time

import numpy as np
import cvxpy as cp


# ─────────────────────────────────────────────────────────────────────────────
# Reference values (Caprotti et al. 2026, Table 1; N=6,7 from SCS)
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
# 1-qubit POVM effects
# ─────────────────────────────────────────────────────────────────────────────

def xz_effects_1q():
    return np.array([
        (_I + _X) / 4,
        (_I - _X) / 4,
        (_I + _Z) / 4,
        (_I - _Z) / 4,
    ], dtype=complex)


def xy_effects_1q():
    return np.array([
        (_I + _X) / 4,
        (_I - _X) / 4,
        (_I + _Y) / 4,
        (_I - _Y) / 4,
    ], dtype=complex)


# ─────────────────────────────────────────────────────────────────────────────
# Build 1-qubit basis and Kronecker-product pcm
# ─────────────────────────────────────────────────────────────────────────────

def build_1q_basis(effects_1q):
    """
    SVD of the Pauli-decomposition matrix → (pcm1, bm1), both real.
    Works for complex effects (XY POVM) because Re(Tr(P_k E_i)) is always real.
    """
    n = len(effects_1q)
    can_mat = np.array(
        [[np.real(np.trace(P @ effects_1q[i])) / 2 for P in PAULIS]
         for i in range(n)],
        dtype=float,
    )
    U, s, Vt = np.linalg.svd(can_mat, full_matrices=False)
    r   = int(np.sum(s > 1e-8 * s[0]))
    pcm1 = (U[:, :r] * s[:r]).astype(float)  # (4, r)
    bm1  = Vt[:r].T.astype(float)            # (4, r)
    return pcm1, bm1


def build_kron_pcm(pcm1, N):
    """
    Build pcm_N = pcm1 ⊗ ... ⊗ pcm1  (N copies) as a (4^N × r^N) matrix.
    Uses iterative Kronecker products; avoids building the full 4^N × d² matrix
    and running SVD on it (which is prohibitive for N ≥ 6).
    """
    pcm = pcm1
    for _ in range(N - 1):
        pcm = np.kron(pcm, pcm1)
    return pcm


def build_nqubit_effects(effects_1q, N):
    """
    Build all 4^N N-qubit effects as a (4^N, 2^N, 2^N) array.
    Uses Kronecker products of the 1-qubit effects.
    Memory: 4^N × 4^N × 8 bytes  (128 MB at N=6, 2 GB at N=7).
    """
    effects = effects_1q.copy()
    for _ in range(N - 1):
        effects = np.array([np.kron(ea, eb)
                            for ea in effects for eb in effects_1q])
    return effects


# ─────────────────────────────────────────────────────────────────────────────
# Observable projection
# ─────────────────────────────────────────────────────────────────────────────

def obs_to_flat(obs, bm1, N):
    """
    Project N-qubit Hermitian obs onto the r^N-dimensional observable subspace.
    Returns (r^N,) real array via N tensor contractions.
    """
    basis_mats = np.einsum('kj,kab->jab', bm1, np.array(PAULIS))  # (r, 2, 2)
    T = obs.reshape([2] * (2 * N))
    for k in range(N):
        T = np.tensordot(T, basis_mats, axes=([k, N], [1, 2]))
        T = np.moveaxis(T, -1, k)
    return T.reshape(-1).real / (2 ** N)


# ─────────────────────────────────────────────────────────────────────────────
# Observables
# ─────────────────────────────────────────────────────────────────────────────

def product_observable(theta, N):
    """Example 1: O^(N) = |ψ(θ)⟩⟨ψ(θ)|^⊗N."""
    psi = np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=complex)
    rho = np.outer(psi, psi.conj())
    obs = rho.copy()
    for _ in range(N - 1):
        obs = np.kron(obs, rho)
    return obs


def sum_local_observable(theta, N):
    """Example 2: O^(N) = sum_k σ_k(θ),  σ(θ) = cos(θ/2) X + sin(θ/2) Z."""
    d     = 2 ** N
    sigma = np.cos(theta / 2) * _X + np.sin(theta / 2) * _Z
    obs   = np.zeros((d, d), dtype=complex)
    for i in range(N):
        obs += np.kron(np.kron(np.eye(2 ** i, dtype=complex), sigma),
                       np.eye(2 ** (N - 1 - i), dtype=complex))
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# Minimax SDP
# ─────────────────────────────────────────────────────────────────────────────

def solve_minimax_sdp(effects, pcm, obs, obs_flat, solver='MOSEK', verbose=False):
    """
    Exact minimax SDP:  V* = max_ρ { b^T W(ρ)^{-1} b  −  Tr(O ρ)² }

    Formulation: Legendre identity + Schur complements.
    4^N semidefinite constraints (each 2×2) + one d×d PSD constraint.

    Parameters
    ----------
    effects  : (4^N, d, d) complex array    N-qubit POVM effects
    pcm      : (4^N, r^N) float array       POVM coefficient matrix
    obs      : (d, d) complex array         observable O
    obs_flat : (r^N,) float array           O in subspace coordinates
    solver   : 'MOSEK' or 'CLARABEL' (fallback without licence)
    verbose  : bool  — pass to CVXPY

    Returns
    -------
    v_opt  : float    V*
    rho    : (d, d)   optimal density matrix ρ*
    status : str      solver status string
    """
    n, d, _ = effects.shape
    nz = pcm.shape[1]

    # ── Decision variables ──────────────────────────────────────────────────
    rho_var = cp.Variable((d, d), hermitian=True)   # density matrix ρ  [d×d]
    y_var   = cp.Variable(nz)                       # dual variable y   [r^N]
    t_vars  = cp.Variable(n)                        # auxiliary t_i     [4^N]
    s_var   = cp.Variable()                         # auxiliary s ≥ e²

    # ── Affine expressions ───────────────────────────────────────────────────
    # p_i(ρ) = Tr(E_i ρ): vectorise effects to rows, dot with vec(ρ)
    # For complex effects: Tr(E_i ρ) = Re(Tr(E_i ρ)) since p_i is real.
    p_exprs  = [cp.real(cp.trace(effects[i] @ rho_var)) for i in range(n)]
    py_exprs = pcm @ y_var          # (4^N,) affine expression: (Py)_i = pcm[i,:] y
    e_expr   = cp.real(cp.trace(obs @ rho_var))                # e(ρ) = Tr(O ρ)

    # ── Objective ────────────────────────────────────────────────────────────
    obj = cp.Maximize(2.0 * (obs_flat @ y_var) - cp.sum(t_vars) - s_var)

    # ── Constraints ──────────────────────────────────────────────────────────
    constraints = [
        rho_var >> 0,
        cp.real(cp.trace(rho_var)) == 1,
    ]

    # 2×2 Schur block per effect: ⎡t_i  (Py)_i⎤ ⪰ 0  ⟺  t_i * p_i ≥ (Py)_i²
    #                              ⎣(Py)_i  p_i⎦
    for i in range(n):
        blk = cp.bmat([
            [cp.reshape(t_vars[i],       (1, 1)),
             cp.reshape(py_exprs[i],     (1, 1))],
            [cp.reshape(py_exprs[i],     (1, 1)),
             cp.reshape(p_exprs[i],      (1, 1))],
        ])
        constraints.append(blk >> 0)

    # 2×2 Schur block for squared expectation: ⎡s  e⎤ ⪰ 0  ⟺  s ≥ e²
    #                                           ⎣e  1⎦
    blk_s = cp.bmat([
        [cp.reshape(s_var,   (1, 1)), cp.reshape(e_expr, (1, 1))],
        [cp.reshape(e_expr,  (1, 1)), np.ones((1, 1))],
    ])
    constraints.append(blk_s >> 0)

    # ── Solve ────────────────────────────────────────────────────────────────
    prob = cp.Problem(obj, constraints)
    solver_id = getattr(cp, solver, cp.MOSEK)
    prob.solve(solver=solver_id, verbose=verbose)

    v_opt  = float(prob.value) if prob.value is not None else float('nan')
    rho    = rho_var.value
    return v_opt, rho, prob.status


# ─────────────────────────────────────────────────────────────────────────────
# Canonical estimator SDP  (V_can)
# ─────────────────────────────────────────────────────────────────────────────

def solve_canonical_sdp(effects, obs, pcm, obs_flat, solver='MOSEK', verbose=False):
    """
    Canonical worst-case variance SDP.

    c_can = pcm (pcm^T pcm)^{-1} obs_flat  is the minimum-norm unbiased estimator.
    V_can* = max_ρ { Tr(A_eff ρ) − e(ρ)² }  where A_eff = sum_i c_i² E_i.

    Since Tr(A_eff ρ) is linear in ρ, this is a simpler SDP (just one 2×2 block
    for the e² term, no t_i variables).
    """
    n, d, _ = effects.shape
    nz = pcm.shape[1]

    # Canonical coefficients
    coeff = np.linalg.solve(pcm.T @ pcm, obs_flat)
    c_can = pcm @ coeff                             # (4^N,) real

    # A_eff = sum_i c_i² E_i — Hermitian, real for XZ POVM, complex for XY POVM
    A_eff = sum(float(c_can[i]) ** 2 * effects[i] for i in range(n))

    rho_var = cp.Variable((d, d), hermitian=True)
    s_var   = cp.Variable()
    e_expr  = cp.real(cp.trace(obs @ rho_var))

    blk_s = cp.bmat([
        [cp.reshape(s_var,  (1, 1)), cp.reshape(e_expr, (1, 1))],
        [cp.reshape(e_expr, (1, 1)), np.ones((1, 1))],
    ])
    prob = cp.Problem(
        cp.Maximize(cp.real(cp.trace(A_eff @ rho_var)) - s_var),
        [rho_var >> 0, cp.real(cp.trace(rho_var)) == 1, blk_s >> 0],
    )
    solver_id = getattr(cp, solver, cp.MOSEK)
    prob.solve(solver=solver_id, verbose=verbose)

    v_can = float(prob.value) if prob.value is not None else float('nan')
    return v_can, rho_var.value, prob.status


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(N, theta, povm='xz', example=2, solver='MOSEK', verbose=False):
    """
    Build the SDP and solve for one (N, theta) configuration.

    Returns a dict with v_opt, v_can, t_build, t_solve, status.
    """
    t0 = time.perf_counter()

    # 1-qubit POVM and basis
    effects_1q = xz_effects_1q() if povm == 'xz' else xy_effects_1q()
    pcm1, bm1  = build_1q_basis(effects_1q)

    # N-qubit POVM coefficient matrix via Kronecker (avoids full SVD)
    pcm    = build_kron_pcm(pcm1, N)            # (4^N, 3^N)
    effects = build_nqubit_effects(effects_1q, N)  # (4^N, d, d)

    # Observable
    obs_fn   = product_observable if example == 1 else sum_local_observable
    obs      = obs_fn(theta, N)
    obs_flat = obs_to_flat(obs, bm1, N)         # (3^N,)

    t_build = time.perf_counter() - t0

    # Memory warning for large N
    d  = 2 ** N
    n  = 4 ** N
    nz = 3 ** N
    mem_gb = n * d * d * 16 / 1e9              # complex128 effects array
    if mem_gb > 1.0:
        print(f"  WARNING: effects array is {mem_gb:.1f} GB. Consider frank_wolfe.py.")

    print(f"  SDP: n={n}  d={d}  nz={nz}  "
          f"Schur blocks={n}  t_build={t_build:.2f}s", flush=True)

    # Minimax SDP
    t_solve = time.perf_counter()
    v_opt, rho_opt, status_opt = solve_minimax_sdp(
        effects, pcm, obs, obs_flat, solver=solver, verbose=verbose)
    t_opt = time.perf_counter() - t_solve

    # Canonical SDP
    t_solve = time.perf_counter()
    v_can, _, status_can = solve_canonical_sdp(
        effects, obs, pcm, obs_flat, solver=solver, verbose=verbose)
    t_can = time.perf_counter() - t_solve

    return dict(
        N=N, theta=theta, povm=povm, example=example,
        v_opt=v_opt, v_can=v_can,
        status_opt=status_opt, status_can=status_can,
        t_build=t_build, t_opt=t_opt, t_can=t_can,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--N',        type=int,   default=4,
                    help='Number of qubits  (default 4; N≥7 requires frank_wolfe.py)')
    ap.add_argument('--max_N',    type=int,   default=None,
                    help='If set, sweep N = N..max_N')
    ap.add_argument('--theta',    type=float, default=0.0,
                    help='Observable angle in [0, π/2]  (default 0.0)')
    ap.add_argument('--povm',     type=str,   default='xz',
                    choices=['xz', 'xy'],
                    help='POVM type  (default xz)')
    ap.add_argument('--example',  type=int,   default=2,
                    choices=[1, 2],
                    help='Observable: 1=product, 2=sum local  (default 2)')
    ap.add_argument('--solver',   type=str,   default='MOSEK',
                    choices=['MOSEK', 'CLARABEL', 'SCS'],
                    help='SDP solver  (default MOSEK; requires licence)')
    ap.add_argument('--verbose',  action='store_true',
                    help='Pass verbose=True to solver')
    args = ap.parse_args()

    min_N = args.N
    max_N = args.max_N if args.max_N is not None else args.N

    print('=' * 65)
    print(f'  mosek_sdp.py  |  Exact SDP  |  solver={args.solver}')
    print(f'  POVM={args.povm.upper()}  Example={args.example}  theta={args.theta:.4f}')
    print(f'  N = {list(range(min_N, max_N + 1))}')
    print('=' * 65)

    results = []
    for N in range(min_N, max_N + 1):
        print(f'\n--- N={N} ---', flush=True)
        try:
            r = run(N, theta=args.theta, povm=args.povm, example=args.example,
                    solver=args.solver, verbose=args.verbose)
        except Exception as e:
            print(f'  ERROR: {e}')
            break

        ref     = REFERENCE_EX2.get(N)
        ref_str = f'{(r["v_opt"] - ref) / ref * 100:+.5f}%' if ref else '—'
        ratio   = r['v_can'] / r['v_opt'] if r['v_opt'] > 0 else float('nan')
        print(f'  V*    = {r["v_opt"]:.6f}  [{r["status_opt"]}]  (ref: {ref_str})')
        print(f'  V_can = {r["v_can"]:.6f}  [{r["status_can"]}]  ratio={ratio:.4f}')
        print(f'  t_build={r["t_build"]:.2f}s  t_opt={r["t_opt"]:.2f}s  t_can={r["t_can"]:.2f}s')
        results.append(r)

    print('\n' + '=' * 65)
    print(f'  {"N":>3}  {"V*":>12}  {"V_can":>10}  {"t_opt":>8}  {"vs ref":>10}')
    print('  ' + '-' * 55)
    for r in results:
        ref     = REFERENCE_EX2.get(r['N'])
        ref_str = f'{(r["v_opt"] - ref) / ref * 100:+.5f}%' if ref else '—'
        print(f'  {r["N"]:>3}  {r["v_opt"]:>12.6f}  {r["v_can"]:>10.6f}  '
              f'{r["t_opt"]:>7.2f}s  {ref_str:>10}')
    print('=' * 65)
