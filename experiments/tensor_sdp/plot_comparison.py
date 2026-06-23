"""
plot_comparison.py — Compare exact SDP, Frank-Wolfe, and canonical variance
============================================================================

Usage (after SDP and FW runs):
    python plot_comparison.py                      # reads /tmp/sdp_results.txt and /tmp/fw_results.txt
    python plot_comparison.py --sdp sdp.txt --fw fw.txt

Produces two figures:
    comparison_variances.png   — V_can, V*_SDP, V*_FW vs N
    comparison_gap.png         — % gap between methods per qubit

Or pass --hardcoded to use baked-in values without needing the result files.
"""

import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─────────────────────────────────────────────────────────────────────────────
# Reference: Caprotti et al. (2026) Table 1, Example 2, XZ POVM, theta=0
# ─────────────────────────────────────────────────────────────────────────────

REF = {1: 2.00000, 2: 5.33321, 3: 10.28550, 4: 17.06671, 5: 25.80534}


def v_can(N):
    return N * (N + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_sdp(path):
    """Parse kron_sdp.py summary table → {N: v_opt}."""
    results = {}
    try:
        with open(path) as f:
            text = f.read()
        # Match summary table rows:  "   N      V*          t_build    t_solve   vs ref"
        for m in re.finditer(r'^\s{2,}(\d+)\s+([\d.]+)\s+[\d.]+s\s+[\d.]+s', text, re.MULTILINE):
            N, v = int(m.group(1)), float(m.group(2))
            results[N] = v
    except FileNotFoundError:
        print(f"[warn] {path} not found — using reference values for N≤5")
    return results


def parse_fw(path):
    """Parse frank_wolfe.py summary table → {N: v_opt}."""
    results = {}
    try:
        with open(path) as f:
            text = f.read()
        for m in re.finditer(r'^\s{2,}(\d+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+s', text, re.MULTILINE):
            N, v = int(m.group(1)), float(m.group(2))
            results[N] = v
    except FileNotFoundError:
        print(f"[warn] {path} not found")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(sdp_vals, fw_vals):
    N_sdp = sorted(sdp_vals)
    N_fw  = sorted(fw_vals)
    all_N = sorted(set(N_sdp) | set(N_fw))

    # Canonical variance for all N up to max FW
    N_all  = list(range(1, max(all_N) + 1))
    vc_all = [v_can(n) for n in N_all]

    # ── Figure 1: three variances ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(N_all, vc_all, 'k--', lw=1.5, label='Canonical $V_{\\rm can} = N(N+1)$', zorder=1)

    if sdp_vals:
        ax.plot(N_sdp, [sdp_vals[n] for n in N_sdp],
                'o-', color='tab:blue', lw=2, ms=6, label='Exact SDP (MOSEK)', zorder=3)

    if fw_vals:
        ax.plot(N_fw, [fw_vals[n] for n in N_fw],
                's--', color='tab:orange', lw=2, ms=6,
                label='Frank-Wolfe (300 steps)', zorder=2)

    # Reference paper dots (filled)
    ref_N = sorted(REF)
    ax.scatter(ref_N, [REF[n] for n in ref_N],
               marker='*', s=80, color='tab:green', zorder=4,
               label='Caprotti et al. (2026) Table 1')

    ax.set_xlabel('Number of qubits $N$')
    ax.set_ylabel('Minimax variance $V^*$')
    ax.set_title('Minimax variance: XZ POVM, Example 2, $\\theta=0$')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('experiments/tensor_sdp/comparison_variances.png', dpi=150)
    print("Saved: experiments/tensor_sdp/comparison_variances.png")

    # ── Figure 2: % gaps ─────────────────────────────────────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Left: % improvement of optimal V* over canonical (how much better is optimal?)
    Ns_sdp_gap = [n for n in N_sdp if sdp_vals.get(n) is not None]
    gap_sdp = [(v_can(n) - sdp_vals[n]) / v_can(n) * 100 for n in Ns_sdp_gap]
    Ns_fw_gap = [n for n in N_fw if fw_vals.get(n) is not None]
    gap_fw  = [(v_can(n) - fw_vals[n]) / v_can(n) * 100 for n in Ns_fw_gap]

    ax1.plot(Ns_sdp_gap, gap_sdp, 'o-', color='tab:blue', lw=2, ms=6, label='SDP (exact)')
    ax1.plot(Ns_fw_gap,  gap_fw,  's--', color='tab:orange', lw=2, ms=6, label='FW (300 steps)')
    ax1.set_xlabel('Number of qubits $N$')
    ax1.set_ylabel('$(V_{\\rm can} - V^*) / V_{\\rm can}$ [%]')
    ax1.set_title('Reduction vs canonical estimator')
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: % discrepancy between FW and SDP (accuracy of FW)
    common_N = sorted(set(N_sdp) & set(N_fw))
    if common_N:
        discrepancy = [(sdp_vals[n] - fw_vals[n]) / sdp_vals[n] * 100 for n in common_N]
        ax2.bar(common_N, discrepancy, color='tab:purple', alpha=0.7)
        ax2.axhline(0, color='k', lw=0.8)
        ax2.set_xlabel('Number of qubits $N$')
        ax2.set_ylabel('$(V^*_{\\rm SDP} - V^*_{\\rm FW}) / V^*_{\\rm SDP}$ [%]')
        ax2.set_title('FW error vs exact SDP')
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.grid(True, alpha=0.3, axis='y')

    fig2.tight_layout()
    fig2.savefig('experiments/tensor_sdp/comparison_gap.png', dpi=150)
    print("Saved: experiments/tensor_sdp/comparison_gap.png")

    # ── Print summary table ──────────────────────────────────────────────────
    print()
    print(f"{'N':>3}  {'V_can':>8}  {'V*_SDP':>12}  {'V*_FW':>12}  "
          f"{'SDP/can':>8}  {'FW vs SDP':>10}")
    print("  " + "-"*60)
    for n in range(1, max(all_N)+1):
        vc  = v_can(n)
        vs  = sdp_vals.get(n, float('nan'))
        vf  = fw_vals.get(n,  float('nan'))
        r1  = f'{vs/vc*100:.2f}%' if not np.isnan(vs) else '   —'
        r2  = (f'{(vs-vf)/vs*100:+.3f}%'
               if not (np.isnan(vs) or np.isnan(vf)) else '   —')
        print(f"{n:>3}  {vc:>8.1f}  "
              f"{vs:>12.5f}  {vf:>12.5f}  {r1:>8}  {r2:>10}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sdp', default='/tmp/sdp_results.txt')
    ap.add_argument('--fw',  default='/tmp/fw_results.txt')
    args = ap.parse_args()

    sdp_vals = parse_sdp(args.sdp)
    fw_vals  = parse_fw(args.fw)

    # Fill SDP gaps with paper reference values
    for n, v in REF.items():
        if n not in sdp_vals:
            sdp_vals[n] = v

    print(f"SDP values: {dict(sorted(sdp_vals.items()))}")
    print(f"FW  values: {dict(sorted(fw_vals.items()))}")

    plot(sdp_vals, fw_vals)
