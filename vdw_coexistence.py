"""
HW2, Question 2.2 -- van der Waals coexistence curve (reduced variables).

    P(v, T) = 8T / (3v - 1) - 3 / v^2

Conditions at each T < 1 (unknowns P*, v_l, v_g):
    (1)  P(v_l, T) = P(v_g, T) = P*
    (2)  Maxwell equal area, integrated analytically:
         int_{v_l}^{v_g} P dv = (8T/3) ln[(3v_g - 1)/(3v_l - 1)] + 3 (1/v_g - 1/v_l)
                              = P* (v_g - v_l)

Numerical method (robust, bracketed -- no convergence failures near T_c):
  * The isotherm has a local min (liquid spinodal, v_sp_l < 1) and a local max
    (gas spinodal, v_sp_g > 1). P* must lie between those two pressures.
  * For a trial pressure P in that window, find the liquid root v_l in
    (1/3, v_sp_l) and the gas root v_g in (v_sp_g, inf) of P(v,T) = P.
  * The equal-area mismatch g(P) = int_{v_l}^{v_g} [P(v) - P] dv is positive at
    the bottom of the window and negative at the top, so brentq finds P*.
  Every root is bracketed, so brentq always converges.
  The part-3 asymptotics (P* ~ 1 + 4t, v ~ 1 -+ 2 sqrt(-t)) are used only as a check.
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt


def P_vdw(v, T):
    return 8 * T / (3 * v - 1) - 3 / v**2


def dPdv(v, T):
    return -24 * T / (3 * v - 1)**2 + 6 / v**3


def area_mismatch(vl, vg, P, T):
    """int_{vl}^{vg} [P(v,T) - P] dv, written to avoid cancellation near T_c."""
    dv = vg - vl
    return (8 * T / 3) * np.log1p(3 * dv / (3 * vl - 1)) - 3 * dv / (vl * vg) - P * dv


def spinodals(T):
    """Volumes of the local min (liquid side) and local max (gas side) of P(v)."""
    v_sp_l = brentq(dPdv, 1 / 3 + 1e-12, 1.0, args=(T,), xtol=1e-15, rtol=1e-15)
    v_sp_g = brentq(dPdv, 1.0, 1e6, args=(T,), xtol=1e-15, rtol=1e-15)
    return v_sp_l, v_sp_g


def coexisting_volumes(P, T, v_sp_l, v_sp_g):
    """Outer roots of P(v,T) = P: liquid root below v_sp_l, gas root above v_sp_g."""
    f = lambda v: P_vdw(v, T) - P
    vl = brentq(f, 1 / 3 + 1e-14, v_sp_l, xtol=1e-15, rtol=1e-15)
    v_hi = max(10 * v_sp_g, 10 * 8 * T / (3 * P))   # P(v) ~ 8T/3v at large v
    vg = brentq(f, v_sp_g, v_hi, xtol=1e-15, rtol=1e-15)
    return vl, vg


def solve_one(T):
    v_sp_l, v_sp_g = spinodals(T)
    P_lo, P_hi = P_vdw(v_sp_l, T), P_vdw(v_sp_g, T)
    P_lo = max(P_lo, 1e-10)          # below T ~ 0.84 the local min is at P < 0
    eps = 1e-12 * (P_hi - P_lo)      # stay strictly inside the window

    def g(P):
        vl, vg = coexisting_volumes(P, T, v_sp_l, v_sp_g)
        return area_mismatch(vl, vg, P, T)

    P_star = brentq(g, P_lo + eps, P_hi - eps, xtol=1e-16, rtol=1e-15)
    vl, vg = coexisting_volumes(P_star, T, v_sp_l, v_sp_g)
    return P_star, vl, vg


def solve_coexistence(T_grid):
    out = np.array([solve_one(T) for T in T_grid])
    return out[:, 0], out[:, 1], out[:, 2]


if __name__ == "__main__":
    # Dense near T_c (needed for the log-log fit in part 4), down to T = 0.5
    T_grid = np.sort(1 - np.logspace(-6, np.log10(0.5), 400))[::-1]

    P_star, v_l, v_g = solve_coexistence(T_grid)
    dv = v_g - v_l

    # Sanity checks
    for T_check in (0.95, 0.9, 0.8, 0.5):
        i = np.argmin(abs(T_grid - T_check))
        print(f"T = {T_grid[i]:.3f}:  P* = {P_star[i]:.5f},  "
              f"v_l = {v_l[i]:.5f},  v_g = {v_g[i]:.5f}")
    t = T_grid[0] - 1
    print(f"closest to T_c (1-T = {-t:.0e}): dv / [4 sqrt(1-T)] = {dv[0] / (4*np.sqrt(-t)):.5f}"
          "  (should be ~1)")

    # Save the data for later parts (3b, 4, 5)
    np.savetxt("vdw_coexistence.csv",
               np.column_stack([T_grid, P_star, v_l, v_g, dv]),
               delimiter=",", header="T,P_star,v_l,v_g,v_g_minus_v_l", comments="")

    # ---------------- asymptotic forms near T_c (part 3) ----------------
    #   t = T - 1,  pi = 4t  ->  P* ~ 1 + 4t
    #   psi_{g,l} = +- 2 sqrt(-t)  ->  v_{g,l} ~ 1 +- 2 sqrt(1-T),  v_g - v_l ~ 4 sqrt(1-T)
    s = np.sqrt(1 - T_grid)
    P_asym = 1 + 4 * (T_grid - 1)
    vl_asym = 1 - 2 * s
    vg_asym = 1 + 2 * s
    dv_asym = 4 * s

    # The asymptotics are only meaningful near T_c: draw them for T > 0.8
    # (below T ~ 0.89 they even give v_l < 1/3, which is unphysical).
    near = T_grid > 0.8
    near_l = near & (vl_asym > 1 / 3)

    print("\nNumerical vs asymptotic (relative deviation):")
    for T_check in (0.999, 0.99, 0.95, 0.9):
        i = np.argmin(abs(T_grid - T_check))
        print(f"  T = {T_grid[i]:.3f}:  P* {abs(P_star[i]/P_asym[i]-1):.1e},  "
              f"v_l {abs(v_l[i]/vl_asym[i]-1):.1e},  v_g {abs(v_g[i]/vg_asym[i]-1):.1e},  "
              f"dv {abs(dv[i]/dv_asym[i]-1):.1e}")

    # ---------------- plots ----------------
    asym_style = dict(ls="--", lw=1.5, color="k")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(T_grid, P_star, lw=2, label="numerical")
    ax.plot(T_grid[near], P_asym[near], **asym_style, label=r"asymptotic $1 + 4t$")
    ax.plot(1, 1, "ko", label="critical point")
    ax.set_xlabel(r"$\bar T$"); ax.set_ylabel(r"$\bar P^*$")
    ax.set_title("q2.2: Coexistence curve")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(v_l, T_grid, lw=2, label=r"liquid $\bar\nu_l$")
    ax.plot(v_g, T_grid, lw=2, label=r"gas $\bar\nu_g$")
    ax.plot(vl_asym[near_l], T_grid[near_l], **asym_style,
            label=r"asymptotic $1 \pm 2\sqrt{-t}$")
    ax.plot(vg_asym[near], T_grid[near], **asym_style)
    ax.plot(1, 1, "ko")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\bar\nu$"); ax.set_ylabel(r"$\bar T$")
    ax.set_title("q2.2: Coexisting volumes (binodal)")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.plot(T_grid, dv, lw=2, label="numerical")
    ax.plot(T_grid[near], dv_asym[near], **asym_style, label=r"asymptotic $4\sqrt{-t}$")
    ax.set_xlabel(r"$\bar T$"); ax.set_ylabel(r"$\bar\nu_g - \bar\nu_l$")
    ax.set_yscale("log")
    ax.set_title("q2.2: Volume jump across coexistence")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("vdw_coexistence.png", dpi=150)
    plt.show()