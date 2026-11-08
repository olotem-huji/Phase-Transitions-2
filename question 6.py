"""
HW2, Question 2.6 -- heat capacity C_V of the van der Waals fluid along the
critical isochore v = v_c, cooled through T_c.

Requires vdw_coexistence.py (Q2.2) in the same folder (change the import if
you saved it under another name, e.g. main.py).

Free energy per particle of the vdW fluid (units of kB*Tc, reduced variables,
b = v_c/3, a = (9/8) kB Tc v_c, dropping terms linear in T which do not
contribute to C_V):
    f(v, T) = f_ideal(T)  +  f_ex(v, T),
    f_ex    = -T ln(v - 1/3) - 9 / (8 v)
    (check: P = -df/dv = T/(v-1/3) - 9/(8v^2) in units of kB Tc / v_c,
     i.e.  Pbar = (8/3) * that = 8T/(3v-1) - 3/v^2  since Pc vc / kB Tc = 3/8.)
The ideal-gas part gives C_V = (3/2) kB per particle in both regimes.

T > T_c:  single phase at v = 1.  f_ex(1, T) = -T ln(2/3) - 9/8 is linear in T,
          so the vdW interaction adds nothing:  c_V = 3/2.

T < T_c:  liquid + gas coexist. Both have the same mu and P*, and f = mu - P v in
          each phase, so for the whole system (hint)
              F / N = mu(T) - P*(T) v_c.
          In units of kB Tc this is  phi(T) = f_ex(v_l) + (3/8) Pbar* (v_l - 1)
          (+ ideal part); the same value is obtained with v_g in place of v_l.
          Then
              S / N = -d phi/dT = (1 - x) s_l + x s_g,   s = ln(v - 1/3),
          with x = (1 - v_l)/(v_g - v_l) the gas fraction (lever rule), and
              c_V / kB = 3/2 + T dS/dT = 3/2 - T d^2 phi/dT^2.
          We compute dS/dT with a symmetric finite difference (one derivative is
          numerically far better behaved near T_c than two), and check it
          against the direct second derivative of phi = mu - P* v_c.
"""

import numpy as np
import matplotlib.pyplot as plt
from vdw_coexistence import solve_coexistence

C_IDEAL = 1.5  # (3/2) kB per particle


def f_ex(v, T):
    return -T * np.log(v - 1 / 3) - 9 / (8 * v)


def phi_two_phase(T):
    """F/(N kB Tc) - ideal part, at v = v_c, from F = N mu - P* V."""
    P, vl, vg = solve_coexistence(np.atleast_1d(T))
    mu = f_ex(vl, T) + (3 / 8) * P * vl          # mu = f + P v  (same at v_g)
    return mu - (3 / 8) * P * 1.0                 # minus P* v_c


def S_two_phase(T):
    """S_ex/(N kB) at v = v_c: lever-rule average of s = ln(v - 1/3)."""
    P, vl, vg = solve_coexistence(np.atleast_1d(T))
    x = (1 - vl) / (vg - vl)
    return (1 - x) * np.log(vl - 1 / 3) + x * np.log(vg - 1 / 3)


# ------------------------- below T_c ---------------------------------------
tau = np.logspace(-6, np.log10(0.5), 300)   # 1 - T
T_below = 1 - tau
h = 1e-3 * tau                               # keep T + h below T_c

dSdT = (S_two_phase(T_below + h) - S_two_phase(T_below - h)) / (2 * h)
cV_below = C_IDEAL + T_below * dSdT

# Check 1: S = -d phi/dT, with phi = mu - P* v_c from the hint
for T0 in (0.6, 0.9):
    hh = 1e-6
    S_from_phi = -(phi_two_phase(T0 + hh) - phi_two_phase(T0 - hh)) / (2 * hh)
    print(f"T = {T0}:  S from -dF/dT = {S_from_phi[0]:.8f},  lever rule = {S_two_phase(T0)[0]:.8f}")

# Check 2: c_V directly from the second derivative of F = N mu - P* V
print("\nc_V/kB from  -T d^2(mu - P* v_c)/dT^2  vs  T dS/dT:")
for T0 in (0.6, 0.8, 0.9, 0.99):
    hh = 1e-4 * (1 - T0)
    d2phi = (phi_two_phase(T0 + hh) - 2 * phi_two_phase(T0) + phi_two_phase(T0 - hh)) / hh**2
    hs = 1e-3 * (1 - T0)
    cV_S = C_IDEAL + T0 * (S_two_phase(T0 + hs) - S_two_phase(T0 - hs)) / (2 * hs)
    print(f"  T = {T0}:  {C_IDEAL - T0 * d2phi[0]:.5f}   vs   {cV_S[0]:.5f}")

# ------------------------- above T_c ---------------------------------------
T_above = np.linspace(1, 1.5, 50)
cV_above = np.full_like(T_above, C_IDEAL)   # f_ex(1, T) is linear in T

# ------------------------- results -----------------------------------------
print(f"\nc_V(T_c^-) / kB  = {cV_below[0]:.5f}   (at 1 - T = {tau[0]:.0e}; expected 3/2 + 9/2 = 6)")
print(f"c_V(T_c^+) / kB  = {C_IDEAL:.5f}")
print(f"jump  Delta c_V / kB = {cV_below[0] - C_IDEAL:.5f}   (expected 9/2)")

m = tau <= 1e-3
slope = np.polyfit(np.log(tau[m]), np.log(cV_below[m]), 1)[0]
print(f"log-log slope of c_V vs (Tc - T) for 1 - T <= 1e-3: {slope:.2e}  ->  alpha = {-slope:.1e} ~ 0")
lin = np.polyfit(tau[m], cV_below[m], 1)
print(f"near T_c:  c_V/kB ~ {lin[1]:.4f} + ({lin[0]:.3f}) (1 - T)")

# ------------------------- plots -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

ax = axes[0]
ax.plot(T_below, cV_below, lw=2, label=r"$T<T_c$: liquid + gas")
ax.plot(T_above, cV_above, lw=2, label=r"$T>T_c$: single phase")
ax.vlines(1, C_IDEAL, cV_below[0], colors="k", linestyles=":", lw=1.2)
ax.annotate(r"jump $\Delta c_V = \frac{9}{2}k_B$", xy=(1, 3.75), xytext=(1.08, 4.3),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlabel(r"$\bar T$"); ax.set_ylabel(r"$c_V \,/\, k_B$ (per particle)")
ax.set_title(r"q2.6: Heat capacity along the critical isochore $\bar\nu = 1$")
ax.set_ylim(0, 7)
ax.legend(loc="lower left"); ax.grid(alpha=0.3)

ax = axes[1]
ax.loglog(tau, cV_below, lw=2, label=r"$T<T_c$")
ax.loglog(tau, np.full_like(tau, C_IDEAL), lw=2, label=r"$T>T_c$ (at $|T-T_c|$)")
ax.set_xlabel(r"$|\bar T - 1|$"); ax.set_ylabel(r"$c_V \,/\, k_B$")
ax.set_title(r"q2.6: No power-law divergence: $\alpha = 0$")
ax.set_ylim(1, 10)
ax.legend(); ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("q2_6_heat_capacity.png", dpi=150)
plt.show()