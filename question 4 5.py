"""
HW2, Question 2.4 and 2.5 -- critical exponent beta and the latent heat
for the van der Waals fluid.

Requires vdw_coexistence.py (Q2.2) in the same folder -- it provides the
coexistence solver. If you saved that file under another name (e.g. vdw_coexistence.py),
change the import line below accordingly.

Q2.4:  fit  v_g - v_l ~ A (1 - T)^beta  on a log-log plot, and use the local
       slope  d ln(dv) / d ln(1 - T)  to see where the power law holds.

Q2.5:  Clausius-Clapeyron,  dP*/dT = L / (T dv)   =>   L = T dv dP*/dT.
       In reduced units (P = Pc Pbar, v = vc vbar, T = Tc Tbar):
           L / (kB Tc) = (Pc vc / kB Tc) * Tbar * dvbar * dPbar*/dTbar
                       = (3/8) * Tbar * dvbar * dPbar*/dTbar,
       using Zc = Pc vc / kB Tc = 3/8 for van der Waals.
       Near T_c: dP*/dT -> 4 and dv -> 4 sqrt(1-T), so
           L / (kB Tc) ~ (3/8) * 4 * 4 sqrt(1-T) = 6 (1 - T)^(1/2).
       Independent check: s = kB ln(v - b) + f(T) for vdW, so
           L / (kB Tc) = Tbar * ln[(3 v_g - 1) / (3 v_l - 1)].
"""

import numpy as np
import matplotlib.pyplot as plt
from vdw_coexistence import solve_coexistence

ZC = 3 / 8  # Pc vc / (kB Tc) for van der Waals

# Same grid as Q2.2: dense near T_c, down to T = 0.5
tau = np.logspace(-6, np.log10(0.5), 400)          # tau = 1 - T
T_grid = 1 - tau
P_star, v_l, v_g = solve_coexistence(T_grid)
dv = v_g - v_l


# ======================= Q2.4: the exponent beta ============================
ln_tau, ln_dv = np.log(tau), np.log(dv)

# Local (effective) exponent: slope of the log-log curve at each point
beta_eff = np.gradient(ln_dv, ln_tau)

# Global fit in the asymptotic window, and fits in a few windows for comparison
def fit_window(tau_max, tau_min=tau.min()):
    m = (tau >= tau_min) & (tau <= tau_max)
    slope, intercept = np.polyfit(ln_tau[m], ln_dv[m], 1)
    return slope, np.exp(intercept)

print("Q2.4 -- power-law fits  dv = A (1-T)^beta  over 1e-6 <= 1-T <= tau_max:")
for tau_max in (1e-4, 1e-3, 1e-2, 1e-1, 0.5):
    b, A = fit_window(tau_max)
    print(f"  tau_max = {tau_max:<7g}  beta = {b:.4f},  A = {A:.4f}")

beta_fit, A_fit = fit_window(1e-3)
print(f"\n  Best estimate (1-T <= 1e-3): beta = {beta_fit:.4f}  (exact: 1/2, A = 4)")

# Range of validity: where the local exponent is within 1% / 5% of 1/2
for tol in (0.01, 0.05):
    ok = np.abs(beta_eff / 0.5 - 1) < tol
    tau_valid = tau[ok & (np.cumsum(~ok) == 0)].max()  # contiguous from T_c
    print(f"  local exponent within {tol:.0%} of 1/2 for 1 - T < {tau_valid:.2g}"
          f"  (T > {1 - tau_valid:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

ax = axes[0]
ax.loglog(tau, dv, lw=2, label="numerical")
ax.loglog(tau, A_fit * tau**beta_fit, "k--", lw=1.5,
          label=rf"fit: ${A_fit:.3f}\,(1-\bar T)^{{{beta_fit:.4f}}}$")
ax.set_xlabel(r"$1 - \bar T$"); ax.set_ylabel(r"$\bar\nu_g - \bar\nu_l$")
ax.set_title(r"q2.4: Log–log plot: $\bar\nu_g - \bar\nu_l$ vs $1-\bar T$")
ax.legend(); ax.grid(alpha=0.3, which="both")

ax = axes[1]
ax.semilogx(tau, beta_eff, lw=2, label="local slope")
ax.axhline(0.5, color="k", ls="--", lw=1.5, label=r"$\beta = 1/2$")
ax.axhspan(0.5 * 0.99, 0.5 * 1.01, color="0.5", alpha=0.25, label="±1%")
ax.axhspan(0.5 * 0.95, 0.5 * 1.05, color="0.5", alpha=0.12, label="±5%")
ax.set_xlabel(r"$1 - \bar T$")
ax.set_ylabel(r"$d\ln(\bar\nu_g-\bar\nu_l)\,/\,d\ln(1-\bar T)$")
ax.set_title(r"q2.4: Effective exponent $\beta_{\rm eff}$")
ax.set_ylim(0.45, 0.8)  # beta_eff keeps rising far from T_c (to ~6 at T = 0.5)
ax.legend(loc="upper left"); ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("q2_4_beta.png", dpi=150)


# ======================= Q2.5: latent heat ==================================
# dP*/dT by a symmetric finite difference, solving at T +- h at every grid point.
# h is a small fraction of (1 - T), so T + h never crosses T_c. Using np.gradient
# on the grid itself would be less accurate at the coarse T = 0.5 end.
h = 1e-3 * tau
P_plus, _, _ = solve_coexistence(T_grid + h)
P_minus, _, _ = solve_coexistence(T_grid - h)
dPdT = (P_plus - P_minus) / (2 * h)

L_cc = ZC * T_grid * dv * dPdT                          # Clausius-Clapeyron
L_entropy = T_grid * np.log((3 * v_g - 1) / (3 * v_l - 1))  # T * delta s, check
L_asym = 6 * np.sqrt(tau)

print("\nQ2.5 -- latent heat per particle, L / (kB Tc):")
for T_check in (0.5, 0.7, 0.9, 0.99, 0.999):
    i = np.argmin(abs(T_grid - T_check))
    print(f"  T = {T_grid[i]:.3f}:  L = {L_cc[i]:.5f}   "
          f"(T*delta s = {L_entropy[i]:.5f},  6 sqrt(1-T) = {L_asym[i]:.5f})")
print(f"  max relative difference, Clausius-Clapeyron vs T*delta s: "
      f"{np.max(np.abs(L_cc / L_entropy - 1)):.1e}")

m = tau <= 1e-3
expo_L, pref_L = np.polyfit(np.log(tau[m]), np.log(L_cc[m]), 1)
print(f"  fit near T_c: L / (kB Tc) = {np.exp(pref_L):.4f} (1 - T)^{expo_L:.4f}"
      f"   (expected 6 (1 - T)^(1/2))")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

ax = axes[0]
ax.plot(T_grid, L_cc, lw=2, label="Clausius–Clapeyron")
ax.plot(T_grid[::12], L_entropy[::12], "o", ms=4, mfc="none",
        label=r"check: $\bar T \ln\frac{3\bar\nu_g-1}{3\bar\nu_l-1}$")
near = T_grid > 0.8
ax.plot(T_grid[near], L_asym[near], "k--", lw=1.5, label=r"asymptotic $6\sqrt{1-\bar T}$")
ax.set_xlabel(r"$\bar T$"); ax.set_ylabel(r"$L \,/\, k_B T_c$")
ax.set_title("q2.5: Latent heat per particle")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.loglog(tau, L_cc, lw=2, label="numerical")
ax.loglog(tau, 6 * np.sqrt(tau), "k--", lw=1.5, label=r"$6\,(1-\bar T)^{1/2}$")
ax.set_xlabel(r"$1 - \bar T$"); ax.set_ylabel(r"$L \,/\, k_B T_c$")
ax.set_title(r"q2.5: Scaling near $T_c$")
ax.legend(); ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("q2_5_latent_heat.png", dpi=150)
plt.show()