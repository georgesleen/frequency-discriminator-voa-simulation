"""Pure device-physics formulas shared by the charge and mode solvers.

No ``devsim`` or ``femwell`` imports, so every function here is unit-testable
on its own (numpy only). ``charge_sim.py`` reuses the constants below to build
its DEVSIM expression strings, so the strings can't drift from the formulas
tested here; ``mode_sim.py`` calls ``soref_bennett`` directly.

Units (DEVSIM CGS convention, matching ``charge_sim.py``):
    carrier densities  cm^-3
    recombination rate  cm^-3 s^-1
    lifetime            s
    Auger coefficient   cm^6 s^-1
"""

from __future__ import annotations

import numpy as np

# --- silicon constants (match devsim.python_packages.simple_physics) -------
N_I = 1.0e10            # intrinsic carrier density [cm^-3] at 300 K

# --- Soref-Bennett free-carrier dispersion at 1550 nm ----------------------
# dn from carriers (carriers lower the index); dalpha is intensity absorption
# [1/cm]. dN inputs are in cm^-3.
SB_DN_E = 8.8e-22       # dn_e = -SB_DN_E * Ne
SB_DN_H_A = 8.5e-18     # dn_h = -SB_DN_H_A * Nh^SB_DN_H_B
SB_DN_H_B = 0.8
SB_DA_E = 8.5e-18       # dalpha_e = SB_DA_E * Ne   [1/cm]
SB_DA_H = 6.0e-18       # dalpha_h = SB_DA_H * Nh   [1/cm]

# --- Auger recombination coefficients (silicon) ----------------------------
AUGER_CN = 2.8e-31      # electron Auger coefficient [cm^6/s]
AUGER_CP = 9.9e-32      # hole Auger coefficient [cm^6/s]

# --- doping-dependent SRH lifetime (Scharfetter relation) ------------------
# tau(N) = tau_min + (tau_max - tau_min) / (1 + (N / N_ref)^gamma), where N is
# the total impurity concentration (acceptors + donors). Standard silicon
# values; N_ref and tau_max double as the main tuning knobs if a residual
# offset against Lumerical remains.
SRH_TAU_MIN = 0.0       # [s]
SRH_TAU_N = 1.0e-5      # electron tau_max [s]
SRH_TAU_P = 3.0e-6      # hole tau_max [s]
SRH_NREF = 1.0e16       # reference concentration [cm^-3]
SRH_GAMMA = 1.0


def soref_bennett(dNe: np.ndarray, dNh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute dn and dalpha from carrier-density changes.

    Args:
        dNe: change in electron density vs equilibrium [cm^-3].
        dNh: change in hole density vs equilibrium [cm^-3].

    Returns:
        ``(dn, dalpha)`` - real-index change [dimensionless] and intensity
        absorption change [1/cm]. Positive carrier density gives negative dn
        (carriers lower the index) and positive dalpha (carriers add loss).
        Negative dN is clipped to 0 in the hole power-law term, which is
        defined only for positive arguments.
    """
    dn = -SB_DN_E * dNe - SB_DN_H_A * np.power(np.maximum(dNh, 0.0), SB_DN_H_B)
    da = SB_DA_E * dNe + SB_DA_H * dNh
    return dn, da


def scharfetter_lifetime(
    N: np.ndarray,
    tau_max: float,
    n_ref: float = SRH_NREF,
    gamma: float = SRH_GAMMA,
    tau_min: float = SRH_TAU_MIN,
) -> np.ndarray:
    """Doping-dependent SRH lifetime via the Scharfetter relation.

    Args:
        N: total impurity concentration (acceptors + donors) [cm^-3].
        tau_max: low-doping lifetime limit [s] (``SRH_TAU_N`` / ``SRH_TAU_P``).
        n_ref: reference concentration [cm^-3].
        gamma: roll-off exponent.
        tau_min: high-doping lifetime limit [s].

    Returns:
        Lifetime [s], approaching ``tau_max`` for N << n_ref and ``tau_min``
        for N >> n_ref.
    """
    return tau_min + (tau_max - tau_min) / (1.0 + np.power(N / n_ref, gamma))


def srh_recombination(
    n: np.ndarray,
    p: np.ndarray,
    tau_n: np.ndarray,
    tau_p: np.ndarray,
    n_i: float = N_I,
) -> np.ndarray:
    """Shockley-Read-Hall net recombination rate (midgap trap).

    Args:
        n, p: electron and hole densities [cm^-3].
        tau_n, tau_p: electron and hole lifetimes [s] (scalar or per-point).
        n_i: intrinsic carrier density [cm^-3]; sets ``n1 = p1 = n_i``.

    Returns:
        Net recombination rate [cm^-3 s^-1]; positive when ``n*p > n_i^2``.
    """
    return (n * p - n_i**2) / (tau_p * (n + n_i) + tau_n * (p + n_i))


def auger_recombination(
    n: np.ndarray,
    p: np.ndarray,
    n_i: float = N_I,
    cn: float = AUGER_CN,
    cp: float = AUGER_CP,
) -> np.ndarray:
    """Auger net recombination rate.

    Args:
        n, p: electron and hole densities [cm^-3].
        n_i: intrinsic carrier density [cm^-3].
        cn, cp: electron and hole Auger coefficients [cm^6/s].

    Returns:
        Net recombination rate [cm^-3 s^-1]; dominates over SRH above
        ~1e18 cm^-3, capping high-injection carrier density.
    """
    return (cn * n + cp * p) * (n * p - n_i**2)
