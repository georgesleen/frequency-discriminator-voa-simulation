"""Unit tests for the pure physics formulas in ``src/physics.py``.

Each test pins a hand-computed value rather than re-deriving the formula.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import physics as ph


# --- soref_bennett ---------------------------------------------------------

def test_soref_zero_carriers_zero_perturbation():
    dn, da = ph.soref_bennett(np.array([0.0]), np.array([0.0]))
    assert dn[0] == 0.0
    assert da[0] == 0.0


def test_soref_signs():
    """Carriers lower the index (dn < 0) and add loss (dalpha > 0)."""
    dn, da = ph.soref_bennett(np.array([1e18]), np.array([1e18]))
    assert dn[0] < 0.0
    assert da[0] > 0.0


def test_soref_electron_term_matches_formula():
    Ne = 1e18
    dn, da = ph.soref_bennett(np.array([Ne]), np.array([0.0]))
    assert dn[0] == pytest.approx(-ph.SB_DN_E * Ne)
    assert da[0] == pytest.approx(ph.SB_DA_E * Ne)


def test_soref_hole_power_law():
    Nh = 1e18
    dn, _ = ph.soref_bennett(np.array([0.0]), np.array([Nh]))
    assert dn[0] == pytest.approx(-ph.SB_DN_H_A * Nh**ph.SB_DN_H_B)


def test_soref_negative_hole_clipped():
    """Negative dN must not blow up the fractional-power hole term."""
    dn, _ = ph.soref_bennett(np.array([0.0]), np.array([-1e18]))
    assert np.isfinite(dn[0])
    assert dn[0] == 0.0  # max(dNh, 0) kills the hole term


# --- scharfetter_lifetime --------------------------------------------------

def test_lifetime_low_doping_approaches_tau_max():
    tau = ph.scharfetter_lifetime(np.array([1e12]), tau_max=ph.SRH_TAU_N)
    assert tau[0] == pytest.approx(ph.SRH_TAU_N, rel=1e-3)


def test_lifetime_at_nref_is_half():
    """At N = N_ref with gamma=1, tau = tau_min + (tau_max-tau_min)/2."""
    tau = ph.scharfetter_lifetime(np.array([ph.SRH_NREF]), tau_max=ph.SRH_TAU_N)
    assert tau[0] == pytest.approx(0.5 * ph.SRH_TAU_N)


def test_lifetime_high_doping_drops():
    """Heavy implant doping shortens the lifetime by orders of magnitude."""
    tau = ph.scharfetter_lifetime(np.array([4e20]), tau_max=ph.SRH_TAU_N)
    assert tau[0] < ph.SRH_TAU_N / 1e4


def test_lifetime_monotonic_decreasing():
    N = np.array([1e14, 1e16, 1e18, 1e20])
    tau = ph.scharfetter_lifetime(N, tau_max=ph.SRH_TAU_N)
    assert np.all(np.diff(tau) < 0.0)


# --- srh_recombination -----------------------------------------------------

def test_srh_equilibrium_zero():
    """At n*p == n_i^2 the net rate vanishes."""
    n_i = ph.N_I
    U = ph.srh_recombination(
        np.array([n_i]), np.array([n_i]), tau_n=1e-6, tau_p=1e-6
    )
    assert U[0] == pytest.approx(0.0, abs=1e-3)


def test_srh_high_injection_value():
    n = p = np.array([1e18])
    tau = 1e-6
    U = ph.srh_recombination(n, p, tau_n=tau, tau_p=tau)
    expected = (1e36 - ph.N_I**2) / (tau * (1e18 + ph.N_I) + tau * (1e18 + ph.N_I))
    assert U[0] == pytest.approx(expected, rel=1e-9)


# --- auger_recombination ---------------------------------------------------

def test_auger_equilibrium_zero():
    n_i = ph.N_I
    U = ph.auger_recombination(np.array([n_i]), np.array([n_i]))
    assert U[0] == pytest.approx(0.0, abs=1e-3)


def test_auger_value_matches_formula():
    n = p = 1e19
    U = ph.auger_recombination(np.array([n]), np.array([p]))
    expected = (ph.AUGER_CN * n + ph.AUGER_CP * p) * (n * p - ph.N_I**2)
    assert U[0] == pytest.approx(expected, rel=1e-12)


def test_auger_dominates_srh_at_high_injection():
    """The reason Auger is needed: above ~1e18 it outpaces SRH recombination."""
    n = p = np.array([1e19])
    tau = ph.scharfetter_lifetime(np.array([1e15]), tau_max=ph.SRH_TAU_N)
    u_srh = ph.srh_recombination(n, p, tau_n=tau, tau_p=tau)
    u_aug = ph.auger_recombination(n, p)
    assert u_aug[0] > u_srh[0]
