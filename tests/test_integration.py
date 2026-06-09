"""End-to-end DEVSIM + femwell regression test (the ``integration`` tier).

Runs a short voltage sweep through the real charge and mode solvers and
checks the result against the Lumerical reference. This is the executable
form of the project goal - "stay close to Lumerical" - and the regression
guard for the recombination + mobility physics. Needs the FHS + MKL env, so
it is marked ``integration`` and skipped by the default ``pytest`` run.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

SRC = Path(__file__).resolve().parent.parent / "src"
LUM_DAT = Path(__file__).resolve().parent.parent / "lumerical" / "lumerical_data.dat"
sys.path.insert(0, str(SRC))

# A sub-turn-on point, the turn-on knee, and a forward-injection point where
# the old SRH-only model ran away by ~25x.
TEST_VOLTAGES = np.array([0.0, 0.8, 1.5])


def _lumerical_rows(voltages):
    """Return the Lumerical [V, dRe(neff), Im(neff)] rows at ``voltages``.

    The file is four stacked 41-row blocks; the last is the real sweep.
    """
    raw = np.loadtxt(LUM_DAT)
    block = raw[-41:]
    idx = [int(round(v / 0.1)) for v in voltages]
    return block[idx]


@pytest.fixture(scope="module")
def sweep():
    """Run charge + mode at ``TEST_VOLTAGES`` once for the whole module."""
    import charge_sim as cs
    import mode_sim as ms

    cs.build_mesh(cs.MSH_PATH)
    cs.setup_device()
    cs.set_doping()
    cs.initial_potential_solution()
    cs.initial_dd_solution()
    V, xn, yn, n_nodes, p_nodes = cs.voltage_sweep(TEST_VOLTAGES)
    xg, yg, ng, pg = cs.sample_to_grid(xn, yn, n_nodes, p_nodes)

    data = {"V": V, "x": xg, "y": yg, "n": ng, "p": pg}
    V_arr, neff = ms.solve_voltage_sweep(data)
    dneff = np.real(neff) - np.real(neff[0])
    return {"V": V_arr, "neff": neff, "dneff": dneff, "n": ng, "p": pg}


def test_carrier_fields_finite_and_bounded(sweep):
    """No NaNs, and injection stays near the implant level (no runaway)."""
    assert np.isfinite(sweep["n"]).all()
    assert np.isfinite(sweep["p"]).all()
    # Implant peak is 4e20 cm^-3 = 4e26 m^-3; the old runaway hit ~1e27.
    assert sweep["n"].max() < 6e26
    assert sweep["p"].max() < 6e26


def test_index_change_sign_and_monotonic(sweep):
    """Carriers lower the index, and |dneff| grows with forward bias."""
    dneff = sweep["dneff"]
    assert dneff[0] == 0.0
    assert np.all(dneff[1:] < 0.0)
    assert abs(dneff[2]) > abs(dneff[1])


def test_no_forward_bias_runaway(sweep):
    """The 1.5 V point must not blow up the way the SRH-only model did.

    SRH-only Python gave dneff(1.5 V) ~ -0.45 (vs Lumerical ~ -0.019); with
    SRH(Scharfetter) + Auger + Klaassen it is ~ -0.10. The 0.15 bound guards
    against regressing toward the old runaway.
    """
    assert abs(sweep["dneff"][2]) < 0.15


def test_close_to_lumerical(sweep):
    """dneff tracks Lumerical within a loose factor above turn-on.

    This is a regression guard, not a tight match: the current pipeline runs
    ~1.2x at 0.8 V and ~5.4x at 1.5 V. The (0.3, 7.0) band catches both a
    regression toward the old ~25x runaway and any large over-correction.
    """
    lum = _lumerical_rows(TEST_VOLTAGES)
    for i in (1, 2):  # skip V=0 (both identically zero)
        ratio = sweep["dneff"][i] / lum[i, 1]
        assert 0.3 < ratio < 7.0, (
            f"V={TEST_VOLTAGES[i]}: py={sweep['dneff'][i]:.3e} "
            f"lum={lum[i, 1]:.3e} ratio={ratio:.2f}"
        )
