"""Compare the Python pipeline's ``modulator_neff_V.dat`` against a
Lumerical run.

Both files share the same ``[V, Re(Δneff), Im(neff)]`` column layout. The
Lumerical export from ``Mode_Simulation.lsf`` concatenates four 41-row
blocks: two warm-up blocks at numerical-noise level followed by two
identical copies of the real sweep. This script picks the last block.

Outputs:
    compare_neff.png     — two-panel overlay (Δneff and Im(neff) vs V) plus
                           a third panel of per-voltage Python/Lumerical
                           ratios in the low-bias regime.
    Prints a side-by-side numerical table to stdout.

Run:
    uv run src/compare_lumerical.py [path/to/lumerical_data.dat]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PY_DAT = HERE / "modulator_neff_V.dat"
DEFAULT_LUM_DAT = HERE.parent / "lumerical" / "lumerical_data.dat"
OUT_PNG = HERE.parent / "output" / "compare_neff.png"


def load_lumerical(path: Path) -> tuple[np.ndarray, int]:
    """Return the last 41-row block of ``lumerical_data.dat``.

    Lumerical's ``write`` pattern in the LSF emits one block per sweep
    variable; the last block is the actual ``modulator_neff_V`` result.
    """
    raw = np.loadtxt(path)
    if raw.shape[0] % 41 != 0:
        raise ValueError(
            f"{path}: expected a multiple of 41 rows, got {raw.shape[0]}."
        )
    n_blocks = raw.shape[0] // 41
    return raw[-41:], n_blocks


def load_python(path: Path) -> np.ndarray:
    """Load the Python pipeline's ``modulator_neff_V.dat`` (header-aware)."""
    return np.loadtxt(path)


def print_table(V, py, lu) -> None:
    """Print V, Re(Δneff) py/lum, Im(neff) py/lum, ratios as a fixed-width table."""
    print()
    print("  V       dRe(neff)_py    dRe(neff)_lum   ratio    "
          "Im(neff)_py     Im(neff)_lum    ratio")
    print("  " + "-" * 96)
    for i in range(len(V)):
        dpy, dlu = py[i, 1], lu[i, 1]
        ipy, ilu = py[i, 2], lu[i, 2]
        ratio_d = dpy / dlu if abs(dlu) > 1e-12 else np.nan
        ratio_i = ipy / ilu if abs(ilu) > 1e-12 else np.nan
        print(
            f"  {V[i]:4.1f}   {dpy: .4e}    {dlu: .4e}   "
            f"{ratio_d: 6.2f}   {ipy: .4e}    {ilu: .4e}   {ratio_i: 6.2f}"
        )


def plot(V, py, lu) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(V, py[:, 1], marker=".", label="Python (DEVSIM + femwell)")
    ax.plot(V, lu[:, 1], marker="x", label="Lumerical")
    ax.set_xlabel("V (V)")
    ax.set_ylabel("Re(Δneff)")
    ax.set_title("Real index change vs voltage")
    ax.grid(True)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.semilogy(V, np.abs(py[:, 2]), marker=".", label="Python")
    ax.semilogy(V, np.abs(lu[:, 2]), marker="x", label="Lumerical")
    ax.set_xlabel("V (V)")
    ax.set_ylabel("|Im(neff)|")
    ax.set_title("Loss (log scale)")
    ax.grid(True, which="both")
    ax.legend(fontsize=9)

    # Low-bias zoom (where both solvers agree on the perturbation regime).
    ax = axes[2]
    mask = V <= 1.0
    ax.plot(V[mask], py[mask, 1], marker=".", label="Python")
    ax.plot(V[mask], lu[mask, 1], marker="x", label="Lumerical")
    ax.set_xlabel("V (V)")
    ax.set_ylabel("Re(Δneff)")
    ax.set_title("Low-bias zoom (V ≤ 1.0 V)")
    ax.grid(True)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved {OUT_PNG}")


def main() -> None:
    lum_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LUM_DAT
    print(f"Loading Python:    {PY_DAT}")
    print(f"Loading Lumerical: {lum_path}")
    py = load_python(PY_DAT)
    lu, n_blocks = load_lumerical(lum_path)
    print(f"Lumerical file contains {n_blocks} blocks; using the last.")

    if not np.allclose(py[:, 0], lu[:, 0]):
        raise ValueError("Voltage grids do not match.")
    V = py[:, 0]

    print_table(V, py, lu)
    plot(V, py, lu)


if __name__ == "__main__":
    main()
