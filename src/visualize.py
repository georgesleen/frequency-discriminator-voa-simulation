"""Carrier density maps for the SOI PIN modulator cross-section.

Reads ``carriers.npz`` written by ``charge_sim.py`` and produces
``output/carrier_maps.png`` — two rows of four panels:

  Row 1: absolute electron density n(x,y) at four voltages, full slab width.
  Row 2: injected carrier change Δn = n(V) − n(0) at the same voltages,
         zoomed to ±1.5 µm around the waveguide rib.

Run:
    uv run src/visualize.py
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import params as p

HERE = Path(__file__).resolve().parent
NPZ_PATH = HERE / "carriers.npz"
OUT_PATH = HERE.parent / "output" / "carrier_maps.png"

PLOT_VOLTAGES = [0.0, 1.0, 1.6, 4.0]
ZOOM_X_UM = 1.5   # µm half-width for the rib-region zoom (row 2)


def _rib_patch(**kw):
    """Return a Rectangle that outlines the rib in plot coordinates (µm / nm)."""
    x0 = -p.width_rib / 2 * 1e6
    y0 = p.thick_slab * 1e9
    w = p.width_rib * 1e6
    h = p.thick_rib * 1e9
    return Rectangle((x0, y0), w, h, fill=False, linewidth=1.2, **kw)


def main() -> None:
    data = np.load(NPZ_PATH)
    V_arr = data["V"]
    x_um = data["x"] * 1e6          # m → µm
    y_nm = data["y"] * 1e9          # m → nm
    n_cm3 = data["n"] * 1e-6        # m^-3 → cm^-3, shape (41, 301, 81)

    v_idx = [int(np.argmin(np.abs(V_arr - v))) for v in PLOT_VOLTAGES]
    n_base = n_cm3[0]                # V = 0 baseline

    fig, axes = plt.subplots(2, len(PLOT_VOLTAGES), figsize=(14, 5.5))

    # ── Row 1: absolute n, log scale, full slab ───────────────────────────────
    norm_abs = mcolors.LogNorm(vmin=1e8, vmax=6e20)
    im1 = None
    for col, (vi, V) in enumerate(zip(v_idx, PLOT_VOLTAGES)):
        ax = axes[0, col]
        im1 = ax.pcolormesh(x_um, y_nm, n_cm3[vi].T,
                            norm=norm_abs, cmap="inferno", shading="auto")
        ax.add_patch(_rib_patch(edgecolor="white", linestyle="--"))
        ax.axhline(p.thick_slab * 1e9, color="white", linewidth=0.7, linestyle=":")
        # p++ and n++ centre lines
        ax.axvline(p.x_center_p * 1e6, color="cyan",  linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(p.x_center_n * 1e6, color="lime",  linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"V = {V:.1f} V", fontsize=10)
        ax.set_xlabel("x (µm)", fontsize=9)
        if col == 0:
            ax.set_ylabel("y (nm)", fontsize=9)
            ax.text(-3.1, 190, "abs n", fontsize=8, color="white", va="top")
        ax.tick_params(labelsize=8)

    cb1 = fig.colorbar(im1, ax=axes[0, :], label="n  (cm⁻³)", shrink=0.85, pad=0.01)
    cb1.ax.tick_params(labelsize=8)

    # ── Row 2: Δn = n(V)−n(0), rib zoom, log scale ───────────────────────────
    FLOOR = 1e8   # clip small/negative values to this floor before log
    norm_dn = mcolors.LogNorm(vmin=FLOOR, vmax=6e20)
    im2 = None
    for col, (vi, V) in enumerate(zip(v_idx, PLOT_VOLTAGES)):
        ax = axes[1, col]
        dn = np.maximum(n_cm3[vi] - n_base, FLOOR)
        im2 = ax.pcolormesh(x_um, y_nm, dn.T,
                            norm=norm_dn, cmap="plasma", shading="auto")
        ax.add_patch(_rib_patch(edgecolor="white", linestyle="--"))
        ax.axhline(p.thick_slab * 1e9, color="white", linewidth=0.7, linestyle=":")
        ax.set_xlim(-ZOOM_X_UM, ZOOM_X_UM)
        ax.set_xlabel("x (µm)", fontsize=9)
        if col == 0:
            ax.set_ylabel("y (nm)", fontsize=9)
            ax.text(-1.4, 190, "Δn vs 0V", fontsize=8, color="white", va="top")
        ax.tick_params(labelsize=8)

    cb2 = fig.colorbar(im2, ax=axes[1, :], label="Δn  (cm⁻³)", shrink=0.85, pad=0.01)
    cb2.ax.tick_params(labelsize=8)

    # Legend for the dashed lines in row 1
    axes[0, 0].plot([], [], color="cyan",  ls="--", lw=0.8, label="p++ centre")
    axes[0, 0].plot([], [], color="lime",  ls="--", lw=0.8, label="n++ centre")
    axes[0, 0].legend(fontsize=7, loc="lower left", framealpha=0.4)

    fig.suptitle("SOI PIN modulator — carrier injection vs forward bias", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
