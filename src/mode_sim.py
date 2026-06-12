"""femwell waveguide eigenmode solve with Soref-Bennett carrier perturbation.

Open-source replacement for the Lumerical MODE/FDE script
(``../lumerical/Mode_Simulation.lsf``). Reads ``carriers.npz`` written
by ``charge_sim.py``, applies the Soref-Bennett free-carrier dispersion
model to convert (dn_e, dn_h) into a complex permittivity perturbation, and
solves the fundamental quasi-TE eigenmode of the silicon rib waveguide at
each voltage step.

Outputs:
    modulator_neff_V.dat - three whitespace-separated columns
                          [V, Re(dneff), Im(neff)] matching the format the
                          original LSF script emits, for direct comparison
                          against a Lumerical run.
    neff_vs_V.png        - the two-panel plot the LSF generates: relative
                          phase (rad/cm) and loss (dB/cm) vs voltage.

Coordinate frame note:
    The DEVSIM mesh (charge_sim.py) places y = 0 at the bottom of the slab,
    so silicon spans y in [0, thick_slab + thick_rib].
    The femwell mesh built here places y = 0 at the slab/rib interface, so
    the slab is y in [-thick_slab, 0] and the rib is y in [0, thick_rib].
    The perturbation callable handles the constant shift internally.

Run:
    uv run src/mode_sim.py     # requires carriers.npz from charge_sim.py
"""

from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg
from scipy.interpolate import RegularGridInterpolator
from skfem import Basis, ElementTriP0
from skfem.io.meshio import from_meshio

from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict

import params as p
from physics import soref_bennett

HERE = Path(__file__).resolve().parent
NPZ_PATH = HERE / "carriers.npz"
DAT_PATH = HERE / "modulator_neff_V.dat"
PNG_PATH = HERE.parent / "output" / "neff_vs_V.png"


def build_mesh():
    """Build the SOI rib waveguide cross-section in femwell.

    Three subdomains:
        ``core`` - silicon rib (width_rib x thick_rib).
        ``slab`` - silicon slab (width_slab x thick_slab).
        ``clad`` - SiO2 cladding box that wraps the whole cross-section.

    Mesh resolutions are ~20 nm in the rib core and ~40 nm in the slab -
    standard for a silicon photonics mode solve at 1550 nm. All units are
    micrometres (femwell's convention).

    Returns:
        ``(mesh, basis)`` ready to receive a complex permittivity field.
    """
    wg_w = p.width_rib * 1e6
    wg_t = p.thick_rib * 1e6
    slab_w = p.width_slab * 1e6
    slab_t = p.thick_slab * 1e6
    clad_t = 2.0  # 2 um cladding margin top and bottom

    core = sg.box(-wg_w / 2, 0, wg_w / 2, wg_t)
    slab = sg.box(-slab_w / 2, -slab_t, slab_w / 2, 0)
    clad = sg.box(-slab_w / 2, -clad_t, slab_w / 2, clad_t)

    polygons = OrderedDict(core=core, slab=slab, clad=clad)
    resolutions = dict(
        core={"resolution": 0.02, "distance": 0.5},
        slab={"resolution": 0.04, "distance": 0.5},
    )
    mesh = from_meshio(
        mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=10)
    )
    basis = Basis(mesh, ElementTriP0())
    return mesh, basis


def make_perturbation_callable(
    x_si: np.ndarray,
    y_si: np.ndarray,
    n_field: np.ndarray,
    p_field: np.ndarray,
    v_idx: int,
):
    """Return a callable dn_complex(x_um, y_um) for a single voltage step.

    Applies Soref-Bennett to the absolute carrier densities at voltage V and
    at V=0, then takes the difference. This correctly handles the non-linear
    hole term (Nh^0.8): applying it to a delta instead of to absolute values
    inflates the result by up to ~100x at low bias where p(V) ~ p(0).
    Converts dalpha to imag(n) via lambda/(4pi) and wraps the result in a
    RegularGridInterpolator. The callable is what ``basis.project(...)``
    consumes to assemble a per-element perturbation.

    Args:
        x_si, y_si: carrier-grid coordinates in metres (from carriers.npz).
        n_field, p_field: electron and hole densities, shape ``(n_V, nx, ny)``,
                          in m^-3.
        v_idx: index into the voltage axis to evaluate.

    Returns:
        A function mapping micron-scale (x, y) -> complex dn. Both inputs may
        be scalars or arrays.
    """
    # Apply SB to absolute carrier densities, then difference against V=0.
    # Applying the power-law hole term to the *delta* is only accurate when
    # p(V) >> p(0); at low bias (p(V) ~ p(0)) it inflates dn_h by up to 100x.
    Ne_v = n_field[v_idx] * 1e-6   # m^-3 -> cm^-3
    Nh_v = p_field[v_idx] * 1e-6
    Ne_0 = n_field[0] * 1e-6
    Nh_0 = p_field[0] * 1e-6
    dn_v, da_v = soref_bennett(Ne_v, Nh_v)
    dn_0, da_0 = soref_bennett(Ne_0, Nh_0)
    dn_real = dn_v - dn_0
    da = da_v - da_0
    # imag(n) = alpha [1/cm] x lambda [cm] / (4pi), where alpha is intensity absorption.
    wavelength_cm = p.WAVELENGTH * 1e2
    dn_imag = da * wavelength_cm / (4 * np.pi)
    dn_complex = dn_real + 1j * dn_imag

    interp = RegularGridInterpolator(
        (x_si, y_si), dn_complex, bounds_error=False, fill_value=0.0,
    )

    def delta_n(x_um, y_um):
        # femwell frame has y=0 at the slab/rib interface; DEVSIM frame has
        # y=0 at the bottom of the slab. Shift by thick_slab.
        x_m = x_um * 1e-6
        y_m = y_um * 1e-6 + p.thick_slab
        pts = np.stack([x_m, y_m], axis=-1)
        return interp(pts)

    return delta_n


def solve_voltage_sweep(data) -> tuple[np.ndarray, np.ndarray]:
    """Sweep voltage, solve eigenmode at each step, return complex neff(V).

    Builds a fresh complex permittivity field per voltage by projecting the
    Soref-Bennett perturbation onto the silicon subdomains, then runs
    ``compute_modes`` for the fundamental quasi-TE mode (num_modes=1,
    order=2). The V = 0 step uses unperturbed silicon - by construction,
    dn(V=0) = 0 since the perturbation references the V=0 carrier field.

    Args:
        data: dict-like loaded from ``carriers.npz`` with keys
              ``V, x, y, n, p``.

    Returns:
        ``(V_arr, neff_arr)`` where ``neff_arr`` is complex, shape ``(n_V,)``.
    """
    V_arr = data["V"]
    x_si = data["x"]
    y_si = data["y"]
    n_field = data["n"]
    p_field = data["p"]
    wavelength_um = p.WAVELENGTH * 1e6

    _, basis = build_mesh()
    neff_arr = np.empty(len(V_arr), dtype=complex)

    for i, V in enumerate(V_arr):
        eps = basis.zeros(dtype=complex)
        # Background silicon index in core + slab.
        for subdomain in ("core", "slab"):
            eps[basis.get_dofs(elements=subdomain)] = p.N_SI

        if i > 0:
            dn = make_perturbation_callable(x_si, y_si, n_field, p_field, i)
            eps += basis.project(lambda x: dn(x[0], x[1]), dtype=complex)

        # Cladding overwrite (after projection so it wipes any spillover).
        eps[basis.get_dofs(elements="clad")] = p.N_SIO2
        eps *= eps  # n -> eps

        modes = compute_modes(
            basis, eps, wavelength=wavelength_um, num_modes=1, order=2,
        )
        neff_arr[i] = modes[0].n_eff
        print(
            f"  V = {V:5.2f} V  "
            f"Re(neff) = {neff_arr[i].real:.6f}  "
            f"Im(neff) = {neff_arr[i].imag:.3e}"
        )

    return V_arr, neff_arr


def write_dat(V_arr: np.ndarray, neff_arr: np.ndarray, path: Path) -> None:
    """Write ``modulator_neff_V.dat`` matching the LSF column layout.

    Three whitespace-separated columns: ``V``, ``Re(dneff)`` (referenced to
    V = 0), and ``Im(neff)``. This is the artifact intended for diffing
    against a Lumerical run.
    """
    dneff_real = np.real(neff_arr) - np.real(neff_arr[0])
    data = np.column_stack([V_arr, dneff_real, np.imag(neff_arr)])
    np.savetxt(path, data, fmt="% .9e", header="V   dRe(neff)   Im(neff)")


def plot_results(V_arr: np.ndarray, neff_arr: np.ndarray) -> None:
    """Reproduce the two plots emitted by the Lumerical mode script.

    Relative phase ``2pi * dneff / lambda`` in rad/cm, and loss
    ``0.4pi * log10(e) * Im(neff) / lambda`` in dB/cm.
    """
    wavelength_m = p.WAVELENGTH
    dneff_real = np.real(neff_arr) - np.real(neff_arr[0])
    rel_phase_per_cm = 2 * np.pi * dneff_real / wavelength_m * 1e-2
    alpha_dB_cm = 0.4 * np.pi * np.log10(np.e) * np.imag(neff_arr) / wavelength_m

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(V_arr, rel_phase_per_cm, marker=".")
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_ylabel("Relative phase (rad/cm)")
    axes[0].grid(True)

    axes[1].plot(V_arr, alpha_dB_cm, marker=".")
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel("Loss (dB/cm)")
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=120)
    plt.close(fig)


def main() -> None:
    """End-to-end: load carriers, sweep modes, write .dat and plot."""
    print(f"Loading {NPZ_PATH}...")
    data = np.load(NPZ_PATH)

    print("Running eigenmode sweep...")
    V_arr, neff_arr = solve_voltage_sweep(data)

    print(f"Writing {DAT_PATH}...")
    write_dat(V_arr, neff_arr, DAT_PATH)

    print(f"Writing {PNG_PATH}...")
    plot_results(V_arr, neff_arr)

    print("Done.")


if __name__ == "__main__":
    main()
