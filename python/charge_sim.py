"""DEVSIM PIN-junction drift-diffusion simulation.

Open-source replacement for the Lumerical CHARGE script
(``simulation_scripts/Charge_Simulation.lsf``). Produces ``carriers.npz``,
which the downstream mode-solver script (``mode_sim.py``) reads to apply
the Soref-Bennett free-carrier perturbation.

Pipeline:
    1. Generate an L-shaped silicon cross-section mesh in gmsh (slab + rib).
    2. Import into DEVSIM, attach ohmic anode and cathode contacts on top of
       the slab.
    3. Apply doping profiles (p-epi background, Gaussian p++ and n++ implants
       from the top of the slab).
    4. Initial Poisson solve, then full drift-diffusion solve at equilibrium.
    5. Sweep anode voltage from ``voltage_start`` to ``voltage_stop`` in
       ``voltage_interval`` steps, recording electron and hole densities at
       each mesh node.
    6. Resample the carrier fields onto a regular (x, y) grid and save.

The mesh is L-shaped (silicon polygon) rather than a bounding rectangle so
the rib protrudes correctly above the slab — this matches the geometry the
Lumerical LSF builds via two overlapping rectangles.

Caveats:
    - The Lumerical 'adddope' pepi command in the LSF does not specify a
      concentration. We assume a 1e15 cm^-3 p-type background here; adjust
      ``PEPI_CONC`` if the actual Lumerical default differs.
    - The implant profile is a step function laterally (within the implant
      window) and a Gaussian vertically decaying from the top of the slab,
      with vertical sigma = thick_slab / 3. This is the standard shape for
      an ion-implant approximation with junction_width = 0.

Run:
    uv run python charge_sim.py    # produces carriers.npz next to this file
"""

from pathlib import Path

import numpy as np
import gmsh
from scipy.interpolate import griddata

from devsim import (
    add_gmsh_contact,
    add_gmsh_region,
    create_device,
    create_gmsh_mesh,
    finalize_mesh,
    get_node_model_values,
    node_model,
    set_node_values,
    set_parameter,
    solve,
)
from devsim.python_packages.model_create import CreateSolution
from devsim.python_packages.simple_physics import (
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    GetContactBiasName,
    SetSiliconParameters,
)

import params as p

HERE = Path(__file__).resolve().parent
MSH_PATH = HERE / "pin_mesh.msh"
NPZ_PATH = HERE / "carriers.npz"

DEVICE = "pin"
REGION = "bulk"

# Background p-epi concentration (LSF leaves this unspecified; typical Lumerical default).
PEPI_CONC = 1e15 * 1e6  # m^-3


def build_mesh(path: Path) -> None:
    """Generate the L-shaped silicon cross-section mesh via gmsh.

    The polygon walks counterclockwise around the silicon (slab + rib).
    Three physical groups are tagged in the resulting ``.msh``:

        * ``Bulk``            — the silicon surface (2D).
        * ``anode_contact``   — the line segment on top of the slab at
                                ``x ≈ -center_contact`` where the anode metal
                                would sit.
        * ``cathode_contact`` — the equivalent on the +x side.

    Mesh sizing comes from ``params.py``: ``max_edge_length_override`` at the
    rib corners (~7 nm) to resolve the depletion-edge gradient, and the
    coarser ``max_edge_length`` everywhere else.

    Args:
        path: Destination ``.msh`` file path.
    """
    xs = p.width_slab / 2
    rs = p.width_rib / 2
    cl = p.center_contact - p.width_contact / 2  # contact inner edge (absolute)
    cr = p.center_contact + p.width_contact / 2  # contact outer edge
    y0 = 0.0
    y1 = p.thick_slab
    y2 = p.thick_slab + p.thick_rib
    lc_coarse = p.max_edge_length
    lc_fine = p.max_edge_length_override

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("pin")
    geo = gmsh.model.geo

    # 12 corners, counterclockwise from bottom-left.
    pts = [
        geo.add_point(-xs, y0, 0, lc_coarse),  # 0  bottom-left
        geo.add_point( xs, y0, 0, lc_coarse),  # 1  bottom-right
        geo.add_point( xs, y1, 0, lc_coarse),  # 2  slab top-right corner
        geo.add_point( cr, y1, 0, lc_coarse),  # 3  cathode outer edge
        geo.add_point( cl, y1, 0, lc_coarse),  # 4  cathode inner edge
        geo.add_point( rs, y1, 0, lc_fine),    # 5  rib base right
        geo.add_point( rs, y2, 0, lc_fine),    # 6  rib top right
        geo.add_point(-rs, y2, 0, lc_fine),    # 7  rib top left
        geo.add_point(-rs, y1, 0, lc_fine),    # 8  rib base left
        geo.add_point(-cl, y1, 0, lc_coarse),  # 9  anode inner edge
        geo.add_point(-cr, y1, 0, lc_coarse),  # 10 anode outer edge
        geo.add_point(-xs, y1, 0, lc_coarse),  # 11 slab top-left corner
    ]
    n = len(pts)
    lines = [geo.add_line(pts[i], pts[(i + 1) % n]) for i in range(n)]
    # lines[3] = (cathode outer) -> (cathode inner)  ← cathode contact
    # lines[9] = (anode inner)   -> (anode outer)    ← anode contact

    loop = geo.add_curve_loop(lines)
    surface = geo.add_plane_surface([loop])

    geo.synchronize()

    gmsh.model.add_physical_group(2, [surface], name="Bulk")
    gmsh.model.add_physical_group(1, [lines[9]], name="anode_contact")
    gmsh.model.add_physical_group(1, [lines[3]], name="cathode_contact")

    gmsh.model.mesh.generate(2)
    gmsh.write(str(path))
    gmsh.finalize()


def setup_device() -> None:
    """Import the gmsh mesh into DEVSIM and apply silicon material parameters.

    Creates the device, maps the ``Bulk`` physical surface to a DEVSIM
    region, and attaches both contacts. Material parameters (mobilities,
    bandgap, etc.) come from DEVSIM's bundled silicon model at the
    temperature in ``params.TEMPERATURE_K``.
    """
    create_gmsh_mesh(mesh=DEVICE, file=str(MSH_PATH))
    add_gmsh_region(mesh=DEVICE, gmsh_name="Bulk", region=REGION, material="Silicon")
    add_gmsh_contact(
        mesh=DEVICE, gmsh_name="anode_contact",
        region=REGION, material="metal", name="anode",
    )
    add_gmsh_contact(
        mesh=DEVICE, gmsh_name="cathode_contact",
        region=REGION, material="metal", name="cathode",
    )
    finalize_mesh(mesh=DEVICE)
    create_device(mesh=DEVICE, device=DEVICE)
    SetSiliconParameters(DEVICE, REGION, p.TEMPERATURE_K)


def set_doping() -> None:
    """Define ``Acceptors``, ``Donors``, and ``NetDoping`` node models.

    Three contributions:
        * pepi:      constant p-type background (``PEPI_CONC``) everywhere.
        * p++ implant: Gaussian peak ``surface_conc_p`` at the top of the
                       slab, decaying downward with sigma = slab/3; bounded
                       laterally to ``[x_center_p ± x_span_p/2]``.
        * n++ implant: same shape on the opposite side of the rib.

    Step functions implement the lateral implant window. DEVSIM evaluates
    these expressions per node — ``x`` and ``y`` refer to node coordinates.
    """
    p_left = p.x_center_p - p.x_span_p / 2
    p_right = p.x_center_p + p.x_span_p / 2
    n_left = p.x_center_n - p.x_span_n / 2
    n_right = p.x_center_n + p.x_span_n / 2
    sigma_y = p.thick_slab / 3.0
    y_top = p.thick_slab  # implant source face (LSF face_p = 5 = upper z)

    def gaussian_implant(conc: float, xl: float, xr: float) -> str:
        """Return a DEVSIM expression for an implant profile.

        Step function in x within ``[xl, xr]``; Gaussian in y decaying from
        ``y_top`` downward with the shared sigma.
        """
        return (
            f"{conc:.6e}"
            f" * step(x - ({xl:.6e}))"
            f" * step(({xr:.6e}) - x)"
            f" * exp(-(({y_top:.6e} - y) * ({y_top:.6e} - y))"
            f"       / ({sigma_y:.6e} * {sigma_y:.6e}))"
        )

    p_implant = gaussian_implant(p.surface_conc_p, p_left, p_right)
    n_implant = gaussian_implant(p.surface_conc_n, n_left, n_right)

    node_model(
        device=DEVICE, region=REGION, name="Acceptors",
        equation=f"{PEPI_CONC:.6e} + {p_implant};",
    )
    node_model(
        device=DEVICE, region=REGION, name="Donors",
        equation=f"{n_implant};",
    )
    node_model(
        device=DEVICE, region=REGION, name="NetDoping",
        equation="Donors - Acceptors;",
    )


def initial_potential_solution() -> None:
    """Solve Poisson's equation with implicit (Boltzmann) carriers.

    This bootstraps the potential field before turning on the drift-diffusion
    continuity equations. Both contacts are pinned to 0 V for this stage.
    """
    CreateSolution(DEVICE, REGION, "Potential")
    CreateSiliconPotentialOnly(DEVICE, REGION)
    for contact in ("anode", "cathode"):
        set_parameter(device=DEVICE, name=GetContactBiasName(contact), value=0.0)
        CreateSiliconPotentialOnlyContact(DEVICE, REGION, contact)
    solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)


def initial_dd_solution() -> None:
    """Activate the drift-diffusion equations and resolve at equilibrium.

    Seeds ``Electrons`` and ``Holes`` from the equilibrium intrinsic
    densities, then solves the full Poisson + continuity system. This is
    the V = 0 starting point for the bias sweep.
    """
    CreateSolution(DEVICE, REGION, "Electrons")
    CreateSolution(DEVICE, REGION, "Holes")
    set_node_values(
        device=DEVICE, region=REGION, name="Electrons",
        init_from="IntrinsicElectrons",
    )
    set_node_values(
        device=DEVICE, region=REGION, name="Holes",
        init_from="IntrinsicHoles",
    )
    CreateSiliconDriftDiffusion(DEVICE, REGION)
    for contact in ("anode", "cathode"):
        CreateSiliconDriftDiffusionAtContact(DEVICE, REGION, contact)
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=50)


def voltage_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep anode voltage and collect carrier densities at each step.

    The cathode stays at 0 V. The anode is stepped from ``voltage_start`` to
    ``voltage_stop`` inclusive in ``voltage_interval`` increments. At each
    bias point the converged Electrons and Holes node arrays are pulled
    from DEVSIM and stored.

    Returns:
        A tuple ``(V, x, y, n, p)`` where ``V`` is shape ``(n_V,)``, ``x``
        and ``y`` are node coordinates shape ``(N,)``, and ``n``, ``p``
        carrier-density arrays are shape ``(n_V, N)``. Densities are in
        m^-3 (DEVSIM SI convention).
    """
    voltages = np.arange(
        p.voltage_start,
        p.voltage_stop + p.voltage_interval / 2,
        p.voltage_interval,
    )
    x = np.asarray(get_node_model_values(device=DEVICE, region=REGION, name="x"))
    y = np.asarray(get_node_model_values(device=DEVICE, region=REGION, name="y"))

    n_arr = np.empty((len(voltages), len(x)))
    p_arr = np.empty_like(n_arr)

    for i, V in enumerate(voltages):
        set_parameter(device=DEVICE, name=GetContactBiasName("anode"), value=float(V))
        solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
        n_arr[i] = np.asarray(
            get_node_model_values(device=DEVICE, region=REGION, name="Electrons")
        )
        p_arr[i] = np.asarray(
            get_node_model_values(device=DEVICE, region=REGION, name="Holes")
        )
        print(f"  V = {V:5.2f} V done")

    return voltages, x, y, n_arr, p_arr


def sample_to_grid(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    n_arr: np.ndarray,
    p_arr: np.ndarray,
    nx: int = 301,
    ny: int = 81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample carrier densities from the FEM mesh onto a regular grid.

    The downstream mode solver needs structured arrays to project a
    spatially-varying permittivity onto the femwell basis. We cover the full
    slab width in x and the full silicon stack in y (slab + rib).

    Linear interpolation is used; nodes outside the silicon polygon (the gap
    above the slab outside the rib, between the rib and the contacts) fall
    back to 0 — those points won't be in the silicon region of the mode
    solver anyway.

    Args:
        x_nodes, y_nodes: per-node coordinates from DEVSIM, shape ``(N,)``.
        n_arr, p_arr: carrier densities, shape ``(n_V, N)``.
        nx, ny: regular grid resolution.

    Returns:
        ``(x_grid, y_grid, n_grid, p_grid)`` where the grids are
        shape ``(n_V, nx, ny)``.
    """
    x_grid = np.linspace(-p.width_slab / 2, p.width_slab / 2, nx)
    y_grid = np.linspace(0.0, p.thick_slab + p.thick_rib, ny)
    XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    nodes = np.column_stack([x_nodes, y_nodes])

    n_grid = np.empty((n_arr.shape[0], nx, ny))
    p_grid = np.empty_like(n_grid)
    for i in range(n_arr.shape[0]):
        n_grid[i] = griddata(
            nodes, n_arr[i], pts, method="linear", fill_value=0.0
        ).reshape(nx, ny)
        p_grid[i] = griddata(
            nodes, p_arr[i], pts, method="linear", fill_value=0.0
        ).reshape(nx, ny)

    return x_grid, y_grid, n_grid, p_grid


def main() -> None:
    """End-to-end run: mesh, device setup, solve, sweep, save ``carriers.npz``."""
    print("Building gmsh mesh...")
    build_mesh(MSH_PATH)

    print("Setting up DEVSIM device...")
    setup_device()
    set_doping()

    print("Initial potential-only solve...")
    initial_potential_solution()

    print("Initial drift-diffusion solve...")
    initial_dd_solution()

    print("Voltage sweep...")
    V_arr, xn, yn, n_nodes, p_nodes = voltage_sweep()

    print("Resampling to regular grid...")
    x_grid, y_grid, n_grid, p_grid = sample_to_grid(xn, yn, n_nodes, p_nodes)

    print(f"Saving {NPZ_PATH}...")
    np.savez_compressed(
        NPZ_PATH,
        V=V_arr,
        x=x_grid,
        y=y_grid,
        n=n_grid,
        p=p_grid,
    )
    print(f"Done. {len(V_arr)} voltages, grid {n_grid.shape[1]} x {n_grid.shape[2]}.")


if __name__ == "__main__":
    main()
