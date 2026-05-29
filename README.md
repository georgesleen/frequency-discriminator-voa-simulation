# Python port of the Lumerical CHARGE + MODE pipeline

Open-source reimplementation of the two ANSYS Lumerical scripts that ship
with the SIEPIC Active frequency-discriminator supplemental document
(`../Charge_Simulation.lsf`, `../Mode_Simulation.lsf`). Emits the same
`modulator_neff_V.dat` for direct comparison against a collaborator's
Lumerical run.

## What runs where

| Stage         | Original (Lumerical)    | This port (Python)                         |
| ------------- | ----------------------- | ------------------------------------------ |
| Geometry      | `PN_Junction_Setup.lsf` | `params.py`                                |
| CHARGE        | `Charge_Simulation.lsf` | `charge_sim.py` (DEVSIM, drift-diffusion)  |
| MODE/FDE      | `Mode_Simulation.lsf`   | `mode_sim.py`  (femwell, FEM eigenmode)    |
| Soref-Bennett | `np density model`      | inline in `mode_sim.py`                    |

## Setup

### NixOS

```bash
nix develop        # drops you into an FHS bash with mkl, gmsh, uv
uv sync            # populates .venv with DEVSIM, femwell, etc.
```

The devshell is `pkgs.buildFHSEnv`, not `mkShell`. DEVSIM's pip wheel is
manylinux and dlopens `libmkl_rt.so` at runtime; FHS is the cleanest way to
give it `/usr/lib/libmkl_rt.so` without `LD_LIBRARY_PATH` gymnastics. MKL
is unfree (Intel SSL) — the flake narrows `allowUnfreePredicate` to that
one package.

### Other Linux

DEVSIM officially supports AlmaLinux 8 / RHEL 8 with MKL. On other
distros, follow DEVSIM's INSTALL.md: install MKL (`conda install mkl` or
`pip install mkl`), then `uv sync`. Do **not** point `DEVSIM_MATH_LIBS` at
OpenBLAS — DEVSIM's bundled UMFPACK miscalls OpenBLAS's `DGER` and
segfaults during the Newton solve.

## Run

```bash
# inside `nix develop`
uv run src/charge_sim.py     # → src/carriers.npz   (+ pin_mesh.msh)
uv run src/mode_sim.py       # → src/modulator_neff_V.dat + neff_vs_V.png

# or one-shot from outside the shell
nix run . -- -c "uv run src/charge_sim.py"
```

`charge_sim.py` takes the bulk of the wall time (41-point voltage sweep,
drift-diffusion at each). `mode_sim.py` is comparatively fast (one FEM
eigenmode solve per voltage).

## Outputs

- **`carriers.npz`** — `{V, x, y, n, p}`. Electron and hole densities (m⁻³)
  on a regular grid covering the silicon cross-section, one slice per
  voltage step. Analog of Lumerical's `PIN_Charge.mat`.
- **`modulator_neff_V.dat`** — three whitespace-separated columns:
  `V, Re(Δneff), Im(neff)`. Same layout the LSF emits. **This is the
  artifact to ship to the collaborator for diffing.**
- **`neff_vs_V.png`** — relative phase per cm + loss per cm, for quick
  visual inspection.

## Comparison procedure

1. Send `../PN_Junction_Setup.lsf`, `../Charge_Simulation.lsf`,
   `../Mode_Simulation.lsf` to the collaborator.
2. Have them run the LSF pipeline and return their `modulator_neff_V.dat`.
3. Diff column-by-column. Target tolerances:
   - `Re(Δneff)`: <5 % relative error
   - `Im(neff)`:  <10 % relative error

Larger discrepancies usually trace back to one of three places: mesh
density in the rib, the assumed p-epi background concentration
(`PEPI_CONC` in `charge_sim.py` — the LSF leaves it implicit), or the
implant Gaussian vertical sigma.

## Known caveats

- **p-epi concentration assumption** — the LSF's `adddope; ...; pepi` call
  does not specify a value. `charge_sim.py` assumes 1 × 10¹⁵ cm⁻³. If
  Lumerical's default differs, the equilibrium carrier baseline will be
  off and Δneff will scale with it.
- **Implant profile** — modelled as a lateral step × vertical Gaussian
  with σ = `thick_slab / 3`. The LSF uses Lumerical's built-in implant
  Gaussian with `junction_width = 0`; the exact internal sigma is not
  documented in the supplemental.
- **Soref-Bennett coefficients** — using the original 1987 values (the
  same set Lumerical's "Soref and Bennet" option applies). For more recent
  calibrations see Nedeljkovic et al. 2011.
