# Python port of the Lumerical CHARGE + MODE pipeline

Open-source reimplementation of the two ANSYS Lumerical scripts that ship
with the SIEPIC Active frequency-discriminator supplemental document
(`../Charge_Simulation.lsf`, `../Mode_Simulation.lsf`). The Python pipeline
produces an equivalent `modulator_neff_V.dat` file that can be diffed
against a collaborator's Lumerical run for validation.

## What runs where

| Stage     | Original (Lumerical)        | This port (Python)                         |
| --------- | --------------------------- | ------------------------------------------ |
| Geometry  | `PN_Junction_Setup.lsf`     | `params.py`                                |
| CHARGE    | `Charge_Simulation.lsf`     | `charge_sim.py` (DEVSIM, drift-diffusion)  |
| MODE/FDE  | `Mode_Simulation.lsf`       | `mode_sim.py`  (femwell, FEM eigenmode)    |
| Soref-Bennett | `np density model`      | implemented inline in `mode_sim.py`        |

## Setup

### NixOS (recommended on this machine)

```bash
cd simulation_scripts/python
nix develop      # provides python 3.12 + openblas + gmsh
uv sync          # populates .venv with DEVSIM, femwell, etc.
```

DEVSIM dlopens BLAS/LAPACK at runtime; the devshell sets
`LD_LIBRARY_PATH` and `DEVSIM_MATH_LIBS` so it finds the nix-store
openblas. Without the devshell, `import devsim` will fail with
"MISSING DLL".

### Other Linux

Make sure `libopenblas.so` is on the loader path (or set
`DEVSIM_MATH_LIBS` and `LD_LIBRARY_PATH` manually), then:

```bash
cd simulation_scripts/python
uv sync
```

Either path pulls DEVSIM, femwell, scikit-fem, shapely, gmsh, numpy,
scipy, and matplotlib into a project-local `.venv/`.

## Run

```bash
uv run python charge_sim.py     # → carriers.npz + pin_mesh.msh
uv run python mode_sim.py       # → modulator_neff_V.dat + neff_vs_V.png
```

`charge_sim.py` takes the bulk of the wall time (drift-diffusion at 41
bias points). `mode_sim.py` is comparatively fast (one FEM eigenmode solve
per voltage).

## Outputs

- **`carriers.npz`** — `{V, x, y, n, p}`. Electron and hole densities (m⁻³)
  on a regular grid covering the silicon cross-section, one slice per
  voltage step. Analog of Lumerical's `PIN_Charge.mat`.
- **`modulator_neff_V.dat`** — three whitespace-separated columns:
  `V, Re(Δneff), Im(neff)`. Same layout the LSF script writes. **This is
  the artifact to ship to the collaborator for diffing.**
- **`neff_vs_V.png`** — the two plots the LSF generates (relative phase
  per cm, loss per cm) for quick visual inspection.

## Comparison procedure with a Lumerical run

1. Send `../PN_Junction_Setup.lsf`, `../Charge_Simulation.lsf`,
   `../Mode_Simulation.lsf` to the collaborator.
2. Have them run the LSF pipeline and return their `modulator_neff_V.dat`.
3. Diff column-by-column. Target tolerances:
   - `Re(Δneff)`: <5 % relative error
   - `Im(neff)`:  <10 % relative error

Larger discrepancies usually trace back to one of these three places:
mesh density in the rib, the assumed p-epi background concentration
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
- **Soref-Bennett coefficients** — using the original 1987 values
  (the same set Lumerical's "Soref and Bennet" option applies). For more
  recent calibrations see Nedeljkovic et al. 2011.
