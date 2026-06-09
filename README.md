# Frequency Discriminator VOA Simulation

A Python port of a Lumerical CHARGE + MODE pipeline that models a
silicon-on-insulator (SOI) PIN optical modulator in cross-section.

## Overview

The simulated cross-section is a 220 nm SOI rib waveguide (500 nm wide, a 130 nm
rib on a 90 nm slab) with a lateral PIN junction: p++ and n++ implants
($4 \times 10^{20}\ \mathrm{cm}^{-3}$ surface concentration) about 2 um to
either side of the rib, a lightly-doped p-epi region in the middle, and metal
contacts on the slab outboard of the implants. The wavelength is 1550 nm. The
anode (p side) is swept from $0$ to $4$ V in 0.1 V steps while the cathode (n
side) is grounded, which forward-biases the junction and injects free carriers
into the rib.

## Repo Structure

1. `src/params.py` geometry, doping levels, and voltage sweep. Values are in SI
   units.

2. `src/charge_sim.py` builds a mesh of the cross-section and runs a DEVSIM
   drift-diffusion simulation at each bias point, sweeping the applied voltage
   from $0$ to $4$ V. Drift-diffusion is the standard physics model for how
   electrons and holes move through a semiconductor under an applied voltage.
   The resulting carrier densities are written to `src/carriers.npz`.

3. `src/mode_sim.py` reads those carrier densities, converts them into a local
   change in refractive index via the carrier-to-index relations, and then
   solves for the shape and speed of the guided light mode using the femwell
   finite-element solver. It does this at every voltage and writes
   `src/modulator_neff_V.dat` along with a `neff_vs_V.png` plot for a quick
   visual check.

4. `src/visualize.py` reads `carriers.npz` and writes `src/carrier_maps.png`

   > electron density across the slab and the injected-carrier change near the
   > rib, at four voltages.

5. `src/compare_lumerical.py` overlays `src/modulator_neff_V.dat` against a
   Lumerical export (`lumerical_data.dat` at the repo root by default, or a path
   passed as the first argument), prints a side-by-side table, and writes
   `src/compare_neff.png`.

### Nix

With direnv configured, `cd` into the repo and the shell loads automatically.
Otherwise:

```bash
nix develop
uv sync
```

### Linux

Install Nix (Determinate installer):

```bash
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```

Enable flakes:

```bash
mkdir -p ~/.config/nix/
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
```

Install direnv through Nix (the apt package is too old to support `use flake`):

```bash
nix profile add nixpkgs#direnv
```

Add the shell hook and reload:

```bash
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
exec bash
```

Then allow direnv in the project (this runs `nix develop` for you on entry):

```bash
direnv allow
uv sync
```

### Windows

Use WSL2 with Ubuntu and follow the Linux steps inside it; the Nix and DEVSIM
toolchain has no native-Windows path.

```powershell
wsl --install -d Ubuntu-24.04
```

## Run

```bash
uv run src/charge_sim.py              # carriers.npz
uv run src/mode_sim.py                # modulator_neff_V.dat, neff_vs_V.png
uv run src/visualize.py               # carrier_maps.png
uv run src/compare_lumerical.py       # compare_neff.png (needs lumerical_data.dat)
```

`make sim` runs the charge solve, mode solve, and visualize in order; the
`charge`, `mode`, and `viz` targets run them individually (each wraps the
command in `nix run` so it gets the FHS shell).

## Output

`src/modulator_neff_V.dat` holds three whitespace-separated columns: the bias
voltage, the real part of the change in effective index
$\mathrm{Re}(\Delta n_\text{eff})$ (which governs the phase shift), and the
imaginary part $\mathrm{Im}(n_\text{eff})$ (which governs optical loss). Compare
against the Lumerical run.

## Limitations

- The DEVSIM solver ships as a prebuilt binary that expects to find Intel's MKL
  math library. The typically bundled math package OpenBLAS doesn't work.

- DEVSIM's built-in silicon physics is written in centimetre-gram-second units,
  while the rest of the project works in SI.

- Smaller compatibility constraints: DEVSIM only reads the legacy mesh file
  format, and one of the plotting dependencies has no build for the newest
  Python, which pins the supported interpreter range.

- **Background doping.** The original script applies a lightly doped silicon
  background without stating its concentration. This port assumes
  $1 \times 10^{15}\ \mathrm{cm}^{-3}$. The equilibrium carrier baseline, and
  therefore the magnitude of $\Delta n_\text{eff}$, scales with this choice.

- **Implant profile.** The doping that defines the junction is modeled as a
  lateral step with a Gaussian falloff in depth.
