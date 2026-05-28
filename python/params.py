"""PIN-junction modulator geometry and doping parameters.

Direct port of ``simulation_scripts/PN_Junction_Setup.lsf``. Variable names
match the LSF script line-for-line so the two sources can be diffed side by
side when validating against the Lumerical pipeline.

Units: SI (metres, cubic metres, volts). Concentration values that the LSF
stores in cm^-3 are converted to m^-3 here (the ``* 1e6`` factor) so they can
be fed to DEVSIM, which works in SI.

Coordinate convention:
    x — lateral, across the junction (anode at -x, cathode at +x)
    z — vertical, slab grows upward from z = 0
    y — propagation direction (irrelevant for the 2D cross-section solve)
"""

# --- wafer and waveguide structure (220 nm SOI rib) -----------------------
thick_rib = 130e-9      # rib thickness above the slab
width_rib = 0.5e-6      # rib width
thick_slab = 90e-9      # slab thickness (130 + 90 = 220 nm full SOI stack)
width_slab = 6.5e-6     # total slab width (also the simulation x-span)

# --- doping ---------------------------------------------------------------
# p-epi background (lightly-doped intrinsic region between the implants)
center_pepi = 0.1e-6
thick_pepi = 0.3e-6

# p++ implant (anode side)
x_center_p = -2e-6
x_span_p = 2.5e-6
diff_dist_fcn = 1            # 0 = erfc, 1 = Gaussian (matches LSF)
face_p = 5                   # implant face: upper z (top of slab)
width_junction_p = 0
surface_conc_p = 4e20 * 1e6  # cm^-3 -> m^-3
reference_conc_p = 1e6 * 1e6

# n++ implant (cathode side)
x_center_n = 2e-6
x_span_n = 2.5e-6
face_n = 5
width_junction_n = 0
surface_conc_n = 4e20 * 1e6
reference_conc_n = 1e6 * 1e6

# --- metal contacts -------------------------------------------------------
center_contact = 2e-6
width_contact = 1.5e-6
thick_contact = 0.7e-6

# --- voltage sweep --------------------------------------------------------
voltage_start = 0.0
voltage_stop = 4.0
voltage_interval = 0.1

# --- mesh sizing ----------------------------------------------------------
# These reproduce the LSF mesh override settings. The 7 nm override applied
# in the rib is what actually controls accuracy near the depletion edge.
min_edge_length = 0.004e-6
max_edge_length = 0.6e-6
max_edge_length_override = 0.007e-6

# --- simulation region ----------------------------------------------------
x_center = 0.0
x_span = width_slab
y_center = 0.0
y_span = 1e-6        # unused in 2D cross-section
z_center = 0.0
z_span = 6e-6

# --- physical constants used downstream -----------------------------------
WAVELENGTH = 1550e-9             # operating wavelength
N_SI = 3.476                     # silicon index at 1550 nm (Palik / SiP standard)
N_SIO2 = 1.444                   # oxide cladding index at 1550 nm
TEMPERATURE_K = 300.0            # solver temperature
