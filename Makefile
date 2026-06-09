.PHONY: charge mode viz sim

# DEVSIM drift-diffusion solve — needs FHS env for MKL dlopen
charge:
	nix run . -- -c "uv run src/charge_sim.py"

# femwell mode solve — also needs FHS env (libstdc++ from pip wheels)
mode:
	nix run . -- -c "uv run src/mode_sim.py"

# Carrier density maps (reads carriers.npz, no solver needed)
viz:
	nix run . -- -c "uv run src/visualize.py"

# Full pipeline: carriers first, then mode solve, then visualize
sim: charge mode viz
