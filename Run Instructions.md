# Running the Microchannel Topology Optimizer

This directory contains the source code for the Topology Optimization framework.

## Setup
1. Ensure Python 3.10+ is installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Repository Structure (Expected inside `src/`)
* `config.py`: Centralizes physical parameters (Water/Aluminium properties), dimensions, and optimizer hyperparameters.
* `mesh_geometry.py`: Handles Q4 structured mesh generation, element classification, and boundary condition assembly.
* `physics_solver.py`: Contains the pre-allocated, in-place CSR finite element solver (Stokes-Darcy & Advection-Diffusion).
* `optimizer.py`: Wraps the MMA optimizer, objective tracking, filtering, and the coupled continuous adjoint method.
* `postprocessing.py`: Handles plotting, metric extraction, and CAD (DXF) generation.
* `main.py`: The primary entry point.
* `fd_check.py` / `debug_run.py`: Diagnostic tools.
* `run_sweep.py`: Automation script to generate the Pareto front.

## Execution
To run a single optimization case (e.g., 40% solid volume fraction with default constraints):
```bash
python src/main.py --nelx 160 --nely 80 --volfrac 0.4 --eps_flow 1.5 --eps_tv 2.0
```

To run diagnostic checks to verify adjoint gradients against finite differences:
```bash
python src/fd_check.py