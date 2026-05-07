# Reproducing the Optimization Results

**Repository Link:** Github - `https://github.com/achintyaarai/Topology-Optimised-Microchannel-Heatsink`
                     Zenodo - 'https://doi.org/10.5281/zenodo.20073583'

This file contains the instructions needed to fully replicate the results, plots, and CSV data shown in the project presentation.

## Step 1: Environment Setup
Ensure you are running inside a virtual environment using Python 3.10+.
Install the required numerical packages:

```bash
pip install -r requirements.txt
```

## Step 2: Validate the Gradients (Optional but recommended)
Before running a massive optimization loop, ensure that the continuous adjoint math matches raw physical reality by running the finite-difference check on a small 30x15 mesh.
```bash
python fd_check.py
```
**Expected Output:** A printed table showing the gradient error (FD vs Adj) for $J_Q$, $J_f$, and $J_{TV}$ across 5 sample elements. The error should strictly be below 1.5%.

## Step 3: Run a Single Baseline Optimization
To generate the primary design highlighted in the presentation (40% solid volume fraction with strict fluid parameters):
```bash
python main.py --nelx 160 --nely 80 --volfrac 0.4 --eps_flow 1.5 --eps_tv 2.0
```
**Expected Output:**
1. The solver will output iteration statistics (`J`, `Da`, `beta`, `Vf_solid`).
2. After ~150-250 iterations, it will converge. 
3. It will generate `optimization_results.png` (the 3-panel velocity/temperature/density plot) and `microchannel_design.dxf` in your working directory.

## Step 4: Recreate the Pareto Front Sweep
To automatically run the optimizer across multiple configurations and generate the 4-panel scatter plot matrix presented in the report:
```bash
python run_sweep.py
```
**Expected Output:**
The script will queue up a parameter sweep using `subprocess`. Be aware that this can take several hours depending on CPU hardware. Once complete, it will read the generated `sweep_results.csv` and output `parameter_sweep_metrics.png`.
