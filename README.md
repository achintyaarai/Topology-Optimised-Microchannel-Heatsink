# Topology Optimization of 2D Extruded Microchannel Heatsinks

**Team:** Divya Nayan Mehta (23B0001) & Achintya Aravind Rai (23B0069)
**Course:** ME701 Final Project

## Project Overview

This project presents a robust finite element-based Topology Optimization framework designed to synthesize organic, fractal-like branching microchannels for high-power electronics thermal management. It utilizes an $\epsilon$-constraint formulation to minimize peak thermal hotspots while strictly controlling flow dissipation and ensuring uniform cooling across the substrate.

## Instructions to Run the Code

Please see `run_instructions.md` and `REPRODUCE.md` for complete setup, execution, and plotting instructions.

## Location of Main Results

Running the optimizer automatically dumps the following to the root/results directory:

1. `sweep_results.csv`: Contains the logged objectives ($J_Q, J_f, J_{TV}$) and constraint violations.
2. `optimization_results.png`: A 3-panel visualization of density, velocity, and temperature fields.
3. `microchannel_design.dxf`: A CAD-ready contour extraction of the final topology.
