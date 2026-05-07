# Topology Optimization of 2D Extruded Microchannel Heatsinks

**Team:** Divya Nayan Mehta (23B0001) & Achintya Aravind Rai (23B0069)
**Course:** ME701 Final Project

## Project Overview

This project presents a robust finite element-based Topology Optimization framework designed to synthesize organic, fractal-like branching microchannels for high-power electronics thermal management. It utilizes an $\epsilon$-constraint formulation to minimize peak thermal hotspots while strictly controlling flow dissipation and ensuring uniform cooling across the substrate.

## Directory Structure

* `1_Report/`: Contains the final IEEE-format LaTeX report and its compiled PDF.
* `2_Presentation/`: Contains the presentation slides (`.pdf`).
* `3_Code/`: Contains the Python source code, required dependencies, and run instructions.
* `4_Data_Results/`: Directory for output metrics, CSV logs, figures, and exported DXF contours.
* `5_Literature/`: Summaries of the core literature defining the mathematical foundations.
* `6_Method_Trace/`: Logs detailing our design decisions, failed attempts, and LLM usage.
* `7_Reproducibility/`: Step-by-step instructions and repository links to recreate our Pareto front.

## Instructions to Run the Code

Please see `3_Code/run_instructions.md` and `7_Reproducibility/REPRODUCE.md` for complete setup, execution, and plotting instructions.

## Location of Main Results

Running the optimizer automatically dumps the following to the root/results directory:

1. `sweep_results.csv`: Contains the logged objectives ($J_Q, J_f, J_{TV}$) and constraint violations.
2. `optimization_results.png`: A 3-panel visualization of density, velocity, and temperature fields.
3. `microchannel_design.dxf`: A CAD-ready contour extraction of the final topology.