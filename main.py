"""
main.py — Microchannel Heatsink Topology Optimizer Entry Point
================================================================
Parses CLI arguments, constructs the mesh/solver/optimizer pipeline,
runs the ε-constraint optimisation, and delegates post-processing.
"""

import argparse
import numpy as np
import warnings
from scipy.sparse import SparseEfficiencyWarning

# Suppress annoying scipy sparse matrix formatting warnings globally
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from config import SimConfig, GeoConfig
from mesh_geometry import build_default_mesh, MeshDomain
from physics_solver import PhysicsSolver
from optimizer import Optimizer
from postprocessing import postprocess_design


def parse_args():
    """Parses command-line arguments for the optimization run."""
    parser = argparse.ArgumentParser(description="Microchannel Heatsink Topology Optimizer")

    # Grid resolution
    parser.add_argument("--nelx", type=int, default=160, help="Number of elements in x-direction")
    parser.add_argument("--nely", type=int, default=80, help="Number of elements in y-direction")

    # Target SOLID volume fraction
    parser.add_argument("--volfrac", type=float, default=0.4,
                        help="Target solid volume fraction constraint (e.g. 0.4 = 40%% solid)")

    # ε-constraint parameters (Wang et al. 2025)
    parser.add_argument("--eps_flow", type=float, default=1.5,
                        help="Flow dissipation constraint multiplier (vs initial baseline)")
    parser.add_argument("--eps_tv", type=float, default=2.0,
                        help="Temperature variance constraint multiplier (vs initial baseline)")
    parser.add_argument("--w_blob", type=float, default=0.1,
                        help="Solid agglomeration 'blob' penalty weight")

    return parser.parse_args()


def main():
    # 1. Parse CLI arguments
    args = parse_args()
    print(f"--- Initializing Microchannel Topology Optimizer ---")
    print(f"Grid Resolution: {args.nelx} x {args.nely}")
    print(f"Target Volume Fraction: {args.volfrac}")
    print(f"eps-flow: {args.eps_flow}, eps-TV: {args.eps_tv}, w_blob: {args.w_blob}")

    # 2. Instantiate Configurations
    sim_config = SimConfig(
        volume_fraction=args.volfrac,
        eps_flow=args.eps_flow,
        eps_tv=args.eps_tv,
        w_blob=args.w_blob,
    )

    # GeoConfig geometry
    geo_config = GeoConfig(nx=args.nelx, ny=args.nely)

    # 3. Generate Mesh
    print("Generating finite element mesh and boundaries...")
    mesh = MeshDomain(geo_config)

    # 4. Initialize Physics Solver
    print("Pre-allocating physics solver matrices...")
    solver = PhysicsSolver(sim_config, mesh)

    # 5. Initialize & Run Optimizer
    print("Booting MMA Optimizer (eps-constraint)...")
    opt = Optimizer(sim_config, mesh, solver)

    # Execute optimization loop
    x_opt = opt.optimize()

    # 6. Final Post-Processing
    postprocess_design(mesh, solver, opt, x_opt, args.volfrac)

    print("--- Optimization Pipeline Complete ---")


if __name__ == "__main__":
    main()