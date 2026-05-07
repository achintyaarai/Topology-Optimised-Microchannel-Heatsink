"""
fd_check.py -- Finite-difference gradient verification
========================================================
Verifies the adjoint sensitivities (JQ, Jf, JTV) against central
finite differences on a small test mesh.
"""

import numpy as np
import time

from config import SimConfig, GeoConfig
from mesh_geometry import MeshDomain
from physics_solver import PhysicsSolver
from optimizer import Optimizer


def run_fd_check():
    # Small mesh for speed
    geo = GeoConfig(nx=30, ny=15)
    cfg = SimConfig(max_opt_iter=1)
    mesh = MeshDomain(geo)
    solver = PhysicsSolver(cfg, mesh)
    opt = Optimizer(cfg, mesh, solver)

    # Base design field (uniform gray)
    xi_base = mesh.initial_design_field()
    xi_base[mesh.design_elements] = 0.5

    # Forward solve
    U, P, T = solver.solve_forward(xi_base)

    # Adjoint sensitivities (now returns 6 values)
    dJQ_full, dJf_full, dJTV_full, JQ_val, Jf_val, JTV_val = \
        opt.compute_sensitivities(xi_base, U, T)

    print(f"Baseline: JQ = {JQ_val:.6e}, Jf = {Jf_val:.6e}, JTV = {JTV_val:.6e}")
    print()

    # Pick design elements spread across the domain
    de = mesh.design_elements
    test_indices = [0, len(de) // 4, len(de) // 2, 3 * len(de) // 4, len(de) - 1]
    test_elems = [de[i] for i in test_indices]

    eps = 1e-6
    print(f"{'Elem':>6s}  {'Obj':>4s}  {'FD':>12s}  {'Adj':>12s}  {'RelErr':>8s}")
    print("-" * 52)

    for elem_id in test_elems:
        # Forward perturbation
        xi_p = xi_base.copy()
        xi_p[elem_id] += eps
        _, _, T_p = solver.solve_forward(xi_p)
        U_p, P_p, T_p = solver.solve_forward(xi_p)

        # Backward perturbation (central FD)
        xi_m = xi_base.copy()
        xi_m[elem_id] -= eps
        U_m, P_m, T_m = solver.solve_forward(xi_m)

        # Compute objectives at both perturbations
        _, _, _, JQ_p, Jf_p, JTV_p = opt.compute_sensitivities(xi_p, U_p, T_p)
        _, _, _, JQ_m, Jf_m, JTV_m = opt.compute_sensitivities(xi_m, U_m, T_m)

        for name, fd_p, fd_m, adj_val in [
            ("JQ",  JQ_p,  JQ_m,  dJQ_full[elem_id]),
            ("Jf",  Jf_p,  Jf_m,  dJf_full[elem_id]),
            ("JTV", JTV_p, JTV_m, dJTV_full[elem_id]),
        ]:
            fd_sens = (fd_p - fd_m) / (2.0 * eps)
            rel_err = abs(adj_val - fd_sens) / (abs(fd_sens) + 1e-12)
            print(f"{elem_id:>6d}  {name:>4s}  {fd_sens:>12.4e}  {adj_val:>12.4e}  {rel_err:>7.2%}")

    print("\nDone.")


if __name__ == '__main__':
    run_fd_check()