"""
postprocessing.py — Post-processing for ε-constraint microchannel optimizer
============================================================================
Handles binarisation, final forward solve, metric extraction, CSV logging,
plot generation, and DXF contour export.

Compatible with the Wang et al. (2025) ε-constraint formulation:
  - Tracks JQ (p-norm), Jf (flow dissipation), JTV (temperature variance)
  - Reports ε-flow, ε-TV, w_blob parameters in CSV output
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mesh_geometry import MeshDomain
from physics_solver import PhysicsSolver
from optimizer import Optimizer


def compute_objectives(opt: Optimizer, xi_phys: np.ndarray,
                       velocity: np.ndarray, T: np.ndarray):
    """
    Lightweight forward-only evaluation of JQ, Jf, and JTV.
    Uses the same math as compute_sensitivities for consistency.
    """
    n_node = opt.mesh.n_node
    n_elem = opt.mesh.n_elem

    # P-norm thermal objective
    p_val = opt.cfg.p_norm_exponent
    T_clipped = np.maximum(T, opt.cfg.t_clip_floor)
    JQ_val = np.mean(T_clipped ** p_val) ** (1.0 / p_val)

    # Temperature Variance
    T_mean = np.mean(T)
    V_domain = opt.mesh.geo.L_total * opt.mesh.geo.H
    JTV_val = (1.0 / V_domain) * (T - T_mean).T @ opt.solver.M_mass @ (T - T_mean)

    # Fluid dissipation (physical viscous part only — no Brinkman penalty)
    U_e = np.zeros((n_elem, 8))
    for i in range(8):
        U_e[:, i] = velocity[opt.mesh.edofMat_vel[:, i]]

    val_visc = np.einsum('ni,ij,nj->n', U_e, opt.solver.ke_visc, U_e)
    Jf_val = 0.5 * np.sum(opt.solver.mu * val_visc)

    return JQ_val, Jf_val, JTV_val


def postprocess_design(mesh: MeshDomain, solver: PhysicsSolver, opt: Optimizer,
                       x_opt: np.ndarray, volfrac: float):
    """
    Filter raw variables, apply threshold binarization, final forward solve,
    extract metrics, write CSV, generate plots, and export DXF contours.
    """
    # 1. Recover the true physical field through filter + projection
    Da, beta = opt.get_continuation_params(opt.iteration)
    solver.alpha_max = 1.0 / Da
    x_design_full = opt.expand_to_full(x_opt)
    x_tilde = opt.H_filter.dot(x_design_full)
    x_phys, _ = opt.apply_projection(x_tilde, beta)

    # 2. Sweep threshold to find the value closest to target Vf
    #    x_phys=1 is fluid, x_phys=0 is solid. solid_frac = mean(1 - binarized).
    best_eta = 0.5
    best_err = float('inf')
    for eta in np.linspace(0.01, 0.99, 200):
        x_bin = np.where(x_phys >= eta, 1.0, 0.0)
        solid_frac = 1.0 - np.mean(x_bin)
        err = abs(solid_frac - volfrac)
        if err < best_err:
            best_err = err
            best_eta = eta

    x_opt_hard = np.where(x_phys >= best_eta, 1.0, 0.0)
    actual_solid = 1.0 - np.mean(x_opt_hard)
    print(f"Binarization: eta={best_eta:.3f}, solid_frac={actual_solid:.3f} (target={volfrac:.3f})")

    # Full-Domain Field Reconstruction
    x_opt_full = mesh.initial_design_field()
    x_opt_full[mesh.design_elements] = x_opt_hard

    # 3. Final Forward Solve
    print("Performing final physics solve for plotting...")
    velocity, P_final, T_final = solver.solve_forward(x_opt_full)
    U_final = velocity[0::2]
    V_final = velocity[1::2]

    # 4. Lightweight objective evaluation
    JQ, Jf, JTV = compute_objectives(opt, x_opt_full, velocity, T_final)

    P_inlet = np.mean(P_final[mesh.bc.inlet_nodes_fluid])
    P_outlet = np.mean(P_final[mesh.bc.outlet_nodes_fluid])
    pressure_drop = P_inlet - P_outlet

    mean_temp = np.mean(T_final)
    peak_temp = np.max(T_final)

    # Stagnation Volume Ratio
    fluid_elem_indices = np.where(x_opt_full > 0.5)[0]
    if len(fluid_elem_indices) > 0:
        fluid_nodes = np.unique(mesh.edofMat_th[fluid_elem_indices].ravel())
        mean_inlet_velocity = np.mean(mesh.bc.inlet_u_profile)
        vel_mag = np.sqrt(U_final ** 2 + V_final ** 2)
        vel_mag_fluid = vel_mag[fluid_nodes]
        if len(vel_mag_fluid) > 0:
            stag_nodes = np.sum(vel_mag_fluid < 0.05 * mean_inlet_velocity)
            stag_ratio = stag_nodes / len(vel_mag_fluid)
        else:
            stag_ratio = 1.0
    else:
        stag_ratio = 1.0

    print("\n" + "=" * 50)
    print("FINAL DESIGN METRICS")
    print(f"Peak Temperature  : {peak_temp:.4e} K")
    print(f"Average Temp      : {mean_temp:.4e} K")
    print(f"Pressure Drop     : {pressure_drop:.4e} Pa")
    print(f"JQ (p-norm)       : {JQ:.4e}")
    print(f"Jf (viscous)      : {Jf:.4e}")
    print(f"JTV (variance)    : {JTV:.4e}")
    print(f"Stagnation ratio  : {stag_ratio:.4f}")
    print("=" * 50 + "\n")

    # 5. CSV Append
    csv_filename = "sweep_results.csv"
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "volfrac", "pressure_drop", "mean_temp", "peak_temp",
                "stagnation_ratio", "JQ", "Jf", "JTV",
                "eps_flow", "eps_tv", "w_blob"
            ])
        writer.writerow([
            actual_solid, pressure_drop, mean_temp, peak_temp,
            stag_ratio, JQ, Jf, JTV,
            opt.cfg.eps_flow, opt.cfg.eps_tv, opt.cfg.w_blob,
        ])

    # 6. Visual Output
    suffix = f"_v{volfrac}_ef{opt.cfg.eps_flow}_et{opt.cfg.eps_tv}"
    img_name = f"optimization_results{suffix}.png"
    dxf_name = f"microchannel_design{suffix}.dxf"

    plot_results(mesh, x_opt_full, U_final, V_final, T_final, filename=img_name)
    export_dxf(mesh, x_opt_full, filename=dxf_name)

    return x_opt_full


def plot_results(mesh, x_opt_full, U, V, T, filename):
    """Generates and saves a clean 3-panel plot of the density / velocity / temperature."""
    print("Generating post-processing plots...")
    nx = mesh.geo.nx
    ny = mesh.geo.ny

    # Element density field
    Density_2D = x_opt_full.reshape(ny, nx)

    # Nodal fields
    Vel_mag_2D = np.sqrt(U ** 2 + V ** 2).reshape(ny + 1, nx + 1)
    Temp_2D = T.reshape(ny + 1, nx + 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # 1. Density
    ax1 = axes[0]
    X_elem_grid = np.linspace(0, mesh.geo.L_total, nx + 1)
    Y_elem_grid = np.linspace(0, mesh.geo.H, ny + 1)
    im1 = ax1.pcolormesh(X_elem_grid, Y_elem_grid, Density_2D,
                         cmap='Greys', shading='flat', vmin=0, vmax=1)
    ax1.set_title(r"Final Density Distribution ($\xi$)")
    ax1.set_aspect('equal')
    fig.colorbar(im1, ax=ax1, label='Fluid Fraction (1 = fluid, 0 = solid)')

    # 2. Velocity
    ax2 = axes[1]
    X_node = mesh.node_x.reshape(ny + 1, nx + 1)
    Y_node = mesh.node_y.reshape(ny + 1, nx + 1)
    vmax_vel = np.percentile(Vel_mag_2D, 99)
    im2 = ax2.pcolormesh(X_node, Y_node, Vel_mag_2D,
                         cmap='viridis', shading='gouraud', vmax=vmax_vel)
    ax2.set_title("Velocity Magnitude Field")
    ax2.set_aspect('equal')
    fig.colorbar(im2, ax=ax2, label='Velocity [m/s]')

    # 3. Temperature
    ax3 = axes[2]
    im3 = ax3.pcolormesh(X_node, Y_node, Temp_2D,
                         cmap='inferno', shading='gouraud')
    ax3.set_title(r"Temperature Field ($\Delta T$)")
    ax3.set_aspect('equal')
    ax3.set_xlabel("X-Coordinate [m]")
    fig.colorbar(im3, ax=ax3, label=r'Temperature Rise $\Delta T$ [K]')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Results plot saved successfully to {filename}")


def export_dxf(mesh, x_opt_full, filename):
    """
    Extract xi=0.5 contour and export as lightweight DXF for CAD import.
    Uses matplotlib's contour engine.
    """
    print("Extracting geometric contours for CAD export...")
    nx = mesh.geo.nx
    ny = mesh.geo.ny

    # Element centers
    X_elem = mesh.elem_cx.reshape(ny, nx)
    Y_elem = mesh.elem_cy.reshape(ny, nx)

    # Mask non-design elements
    x_design = x_opt_full.copy()
    mask = np.ones(mesh.n_elem, dtype=bool)
    mask[mesh.design_elements] = False
    x_design[mask] = 0.0
    Density_2D = x_design.reshape(ny, nx)

    fig, ax = plt.subplots()
    cs = ax.contour(X_elem, Y_elem, Density_2D, levels=[0.5])

    with open(filename, 'w') as f:
        f.write("0\nSECTION\n2\nENTITIES\n")
        for collection in cs.collections:
            for path in collection.get_paths():
                v = path.vertices
                for i in range(len(v) - 1):
                    f.write("0\nLINE\n8\nMicrochannel_Contour\n")
                    f.write(f"10\n{v[i, 0]:.6f}\n20\n{v[i, 1]:.6f}\n30\n0.0\n")
                    f.write(f"11\n{v[i + 1, 0]:.6f}\n21\n{v[i + 1, 1]:.6f}\n31\n0.0\n")
        f.write("0\nENDSEC\n0\nEOF\n")

    plt.close(fig)
    print(f"CAD contour DXF exported successfully to {filename}")