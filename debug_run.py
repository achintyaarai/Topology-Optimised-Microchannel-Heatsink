"""Quick diagnostic: run a few iterations and print timing per solver call."""
import sys, time, numpy as np
from config import SimConfig, GeoConfig
from mesh_geometry import MeshDomain
from physics_solver import PhysicsSolver
from optimizer import Optimizer

cfg = SimConfig(volume_fraction=0.4)
geo = GeoConfig(nx=120, ny=60)
mesh = MeshDomain(geo)
solver = PhysicsSolver(cfg, mesh)
opt = Optimizer(cfg, mesh, solver)

# Simulate what the optimizer does iteration by iteration
rng = np.random.default_rng(42)
target_fluid = cfg.volume_fraction
noise = 0.05 * (rng.random(opt.n_var) - 0.5)
x_raw = np.clip(target_fluid + noise, 0.01, 0.99)

for it in range(1, 25):
    Da, beta = opt.get_continuation_params(it)
    solver.alpha_max = 1.0 / Da

    x_tilde = opt.H_filter.dot(x_raw)
    x_phys, d_phys = opt.apply_projection(x_tilde, beta)

    xi_full = mesh.initial_design_field()
    xi_full[mesh.design_elements] = x_phys

    print(f"Iter {it:3d}: alpha_max={solver.alpha_max:.2e}, beta={beta:.1f}, "
          f"x_phys=[{x_phys.min():.3f}, {x_phys.max():.3f}]", end="", flush=True)

    t0 = time.perf_counter()
    try:
        U, P, T = solver.solve_forward(xi_full)
        dt = time.perf_counter() - t0
        print(f"  solve={dt:.2f}s  T=[{T.min():.3e}, {T.max():.3e}]", flush=True)
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"  FAILED after {dt:.2f}s: {e}", flush=True)
        break

    if dt > 10:
        print("  *** STALL DETECTED -- solve took >10s ***", flush=True)
        break

print("Diagnostic complete.")