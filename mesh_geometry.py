"""
mesh_geometry.py — Microchannel Heatsink Topology Optimizer
============================================================
Structured Q4 mesh generation, DOF assembly, element-type spatial
classification, and boundary-condition packaging for the
pipe ↔ funnel ↔ square ↔ funnel ↔ pipe domain.
 
Coordinate convention
---------------------
  x →  streamwise direction (0 … L_total)
  y ↑  wall-normal direction (0 … H)
 
Q4 local node ordering inside every element (counter-clockwise from SW):
 
    3 ─────── 2
    │         │
    │    e    │        elem index e = ey * nelx + ex
    │         │        node index n = iy * (nelx+1) + ix
    0 ─────── 1
 
  0 = SW (bottom-left)   1 = SE (bottom-right)
  2 = NE (top-right)     3 = NW (top-left)
 
Element mask labels (ElemType)
-------------------------------
  INACTIVE    (0) — void / solid wall corner; excluded from the active solve
                    OR penalised with a large Brinkman coefficient α_max.
  FROZEN_FLUID(1) — element constrained to be pure fluid (ξ = 1);
                    occurs in inlet/outlet pipe stubs and funnel transitions.
  FREE        (2) — design-variable element; only the central square carries
                    these elements.  The optimiser updates ξ here.
 
Velocity DOF layout (Stokes)
-----------------------------
  Node n  →  u-DOF = 2n,   v-DOF = 2n+1
  edofMat_vel column order: [u_SW, v_SW, u_SE, v_SE, u_NE, v_NE, u_NW, v_NW]
 
Pressure DOF layout
--------------------
  Q4 continuous pressure (requires SUPG/GLS stabilisation).
  Pressure DOF = node index  (identical layout to the thermal field).
  edofMat_pres  ≡  edofMat_th
 
Thermal DOF layout
------------------
  Node n  →  temperature DOF = n
  edofMat_th column order:  [T_SW, T_SE, T_NE, T_NW]
"""
 
from __future__ import annotations
 
import os
import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple
 
import numpy as np
from numpy.typing import NDArray
 
# ---------------------------------------------------------------------------
# Make config importable when this file is run as a script directly
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULT_GEO, DEFAULT_SIM, GeoConfig, SimConfig
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 1.  Element classification enum
# ══════════════════════════════════════════════════════════════════════════════
 
class ElemType(IntEnum):
    """Spatial role of each finite element in the coupled solve."""
    INACTIVE      = 0   # void corner / solid wall — excluded / penalised
    FROZEN_FLUID  = 1   # forced fluid  (ξ = 1) — pipe stubs + funnels
    FREE          = 2   # design variable — central square only
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2.  Boundary-condition container
# ══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class BoundaryConditions:
    """
    All Dirichlet data and node-set bookkeeping required by the
    coupled Stokes + advection–diffusion solver and its adjoint.
 
    Stokes (velocity) field
    -----------------------
    vel_dirichlet_dofs  : int32 array, sorted, of every velocity DOF that
                          carries an essential BC.  Covers inlet (parabolic
                          u, v=0) and all no-slip wall nodes (u=v=0).
    vel_dirichlet_vals  : float64 array, same length, prescribed values.
 
    Thermal field
    -------------
    temp_dirichlet_nodes : node indices where T = T_inlet is imposed
                           (inlet pipe opening only).
    temp_dirichlet_vals  : prescribed temperature values (one per node).
    outlet_temp_nodes    : outlet-face nodes within the pipe; kept for
                           diagnostics — no essential BC is applied here
                           (zero-flux is the natural BC for advection-
                           diffusion outflow).
 
    Pressure field
    --------------
    pressure_ref_node : single node index where p = 0 is pinned to
                        remove the constant-pressure null space.
                        Chosen as the outlet node closest to y = H/2.
 
    Raw node-set helpers
    --------------------
    inlet_nodes_fluid  : (n_in,)  inlet face nodes inside the pipe aperture.
    outlet_nodes_fluid : (n_out,) outlet face nodes inside the pipe aperture.
    wall_nodes_top     : (nelx+1,) nodes along y = H.
    wall_nodes_bottom  : (nelx+1,) nodes along y = 0.
    inlet_u_profile    : (n_in,)  parabolic u-velocity values at inlet fluid
                                  nodes (v = 0 everywhere at inlet).
 
    Usage pattern (FE assembler)
    ----------------------------
    Given a stiffness matrix K (n×n) and rhs f (n,):
 
        all_dofs  = np.arange(n)
        fixed     = bc.vel_dirichlet_dofs
        free      = np.setdiff1d(all_dofs, fixed)
        u_known   = np.zeros(n); u_known[fixed] = bc.vel_dirichlet_vals
        f_mod     = f[free] - K[np.ix_(free, fixed)] @ bc.vel_dirichlet_vals
        u[free]   = spsolve(K[np.ix_(free, free)], f_mod)
        u[fixed]  = bc.vel_dirichlet_vals
    """
    # ── Stokes ──────────────────────────────────────────────────────────────
    vel_dirichlet_dofs:   NDArray[np.int32]
    vel_dirichlet_vals:   NDArray[np.float64]
 
    # ── Thermal ─────────────────────────────────────────────────────────────
    temp_dirichlet_nodes: NDArray[np.int32]
    temp_dirichlet_vals:  NDArray[np.float64]
    outlet_temp_nodes:    NDArray[np.int32]
 
    # ── Pressure ─────────────────────────────────────────────────────────────
    pressure_ref_node: int
 
    # ── Raw geometry helpers ─────────────────────────────────────────────────
    inlet_nodes_fluid:    NDArray[np.int32]
    outlet_nodes_fluid:   NDArray[np.int32]
    wall_nodes_top:       NDArray[np.int32]
    wall_nodes_bottom:    NDArray[np.int32]
    inlet_u_profile:      NDArray[np.float64]
 
    # ── Diagnostics ──────────────────────────────────────────────────────────
    def summary(self) -> str:
        n_inlet_vel = np.sum(
            self.vel_dirichlet_vals[
                np.searchsorted(self.vel_dirichlet_dofs,
                                2 * self.inlet_nodes_fluid)
            ] > 0
        )
        lines = [
            "BoundaryConditions",
            "─" * 44,
            f"  Velocity Dirichlet DOFs  : {len(self.vel_dirichlet_dofs):>6d}",
            f"    ├─ inlet (u-parabolic) : {len(self.inlet_nodes_fluid):>6d} nodes",
            f"    └─ wall  (no-slip)     : "
            f"{len(self.vel_dirichlet_dofs)//2 - len(self.inlet_nodes_fluid):>6d} nodes",
            f"  Thermal Dirichlet nodes  : {len(self.temp_dirichlet_nodes):>6d}",
            f"  Outlet (zero-flux) nodes : {len(self.outlet_temp_nodes):>6d}",
            f"  Pressure reference node  : {self.pressure_ref_node:>6d}",
            f"  U_max at inlet           : {self.inlet_u_profile.max()*1e3:>9.4f} mm/s",
            f"  U_mean at inlet          : {self.inlet_u_profile.mean()*1e3:>9.4f} mm/s",
        ]
        return "\n".join(lines)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3.  MeshDomain  — core class
# ══════════════════════════════════════════════════════════════════════════════
 
class MeshDomain:
    """
    Structured Q4 finite-element mesh for the microchannel heatsink domain.
 
    Constructs and exposes:
      • node/element coordinate arrays,
      • edofMat for thermal (4-DOF), velocity (8-DOF), and pressure (4-DOF),
      • element-type mask (INACTIVE / FROZEN_FLUID / FREE),
      • ready-to-use ``BoundaryConditions`` object.
 
    Parameters
    ----------
    geo : GeoConfig
        Geometry configuration.  Defaults to ``DEFAULT_GEO``.
    sim : SimConfig
        Simulation configuration (needed for Re, ν).  Defaults to
        ``DEFAULT_SIM``.
    T_inlet : float
        Non-dimensional (or physical) temperature prescribed at the inlet.
        Default 0.0 (cold inlet).
 
    Key attributes
    --------------
    nelx, nely    int           number of elements along x and y
    n_elem        int           nelx × nely
    n_node        int           (nelx+1) × (nely+1)
    dx, dy        float         element dimensions [m]
    node_x        (n_node,)     nodal x-coordinates [m]
    node_y        (n_node,)     nodal y-coordinates [m]
    elem_cx       (n_elem,)     element centroid x [m]
    elem_cy       (n_elem,)     element centroid y [m]
    edofMat_th    (n_elem, 4)   thermal / pressure DOF indices
    edofMat_vel   (n_elem, 8)   velocity DOF indices
    edofMat_pres  (n_elem, 4)   pressure DOF indices  (alias of edofMat_th)
    elem_type     (n_elem,)     int8 ElemType label per element
    bc            BoundaryConditions
    """
 
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        geo: GeoConfig = DEFAULT_GEO,
        sim: SimConfig = DEFAULT_SIM,
        T_inlet: float = 0.0,
    ) -> None:
        self.geo     = geo
        self.sim     = sim
        self.T_inlet = float(T_inlet)
 
        self.nelx = geo.nx
        self.nely = geo.ny
        self.n_elem = self.nelx * self.nely
        self.n_node = (self.nelx + 1) * (self.nely + 1)
 
        # Number of DOFs per physics field
        self.n_vel_dofs  = 2 * self.n_node   # interleaved u,v
        self.n_temp_dofs = self.n_node        # one per node
        self.n_pres_dofs = self.n_node        # Q4 continuous
 
        # Physical element size [m]
        self.dx = geo.L_total / self.nelx
        self.dy = geo.H       / self.nely
 
        # ── Build the mesh ────────────────────────────────────────────────────
        self.node_x, self.node_y = self._build_node_coords()
        self.elem_cx, self.elem_cy = self._build_elem_centroids()
 
        # ── DOF-connectivity tables ───────────────────────────────────────────
        self.edofMat_th   = self._build_edofMat_thermal()
        self.edofMat_vel  = self._build_edofMat_stokes()
        self.edofMat_pres = self.edofMat_th   # Q4 pressure shares node indices
 
        # ── Spatial classification ────────────────────────────────────────────
        self.elem_type = self._classify_elements()
 
        # ── Boundary conditions ───────────────────────────────────────────────
        self.bc = self._build_bc()
 
    # ══════════════════════════════════════════════════════════════════════════
    # 3a. Node and element coordinate arrays
    # ══════════════════════════════════════════════════════════════════════════
 
    def _build_node_coords(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Create 1-D arrays node_x, node_y of length n_node.
 
        Ordering: node(ix, iy) = iy*(nelx+1) + ix
          → row iy contains ix = 0 … nelx, all at y = iy*dy.
 
        Returns
        -------
        node_x : (n_node,) [m]
        node_y : (n_node,) [m]
        """
        nelx, nely = self.nelx, self.nely
        ix_vals = np.arange(nelx + 1, dtype=np.float64) * self.dx
        iy_vals = np.arange(nely + 1, dtype=np.float64) * self.dy
 
        # Tile x-values for every row; repeat y-values across each row
        node_x = np.tile(ix_vals, nely + 1)           # (n_node,)
        node_y = np.repeat(iy_vals, nelx + 1)         # (n_node,)
        return node_x, node_y
 
    def _build_elem_centroids(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Create 1-D arrays elem_cx, elem_cy of length n_elem.
 
        Ordering: elem(ex, ey) = ey*nelx + ex
          → row ey contains ex = 0 … nelx-1, centroid at y=(ey+0.5)*dy.
 
        Returns
        -------
        elem_cx : (n_elem,) [m]
        elem_cy : (n_elem,) [m]
        """
        nelx, nely = self.nelx, self.nely
        ex = np.tile(np.arange(nelx), nely)       # repeats ex pattern per row
        ey = np.repeat(np.arange(nely), nelx)     # repeats each ey nelx times
        return (ex + 0.5) * self.dx, (ey + 0.5) * self.dy
 
    # ══════════════════════════════════════════════════════════════════════════
    # 3b. DOF-connectivity matrices
    # ══════════════════════════════════════════════════════════════════════════
 
    def _build_edofMat_thermal(self) -> NDArray[np.int32]:
        """
        Return (n_elem, 4) array of thermal DOF indices.
 
        For the thermal (and pressure) field, DOF index = node index.
 
        Column layout  →  [SW, SE, NE, NW]  (counter-clockwise from bottom-left)
 
        For element (ex, ey):
            SW = ey*(nelx+1) + ex
            SE = SW + 1
            NE = SE + (nelx+1)
            NW = SW + (nelx+1)
        """
        nelx = self.nelx
        ex = np.tile(np.arange(nelx), self.nely).astype(np.int32)
        ey = np.repeat(np.arange(self.nely), nelx).astype(np.int32)
 
        n_sw = ey * (nelx + 1) + ex          # bottom-left  (SW)
        n_se = n_sw + 1                       # bottom-right (SE)
        n_ne = n_se + (nelx + 1)              # top-right    (NE)
        n_nw = n_sw + (nelx + 1)              # top-left     (NW)
 
        return np.column_stack([n_sw, n_se, n_ne, n_nw]).astype(np.int32)
 
    def _build_edofMat_stokes(self) -> NDArray[np.int32]:
        """
        Return (n_elem, 8) array of Stokes velocity DOF indices.
 
        DOF layout per node n:  u-DOF = 2n,  v-DOF = 2n+1
 
        Column layout (8 cols):
            [u_SW, v_SW,  u_SE, v_SE,  u_NE, v_NE,  u_NW, v_NW]
 
        Derived by interleaving the four node indices from edofMat_th.
        """
        th = self.edofMat_th                    # (n_elem, 4) node indices
        out = np.empty((self.n_elem, 8), dtype=np.int32)
        for local_node in range(4):             # 0=SW, 1=SE, 2=NE, 3=NW
            out[:, 2 * local_node]     = 2 * th[:, local_node]      # u
            out[:, 2 * local_node + 1] = 2 * th[:, local_node] + 1  # v
        return out
 
    # ══════════════════════════════════════════════════════════════════════════
    # 3c. Spatial element classification  (pipe / funnel / square)
    # ══════════════════════════════════════════════════════════════════════════
 
    def _classify_elements(self) -> NDArray[np.int8]:
        """
        Assign an ``ElemType`` label to every element based on its centroid.
 
        Domain decomposition (no funnels)
        ---------------------------------
        Pipe → Square → Pipe.  Abrupt expansion from W_pipe to full H.
 
            y_lo = (H − W_pipe) / 2,   y_hi = (H + W_pipe) / 2.
 
        Region                  x-range               Fluid region (y)
        ─────────────────────── ───────────────────── ───────────────────
        Inlet pipe              [0, L_pipe)           [y_lo, y_hi]
        Central square          [L_pipe, L_pipe+L_sq) [0, H]  (all FREE)
        Outlet pipe             [L_pipe+L_sq, L_total] [y_lo, y_hi]
 
        Corner cells outside the pipe aperture are INACTIVE.
        """
        geo = self.geo
        xc  = self.elem_cx    # (n_elem,) centroid x
        yc  = self.elem_cy    # (n_elem,) centroid y
 
        # Pipe y-bounds
        y_lo_pipe = (geo.H - geo.W_pipe) / 2.0
        y_hi_pipe = (geo.H + geo.W_pipe) / 2.0
 
        # x-breakpoints (no funnels)
        x_sq_start = geo.x_square_start              # = L_pipe
        x_sq_end   = geo.x_square_end                # = L_pipe + L_square
 
        # Initialise everything as INACTIVE (wall / void)
        mask = np.full(self.n_elem, int(ElemType.INACTIVE), dtype=np.int8)
 
        # ── Region 1: inlet pipe stub ──────────────────────────────────────
        r = xc < x_sq_start
        in_y = (yc >= y_lo_pipe) & (yc <= y_hi_pipe)
        mask[r & in_y] = ElemType.FROZEN_FLUID
 
        # ── Region 2: central square (topology-optimisable zone) ────────────
        mask[(xc >= x_sq_start) & (xc < x_sq_end)] = ElemType.FREE
 
        # ── Region 3: outlet pipe stub ──────────────────────────────────────
        r = xc >= x_sq_end
        mask[r & (yc >= y_lo_pipe) & (yc <= y_hi_pipe)] = ElemType.FROZEN_FLUID
 
        return mask
 
    # ══════════════════════════════════════════════════════════════════════════
    # 3d. Boundary conditions
    # ══════════════════════════════════════════════════════════════════════════
 
    def _build_bc(self) -> BoundaryConditions:
        """
        Assemble all Dirichlet data and raw node sets.
 
        Inlet velocity
        --------------
        A parabolic (Hagen-Poiseuille) profile is prescribed at x = 0 for
        nodes inside the pipe aperture [y_lo_pipe, y_hi_pipe]:
 
            u(y) = U_max · [1 − ((y − H/2) / (W_pipe/2))²]
            v(y) = 0
 
        with   U_mean = Re · ν / W_pipe,   U_max = (3/2) · U_mean.
 
        Wall (no-slip)
        --------------
        u = v = 0 on:
          • top wall   (y = H),
          • bottom wall (y = 0),
          • inlet-face nodes outside the pipe aperture  (x = 0, |y−H/2|>W_pipe/2),
          • outlet-face nodes outside the pipe aperture (x = L, |y−H/2|>W_pipe/2).
 
        Outlet (do-nothing / outflow)
        -----------------------------
        No essential BC on velocity; the natural BC of the weak form yields
        zero traction  σ·n = 0  on the outlet face.
 
        Thermal
        -------
        T = T_inlet (cold) prescribed at inlet fluid nodes.
        Zero-flux outflow is the natural BC and requires no action.
        """
        geo  = self.geo
        sim  = self.sim
        nelx, nely = self.nelx, self.nely
 
        y_lo_pipe = (geo.H - geo.W_pipe) / 2.0
        y_hi_pipe = (geo.H + geo.W_pipe) / 2.0
        y_center  = geo.H / 2.0
 
        # ── Raw boundary face node sets ───────────────────────────────────
        all_inlet_nodes  = (np.arange(nely + 1) * (nelx + 1)).astype(np.int32)
        all_outlet_nodes = (np.arange(nely + 1) * (nelx + 1) + nelx).astype(np.int32)
        nodes_wall_bottom = np.arange(nelx + 1, dtype=np.int32)
        nodes_wall_top    = np.arange(
            nely * (nelx + 1), nely * (nelx + 1) + nelx + 1, dtype=np.int32
        )
 
        # ── Partition inlet/outlet faces into pipe-fluid vs wall ──────────
        _tol = 1e-13  # floating-point comparison tolerance
 
        y_in  = self.node_y[all_inlet_nodes]
        in_fluid_mask   = (y_in >= y_lo_pipe - _tol) & (y_in <= y_hi_pipe + _tol)
        inlet_nodes_fluid = all_inlet_nodes[in_fluid_mask]
        inlet_nodes_wall  = all_inlet_nodes[~in_fluid_mask]
 
        y_out = self.node_y[all_outlet_nodes]
        out_fluid_mask    = (y_out >= y_lo_pipe - _tol) & (y_out <= y_hi_pipe + _tol)
        outlet_nodes_fluid = all_outlet_nodes[out_fluid_mask]
        outlet_nodes_wall  = all_outlet_nodes[~out_fluid_mask]
 
        # ── Parabolic inlet velocity profile ──────────────────────────────
        nu     = sim.water.kinematic_viscosity         # [m² s⁻¹]
        U_mean = sim.reynolds_number * nu / geo.W_pipe  # Re = U_mean·W/ν
        U_max  = 1.5 * U_mean                          # Hagen-Poiseuille: 3/2 U_mean
        R      = geo.W_pipe / 2.0                      # half-width of pipe
 
        y_in_fluid = self.node_y[inlet_nodes_fluid]
        u_parabolic = U_max * (1.0 - ((y_in_fluid - y_center) / R) ** 2)
        u_parabolic = np.maximum(u_parabolic, 0.0)     # clip corners to zero
 
        # ── Stokes Dirichlet: inlet (u-parabolic, v=0) ────────────────────
        u_dofs_inlet = (2 * inlet_nodes_fluid).astype(np.int32)
        v_dofs_inlet = (2 * inlet_nodes_fluid + 1).astype(np.int32)
        u_vals_inlet = u_parabolic
        v_vals_inlet = np.zeros(len(inlet_nodes_fluid), dtype=np.float64)
 
        # ── Stokes Dirichlet: no-slip walls ───────────────────────────────
        # Collect all wall nodes: top + bottom + inlet stub + outlet stub
        wall_nodes_all = np.unique(np.concatenate([
            nodes_wall_top,
            nodes_wall_bottom,
            inlet_nodes_wall,
            outlet_nodes_wall,
        ])).astype(np.int32)
        u_dofs_wall = (2 * wall_nodes_all).astype(np.int32)
        v_dofs_wall = (2 * wall_nodes_all + 1).astype(np.int32)
 
        # ── Merge all velocity Dirichlet entries (sorted by DOF index) ────
        all_vel_dofs = np.concatenate([
            u_dofs_inlet, v_dofs_inlet,
            u_dofs_wall,  v_dofs_wall,
        ])
        all_vel_vals = np.concatenate([
            u_vals_inlet, v_vals_inlet,
            np.zeros(2 * len(wall_nodes_all), dtype=np.float64),
        ])
        sort_idx = np.argsort(all_vel_dofs, kind="stable")
        all_vel_dofs = all_vel_dofs[sort_idx].astype(np.int32)
        all_vel_vals = all_vel_vals[sort_idx]
 
        # Guard against duplicates (inlet-wall corner nodes appear in both sets)
        unique_idx = np.concatenate([[True], np.diff(all_vel_dofs) > 0])
        all_vel_dofs = all_vel_dofs[unique_idx]
        all_vel_vals = all_vel_vals[unique_idx]
 
        # ── Thermal Dirichlet: T = T_inlet at inlet fluid nodes ───────────
        temp_dir_nodes = inlet_nodes_fluid.copy()
        temp_dir_vals  = np.full(len(temp_dir_nodes), self.T_inlet, dtype=np.float64)
 
        # ── Pressure reference: outlet node nearest y = H/2 ───────────────
        y_out_fluid = self.node_y[outlet_nodes_fluid]
        pref_local  = np.argmin(np.abs(y_out_fluid - y_center))
        pref_node   = int(outlet_nodes_fluid[pref_local])
 
        return BoundaryConditions(
            vel_dirichlet_dofs   = all_vel_dofs,
            vel_dirichlet_vals   = all_vel_vals,
            temp_dirichlet_nodes = temp_dir_nodes,
            temp_dirichlet_vals  = temp_dir_vals,
            outlet_temp_nodes    = outlet_nodes_fluid,
            pressure_ref_node    = pref_node,
            inlet_nodes_fluid    = inlet_nodes_fluid,
            outlet_nodes_fluid   = outlet_nodes_fluid,
            wall_nodes_top       = nodes_wall_top,
            wall_nodes_bottom    = nodes_wall_bottom,
            inlet_u_profile      = u_parabolic,
        )
 
    # ══════════════════════════════════════════════════════════════════════════
    # 3e. Convenience properties and helpers
    # ══════════════════════════════════════════════════════════════════════════
 
    @property
    def design_elements(self) -> NDArray[np.int32]:
        """Indices of FREE elements — the topology design variables."""
        return np.where(self.elem_type == ElemType.FREE)[0].astype(np.int32)
 
    @property
    def frozen_elements(self) -> NDArray[np.int32]:
        """Indices of FROZEN_FLUID elements (pipe stubs + funnels)."""
        return np.where(self.elem_type == ElemType.FROZEN_FLUID)[0].astype(np.int32)
 
    @property
    def inactive_elements(self) -> NDArray[np.int32]:
        """Indices of INACTIVE (void/wall corner) elements."""
        return np.where(self.elem_type == ElemType.INACTIVE)[0].astype(np.int32)
 
    @property
    def active_elements(self) -> NDArray[np.int32]:
        """Union of FROZEN_FLUID and FREE — all elements participating in the solve."""
        return np.where(self.elem_type != ElemType.INACTIVE)[0].astype(np.int32)
 
    @property
    def n_design_vars(self) -> int:
        """Number of topology design variables (== len(design_elements))."""
        return int(np.sum(self.elem_type == ElemType.FREE))
 
    def free_velocity_dofs(self) -> NDArray[np.int32]:
        """
        Return velocity DOF indices NOT in ``bc.vel_dirichlet_dofs``.
        These are the unknowns solved for in the Stokes system.
        """
        all_dofs = np.arange(self.n_vel_dofs, dtype=np.int32)
        return np.setdiff1d(all_dofs, self.bc.vel_dirichlet_dofs, assume_unique=True)
 
    def free_temp_dofs(self) -> NDArray[np.int32]:
        """
        Return thermal DOF indices NOT prescribed at the inlet.
        """
        all_dofs = np.arange(self.n_temp_dofs, dtype=np.int32)
        return np.setdiff1d(all_dofs, self.bc.temp_dirichlet_nodes, assume_unique=True)
 
    def initial_design_field(self, fill: float = 0.5) -> NDArray[np.float64]:
        """
        Construct an initial element-wise density field ξ ∈ [0, 1].
 
        Parameters
        ----------
        fill : float
            Starting density for FREE elements (0.5 = uniform grey).
 
        Returns
        -------
        xi : (n_elem,) float64
            ξ = 0.0 for INACTIVE, ξ = 1.0 for FROZEN_FLUID,
            ξ = fill  for FREE.
        """
        if not 0.0 <= fill <= 1.0:
            raise ValueError(f"fill must be in [0, 1], got {fill}")
        xi = np.zeros(self.n_elem, dtype=np.float64)
        xi[self.frozen_elements]  = 1.0
        xi[self.design_elements]  = fill
        return xi
 
    def node_id(self, ix: int, iy: int) -> int:
        """Return global node index for integer grid coordinates (ix, iy)."""
        return int(iy * (self.nelx + 1) + ix)
 
    def elem_id(self, ex: int, ey: int) -> int:
        """Return global element index for integer grid coordinates (ex, ey)."""
        return int(ey * self.nelx + ex)
 
    def elem_nodes(self, elem_idx: int) -> NDArray[np.int32]:
        """Return the four global node indices [SW, SE, NE, NW] for element idx."""
        return self.edofMat_th[elem_idx].copy()
 
    # ══════════════════════════════════════════════════════════════════════════
    # 3f. Diagnostics / visualisation
    # ══════════════════════════════════════════════════════════════════════════
 
    def summary(self) -> str:
        """Human-readable summary of mesh statistics and BCs."""
        n_free   = int(np.sum(self.elem_type == ElemType.FREE))
        n_frozen = int(np.sum(self.elem_type == ElemType.FROZEN_FLUID))
        n_inact  = int(np.sum(self.elem_type == ElemType.INACTIVE))
        n_free_vel  = len(self.free_velocity_dofs())
        n_free_temp = len(self.free_temp_dofs())
        U_mean = self.bc.inlet_u_profile.mean()
 
        lines = [
            "MeshDomain",
            "═" * 52,
            f"  Grid              :  {self.nelx} × {self.nely}"
            f"  ({self.n_elem} elements,  {self.n_node} nodes)",
            f"  Element size Δx   :  {self.dx * 1e6:.2f} µm",
            f"  Element size Δy   :  {self.dy * 1e6:.2f} µm",
            f"  Aspect ratio Δx/Δy:  {self.dx/self.dy:.3f}",
            "  ─────────────────────────────────────────────────────",
            f"  FREE elements     :  {n_free:>6d}  (design variables)",
            f"  FROZEN_FLUID      :  {n_frozen:>6d}  (pipe + funnels)",
            f"  INACTIVE          :  {n_inact:>6d}  (void corners)",
            f"  Active total      :  {n_free + n_frozen:>6d}",
            "  ─────────────────────────────────────────────────────",
            f"  Velocity DOFs     :  {self.n_vel_dofs}  "
            f"(free: {n_free_vel},  Dirichlet: {len(self.bc.vel_dirichlet_dofs)})",
            f"  Thermal DOFs      :  {self.n_temp_dofs}  "
            f"(free: {n_free_temp},  Dirichlet: {len(self.bc.temp_dirichlet_nodes)})",
            f"  Pressure DOFs     :  {self.n_pres_dofs}  (pinned at node "
            f"{self.bc.pressure_ref_node})",
            "  ─────────────────────────────────────────────────────",
            f"  U_mean inlet      :  {U_mean * 1e3:.4f} mm/s",
            f"  U_max  inlet      :  {self.bc.inlet_u_profile.max() * 1e3:.4f} mm/s",
            f"  Re                :  {self.sim.reynolds_number}",
            "",
            self.bc.summary(),
        ]
        return "\n".join(lines)
 
    def ascii_map(self, max_cols: int = 72) -> str:
        """
        Render a bird's-eye ASCII view of element classification.
 
        Legend:  F = FREE   f = FROZEN_FLUID   · = INACTIVE
        """
        nex, ney = self.nelx, self.nely
        step_x = max(1, nex // max_cols)
        step_y = max(1, ney // (max_cols // 4))
 
        chars = {
            int(ElemType.INACTIVE):     "·",
            int(ElemType.FROZEN_FLUID): "f",
            int(ElemType.FREE):         "F",
        }
        # Reshape to (ney, nex): grid[ey, ex]
        grid = self.elem_type.reshape(ney, nex).astype(np.int8)
 
        lines = [
            f"  ASCII element map  "
            f"(F=FREE  f=FROZEN_FLUID  ·=INACTIVE)  "
            f"[step {step_x}×{step_y}]",
            "  +" + "─" * len(range(0, nex, step_x)) + "+",
        ]
        for ey in range(ney - 1, -1, -step_y):    # top → bottom in physical y
            row = "  |"
            for ex in range(0, nex, step_x):
                row += chars[int(grid[ey, ex])]
            row += "|"
            lines.append(row)
        lines.append("  +" + "─" * len(range(0, nex, step_x)) + "+")
        return "\n".join(lines)
 
    def plot_domain(
        self,
        show_bc: bool = True,
        figsize: Tuple[int, int] = (14, 5),
    ) -> None:
        """
        Visualise element classification and boundary conditions with matplotlib.
 
        Requires matplotlib.  Raises ImportError if not installed.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.colors import ListedColormap
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_domain(); "
                "install it with:  pip install matplotlib"
            ) from exc
 
        cmap = ListedColormap(["#2d2d2d", "#4fc3f7", "#ef9a9a"])  # inactive/fluid/free
        grid = self.elem_type.reshape(self.nely, self.nelx).astype(float)
 
        fig, ax = plt.subplots(figsize=figsize)
        extent = [0, self.geo.L_total * 1e6, 0, self.geo.H * 1e6]
        im = ax.imshow(
            np.flipud(grid), cmap=cmap, vmin=-0.5, vmax=2.5,
            extent=extent, aspect="equal", interpolation="nearest",
        )
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        ax.set_title("MeshDomain — element classification", fontweight="bold")
 
        if show_bc:
            # Inlet velocity profile (quiver)
            y_in = self.node_y[self.bc.inlet_nodes_fluid] * 1e6
            u_in = self.bc.inlet_u_profile
            ax.barh(y_in, u_in / u_in.max() * 40, left=0,
                    height=self.dy * 1e6 * 0.8, color="#fff176", alpha=0.9,
                    label="Inlet velocity (parabolic)")
            # Outlet marker
            x_out = self.geo.L_total * 1e6
            ax.axvline(x_out, color="#a5d6a7", lw=2, ls="--", label="Outlet (do-nothing)")
            # Pressure pin
            px = self.node_x[self.bc.pressure_ref_node] * 1e6
            py = self.node_y[self.bc.pressure_ref_node] * 1e6
            ax.plot(px, py, "m*", ms=10, label=f"Pressure pin (node {self.bc.pressure_ref_node})")
 
        legend_patches = [
            mpatches.Patch(color="#2d2d2d", label="INACTIVE (wall/void)"),
            mpatches.Patch(color="#4fc3f7", label="FROZEN_FLUID"),
            mpatches.Patch(color="#ef9a9a", label="FREE (design variables)"),
        ]
        ax.legend(handles=legend_patches + (ax.get_legend_handles_labels()[0]
                                             if show_bc else []),
                  loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.show()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4.  Module-level convenience factory
# ══════════════════════════════════════════════════════════════════════════════
 
def build_default_mesh(
    geo: GeoConfig = DEFAULT_GEO,
    sim: SimConfig = DEFAULT_SIM,
    T_inlet: float = 0.0,
) -> MeshDomain:
    """
    Convenience factory — equivalent to ``MeshDomain(geo, sim, T_inlet)``
    but signals intent clearly in calling code:
 
        mesh = build_default_mesh()
    """
    return MeshDomain(geo=geo, sim=sim, T_inlet=T_inlet)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 5.  Quick self-test  (python mesh_geometry.py)
# ══════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    import traceback
 
    print("Building default MeshDomain …")
    mesh = build_default_mesh()
 
    print(mesh.summary())
    print()
    print(mesh.ascii_map())
    print()
 
    # ── Structural checks ─────────────────────────────────────────────────
    errors: list[str] = []
 
    # 1. DOF matrix shapes
    assert mesh.edofMat_th.shape  == (mesh.n_elem, 4), "edofMat_th shape mismatch"
    assert mesh.edofMat_vel.shape == (mesh.n_elem, 8), "edofMat_vel shape mismatch"
    assert mesh.edofMat_pres.shape == (mesh.n_elem, 4), "edofMat_pres shape mismatch"
 
    # 2. Node indices in range
    assert mesh.edofMat_th.min()  >= 0 and mesh.edofMat_th.max()  < mesh.n_node
    assert mesh.edofMat_vel.min() >= 0 and mesh.edofMat_vel.max() < 2 * mesh.n_node
 
    # 3. Velocity DOFs interleaving: col 0 = 2*col_th0, col 1 = 2*col_th0+1
    for loc in range(4):
        assert np.all(mesh.edofMat_vel[:, 2*loc]   == 2 * mesh.edofMat_th[:, loc])
        assert np.all(mesh.edofMat_vel[:, 2*loc+1] == 2 * mesh.edofMat_th[:, loc] + 1)
 
    # 4. Element type coverage
    n_free   = len(mesh.design_elements)
    n_frozen = len(mesh.frozen_elements)
    n_inact  = len(mesh.inactive_elements)
    assert n_free + n_frozen + n_inact == mesh.n_elem, "Element count mismatch"
    assert n_free  > 0, "No FREE elements — check geometry"
    assert n_frozen > 0, "No FROZEN_FLUID elements — check geometry"
 
    # 5. Initial design field values
    xi = mesh.initial_design_field(fill=0.5)
    assert xi.shape == (mesh.n_elem,)
    assert np.allclose(xi[mesh.frozen_elements], 1.0)
    assert np.allclose(xi[mesh.design_elements], 0.5)
    assert np.allclose(xi[mesh.inactive_elements], 0.0)
 
    # 6. Velocity BC: no duplicate DOFs
    assert len(mesh.bc.vel_dirichlet_dofs) == len(
        np.unique(mesh.bc.vel_dirichlet_dofs)
    ), "Duplicate velocity Dirichlet DOFs"
 
    # 7. Free + fixed = all DOFs
    free_v = mesh.free_velocity_dofs()
    assert len(free_v) + len(mesh.bc.vel_dirichlet_dofs) == mesh.n_vel_dofs
 
    # 8. Parabolic profile non-negative and peaks near channel centre
    assert np.all(mesh.bc.inlet_u_profile >= -1e-12)
    y_in    = mesh.node_y[mesh.bc.inlet_nodes_fluid]
    y_peak  = y_in[np.argmax(mesh.bc.inlet_u_profile)]
    assert abs(y_peak - mesh.geo.H / 2) < mesh.dy + 1e-12, \
        "Parabolic peak not near channel centre"
 
    # 9. Pressure ref node is inside domain
    assert 0 <= mesh.bc.pressure_ref_node < mesh.n_node
 
    print("✓  All self-tests passed.\n")
 
    # ── Print element counts per region ────────────────────────────────────
    print("Element census")
    print("─" * 36)
    print(f"  FREE          : {n_free:>6d}  ({100*n_free/mesh.n_elem:.1f} %)")
    print(f"  FROZEN_FLUID  : {n_frozen:>6d}  ({100*n_frozen/mesh.n_elem:.1f} %)")
    print(f"  INACTIVE      : {n_inact:>6d}  ({100*n_inact/mesh.n_elem:.1f} %)")
    print(f"  TOTAL         : {mesh.n_elem:>6d}")
 