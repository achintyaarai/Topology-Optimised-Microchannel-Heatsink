"""
optimizer.py -- eps-Constraint MMA Topology Optimizer
====================================================
Implements the Wang et al. (2025) fractal microchannel methodology:
  - P-norm thermal objective JQ with full adjoint
  - Temperature variance JTV with adjoint through same factorised matrix
  - Borrvall-Petersson flow dissipation Jf (direct sensitivity)
  - Blob penalisation J_blob to suppress solid agglomeration
  - eps-constraint NLopt formulation (JQ primary, Jf and JTV as constraints)
  - Coupled flow->temperature adjoint for advection sensitivity
  - Half-plane symmetry about y=H/2 to halve design variables
  - Strict dual volume-fraction constraint (target +/- 0.05)
"""

import numpy as np
import nlopt
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from config import SimConfig
from mesh_geometry import MeshDomain
from physics_solver import PhysicsSolver


class Optimizer:
    """
    MMA topology optimizer for microchannel heatsink with half-plane symmetry.
    Design variables live on the TOP half (y >= H/2) only; the bottom half
    is mirrored.
    """

    def __init__(self, sim_config: SimConfig, mesh: MeshDomain, solver: PhysicsSolver):
        self.cfg = sim_config
        self.mesh = mesh
        self.solver = solver

        # Design variables correspond to FREE elements (the topology zone)
        self.design_elements = mesh.design_elements
        self.n_design = len(self.design_elements)
        self.iteration = 0

        # ── Build half-plane symmetry map ─────────────────────────────────────
        self._build_symmetry_map()

        # Build spatial density filter matrix H (on FULL design space)
        self._build_filter()

        # Base normalisation factors (set during optimize())
        self.base_JQ = None
        self.base_Jf = None
        self.base_JTV = None

    # ──────────────────────────────────────────────────────────────────────────
    # Half-plane symmetry mapping
    # ──────────────────────────────────────────────────────────────────────────

    def _build_symmetry_map(self):
        """
        Build top-half / bottom-half element pairing.

        For each design element, determine if its centroid is in the top
        half (y >= H/2) or bottom half.  Every bottom-half element is mapped
        to its y-mirror partner in the top half.

        Attributes set:
            n_var         : number of independent design variables (top half)
            top_indices   : indices into design_elements for top-half elements
            sym_expand    : (n_design,) array mapping each design element to
                            the independent variable index in [0, n_var)
        """
        H = self.mesh.geo.H
        y_mid = H / 2.0
        cx = self.mesh.elem_cx[self.design_elements]
        cy = self.mesh.elem_cy[self.design_elements]

        # Identify top-half elements (y_centroid >= y_mid)
        top_mask = cy >= y_mid
        self.top_indices = np.where(top_mask)[0]
        self.n_var = len(self.top_indices)

        # Build KDTree on top-half centroids to find mirror partners
        top_cx = cx[self.top_indices]
        top_cy = cy[self.top_indices]
        tree_top = cKDTree(np.column_stack([top_cx, top_cy]))

        # For each design element, find the closest top-half element
        # (which is itself for top-half elements, or the y-mirror for bottom)
        mirror_cy = np.where(top_mask, cy, H - cy)  # flip bottom y to top
        query_pts = np.column_stack([cx, mirror_cy])
        _, partner_idx = tree_top.query(query_pts)

        self.sym_expand = partner_idx  # (n_design,) -> index into top variables

        n_bottom = np.sum(~top_mask)
        print(f"  Symmetry: {self.n_var} top-half vars, {n_bottom} mirrored bottom-half elements")

    def expand_to_full(self, x_half):
        """Expand top-half design variables to full design field."""
        return x_half[self.sym_expand]

    def reduce_gradient(self, grad_full):
        """Sum gradients from paired elements back to top-half variables."""
        grad_half = np.zeros(self.n_var, dtype=np.float64)
        np.add.at(grad_half, self.sym_expand, grad_full)
        return grad_half

    # ──────────────────────────────────────────────────────────────────────────
    # Density filter (cone-based, row-normalised, on FULL design space)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_filter(self):
        """Constructs a sparse neighborhood cone filter matrix H for density smoothing."""
        cx = self.mesh.elem_cx[self.design_elements]
        cy = self.mesh.elem_cy[self.design_elements]

        tree = cKDTree(np.column_stack([cx, cy]))
        r_min = self.cfg.filter_radius

        sp_dist = tree.sparse_distance_matrix(tree, r_min, output_type='coo_matrix')

        V = r_min - sp_dist.data
        I = sp_dist.row
        J = sp_dist.col

        H = sp.coo_matrix((V, (I, J)), shape=(self.n_design, self.n_design)).tocsr()
        diag = H.diagonal()
        H.setdiag(np.maximum(diag, r_min))

        # Row-normalize
        row_sums = np.array(H.sum(axis=1)).ravel()
        self.H_filter = sp.diags(1.0 / row_sums) @ H

    # ──────────────────────────────────────────────────────────────────────────
    # Continuation schedule
    # ──────────────────────────────────────────────────────────────────────────

    def get_continuation_params(self, iteration):
        """Cosine Da decay + exponential beta increase."""
        progress = min(iteration / max(self.cfg.max_opt_iter, 1), 1.0)

        Da_start, Da_end = 1e-3, 1e-5
        Da = Da_end + 0.5 * (Da_start - Da_end) * (1 + np.cos(np.pi * progress))

        b_i, b_m = self.cfg.filter_beta_init, self.cfg.filter_beta_max
        beta = b_i * (b_m / b_i) ** progress if b_m > b_i else b_i

        return Da, beta

    # ──────────────────────────────────────────────────────────────────────────
    # Heaviside projection
    # ──────────────────────────────────────────────────────────────────────────

    def apply_projection(self, x_tilde, beta, eta=0.5):
        """Smoothed Heaviside: returns projected field and its derivative."""
        denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        x_phys = (np.tanh(beta * eta) + np.tanh(beta * (x_tilde - eta))) / denom
        d_phys_d_tilde = beta * (1.0 - np.tanh(beta * (x_tilde - eta)) ** 2) / denom
        return x_phys, d_phys_d_tilde

    # ──────────────────────────────────────────────────────────────────────────
    # Adjoint sensitivity computation  (JQ, Jf, JTV)
    # ──────────────────────────────────────────────────────────────────────────

    def compute_sensitivities(self, xi_phys, velocity, T):
        """
        Compute objectives and their design sensitivities via discrete adjoint.

        Returns
        -------
        dJQ_full, dJf_full, dJTV_full : (n_elem,) arrays
        JQ_val, Jf_val, JTV_val : float
        """
        n_node = self.mesh.n_node
        n_elem = self.mesh.n_elem

        # ── 1. Thermal adjoints (JQ and JTV) ─────────────────────────────────

        # JQ: P-norm thermal hotspot objective
        p_val = self.cfg.p_norm_exponent
        T_clipped = np.maximum(T, self.cfg.t_clip_floor)
        T_mean_p = np.mean(T_clipped ** p_val)
        JQ_val = T_mean_p ** (1.0 / p_val)

        dT_dp = (1.0 / n_node) * (JQ_val ** (1.0 - p_val)) * (T_clipped ** (p_val - 1.0))
        rhs_T_adj_Q = -dT_dp
        rhs_T_adj_Q[self.solver._temp_fixed] = 0.0

        # JTV: Temperature variance
        T_mean = np.mean(T)
        V_domain = self.mesh.geo.L_total * self.mesh.geo.H
        JTV_val = (1.0 / V_domain) * (T - T_mean).T @ self.solver.M_mass @ (T - T_mean)

        rhs_T_adj_TV = -(2.0 / V_domain) * self.solver.M_mass.dot(T - T_mean)
        rhs_T_adj_TV[self.solver._temp_fixed] = 0.0

        # Solve thermal adjoint
        csr_T_trans = self.solver._last_therm_csr.transpose().tocsc()
        csr_T_trans = csr_T_trans + sp.eye(n_node, format='csc') * 1e-12

        lambda_T_Q  = spla.spsolve(csr_T_trans, rhs_T_adj_Q)
        lambda_T_TV = spla.spsolve(csr_T_trans, rhs_T_adj_TV)

        # ── Thermal sensitivity: dk_eff/dxi contribution ─────────────────────
        dJQ_full  = np.zeros(n_elem)
        dJTV_full = np.zeros(n_elem)

        p = self.cfg.k_penalty_power
        d_keff_dxi = -p * (self.solver.k_s - self.solver.k_f) * (1.0 - xi_phys) ** (p - 1.0)

        lambda_T_Q_e  = lambda_T_Q[self.mesh.edofMat_th]
        lambda_T_TV_e = lambda_T_TV[self.mesh.edofMat_th]
        T_e = T[self.mesh.edofMat_th]

        for g in range(4):
            dNdx = self.solver._gp_dNdx[g]
            dNdy = self.solver._gp_dNdy[g]
            w = self.solver._gp_detJ[g]
            B = np.vstack([dNdx, dNdy])
            Kd_ref = B.T @ B * w

            val_Q  = np.einsum('ni,ij,nj->n', lambda_T_Q_e,  Kd_ref, T_e)
            val_TV = np.einsum('ni,ij,nj->n', lambda_T_TV_e, Kd_ref, T_e)

            dJQ_full  += d_keff_dxi * val_Q
            dJTV_full += d_keff_dxi * val_TV

        # ── 2. Fluid dissipation sensitivity (direct) ────────────────────────
        q = self.solver.q_penalty
        a_max = self.solver.alpha_max
        d_alpha_dxi = -a_max * (1.0 + q) / (1.0 + q * xi_phys) ** 2

        U_e = np.zeros((n_elem, 8))
        for i in range(8):
            U_e[:, i] = velocity[self.mesh.edofMat_vel[:, i]]

        val_f = np.einsum('ni,ij,nj->n', U_e, self.solver.me_vel, U_e)
        dJf_full = 0.5 * d_alpha_dxi * val_f

        val_visc = np.einsum('ni,ij,nj->n', U_e, self.solver.ke_visc, U_e)
        alpha_e = self.solver._alpha_brinkman(xi_phys)
        Jf_val = 0.5 * np.sum(self.solver.mu * val_visc + alpha_e * val_f)

        # ── 3. Coupled advective adjoint (flow -> temperature) ────────────────
        rhs_F_Q  = np.zeros(self.mesh.n_vel_dofs, dtype=np.float64)
        rhs_F_TV = np.zeros(self.mesh.n_vel_dofs, dtype=np.float64)
        rho_cp = self.solver.rho * self.solver.cp

        for g in range(4):
            N    = self.solver._gp_N[g]
            dNdx = self.solver._gp_dNdx[g]
            dNdy = self.solver._gp_dNdy[g]
            w    = self.solver._gp_detJ[g]

            lambda_Q_star  = np.dot(lambda_T_Q_e, N)
            lambda_TV_star = np.dot(lambda_T_TV_e, N)
            dTdx = np.dot(T_e, dNdx)
            dTdy = np.dot(T_e, dNdy)

            edof_vel = self.mesh.edofMat_vel
            for i in range(4):
                f_u_Q  = rho_cp * w * N[i] * lambda_Q_star  * dTdx
                f_v_Q  = rho_cp * w * N[i] * lambda_Q_star  * dTdy
                f_u_TV = rho_cp * w * N[i] * lambda_TV_star * dTdx
                f_v_TV = rho_cp * w * N[i] * lambda_TV_star * dTdy
                np.add.at(rhs_F_Q,  edof_vel[:, 2 * i],     f_u_Q)
                np.add.at(rhs_F_Q,  edof_vel[:, 2 * i + 1], f_v_Q)
                np.add.at(rhs_F_TV, edof_vel[:, 2 * i],     f_u_TV)
                np.add.at(rhs_F_TV, edof_vel[:, 2 * i + 1], f_v_TV)

        # Stokes adjoint solve
        n_vel_dofs  = self.mesh.n_vel_dofs
        n_pres_dofs = self.mesh.n_pres_dofs
        p_ref = self.mesh.bc.pressure_ref_node + n_vel_dofs

        rhs_F_full_Q = np.zeros(n_vel_dofs + n_pres_dofs, dtype=np.float64)
        rhs_F_full_Q[:n_vel_dofs] = rhs_F_Q
        rhs_F_full_Q[self.solver._vel_fixed] = 0.0
        rhs_F_full_Q[p_ref] = 0.0

        rhs_F_full_TV = np.zeros(n_vel_dofs + n_pres_dofs, dtype=np.float64)
        rhs_F_full_TV[:n_vel_dofs] = rhs_F_TV
        rhs_F_full_TV[self.solver._vel_fixed] = 0.0
        rhs_F_full_TV[p_ref] = 0.0

        csc_S = self.solver._last_stokes_csc.copy()

        lambda_stokes_Q  = spla.spsolve(csc_S, -rhs_F_full_Q)
        lambda_stokes_TV = spla.spsolve(csc_S, -rhs_F_full_TV)

        lambda_U_Q  = lambda_stokes_Q[:n_vel_dofs]
        lambda_U_TV = lambda_stokes_TV[:n_vel_dofs]

        lambda_U_Q_e  = np.zeros((n_elem, 8))
        lambda_U_TV_e = np.zeros((n_elem, 8))
        for i in range(8):
            lambda_U_Q_e[:, i]  = lambda_U_Q[self.mesh.edofMat_vel[:, i]]
            lambda_U_TV_e[:, i] = lambda_U_TV[self.mesh.edofMat_vel[:, i]]

        val_adv_Q  = np.einsum('ni,ij,nj->n', lambda_U_Q_e,  self.solver.me_vel, U_e)
        val_adv_TV = np.einsum('ni,ij,nj->n', lambda_U_TV_e, self.solver.me_vel, U_e)

        dJQ_full  += d_alpha_dxi * val_adv_Q
        dJTV_full += d_alpha_dxi * val_adv_TV

        return dJQ_full, dJf_full, dJTV_full, JQ_val, Jf_val, JTV_val

    # ──────────────────────────────────────────────────────────────────────────
    # Main optimization loop (eps-constraint formulation)
    # ──────────────────────────────────────────────────────────────────────────

    def optimize(self):
        self.iteration = 0
        vf_tol = 0.05  # strict volume fraction tolerance band

        # ── Initial design (top-half variables only) ──────────────────────────
        rng = np.random.default_rng(42)
        # Start at target fluid fraction (= 1 - solid_fraction)
        target_fluid = 1.0 - self.cfg.volume_fraction
        noise = 0.02 * (rng.random(self.n_var) - 0.5)
        x_init = np.clip(target_fluid + noise, 0.01, 0.99)

        # ── Baseline calculation ──────────────────────────────────────────────
        print("Calculating initial baseline for objective normalisation...")
        x_design_full = self.expand_to_full(x_init)
        xi_start = self.mesh.initial_design_field()
        xi_start[self.design_elements] = x_design_full
        U0, P0, T0 = self.solver.solve_forward(xi_start)
        _, _, _, JQ0, Jf0, JTV0 = self.compute_sensitivities(xi_start, U0, T0)

        self.base_JQ  = max(abs(JQ0),  1e-12)
        self.base_Jf  = max(abs(Jf0),  1e-12)
        self.base_JTV = max(abs(JTV0), 1e-12)
        print(f"Baselines -> JQ_0: {self.base_JQ:.3e}, Jf_0: {self.base_Jf:.3e}, JTV_0: {self.base_JTV:.3e}")

        # ── NLopt setup (n_var = top-half only) ───────────────────────────────
        opt = nlopt.opt(nlopt.LD_MMA, self.n_var)
        opt.set_lower_bounds(np.full(self.n_var, 0.001))
        opt.set_upper_bounds(np.full(self.n_var, 1.0))
        opt.set_maxeval(4 * self.cfg.max_opt_iter)
        # Very tight tolerances so NLopt doesn't auto-converge
        opt.set_xtol_rel(1e-12)
        opt.set_ftol_rel(1e-12)

        # ── Per-iteration cache ───────────────────────────────────────────────
        self.cached_x = None
        self.cached_x_phys = None
        self.cached_d_phys = None
        self.cached_JQ  = 0.0
        self.cached_Jf  = 0.0
        self.cached_JTV = 0.0
        self.cached_J_blob = 0.0
        self.cached_grad_JQ     = np.zeros(self.n_var)
        self.cached_grad_Jf     = np.zeros(self.n_var)
        self.cached_grad_JTV    = np.zeros(self.n_var)
        self.cached_grad_J_blob = np.zeros(self.n_var)

        # Volume gradient in reduced space (constant)
        self.cached_grad_vol = np.zeros(self.n_var)

        # Convergence history & stall detection
        obj_history = []
        _CONV_WINDOW = 15
        _CONV_TOL = self.cfg.convergence_tol
        _obj_call_count = [0]
        _MAX_STALLS = 200

        def evaluate_physics(x):
            """Full forward + adjoint with symmetry expansion."""
            if np.any(~np.isfinite(x)):
                return

            # Cache hit
            if self.cached_x is not None and np.allclose(x, self.cached_x, atol=1e-12, rtol=1e-12):
                return

            self.iteration += 1
            if self.iteration > self.cfg.max_opt_iter:
                raise nlopt.ForcedStop()

            _obj_call_count[0] = 0  # reset stall counter

            Da, beta = self.get_continuation_params(self.iteration)
            self.solver.alpha_max = 1.0 / Da
            print(f"  [Iter {self.iteration}] alpha_max = {self.solver.alpha_max:.1e}, beta = {beta:.2f}")

            # ── Symmetry expand -> filter -> project ──────────────────────────
            x_design_full = self.expand_to_full(x)        # (n_design,)
            x_tilde = self.H_filter.dot(x_design_full)    # filtered
            x_phys, d_phys_d_tilde = self.apply_projection(x_tilde, beta)

            xi_full = self.mesh.initial_design_field()
            xi_full[self.design_elements] = x_phys

            # Forward solve
            print(f"  [Iter {self.iteration}] Solving forward physics...")
            tt0 = time.perf_counter()
            U, P, T = self.solver.solve_forward(xi_full)
            print(f"  [Iter {self.iteration}] Forward solve done in {time.perf_counter() - tt0:.2f}s")

            solid_frac = 1.0 - np.mean(x_phys)
            print(f"  [Iter {self.iteration}] solid_frac = {solid_frac:.3f} "
                  f"(target = {self.cfg.volume_fraction:.3f}), "
                  f"x_phys range = [{x_phys.min():.3f}, {x_phys.max():.3f}]")

            if not np.all(np.isfinite(T)) or not np.all(np.isfinite(U)):
                print(f"  [Iter {self.iteration}] WARNING: NaN in physics -- reusing cache")
                self.iteration -= 1
                return

            # Adjoint sensitivities
            print(f"  [Iter {self.iteration}] Computing sensitivities...")
            tt0 = time.perf_counter()
            try:
                dJQ_full, dJf_full, dJTV_full, JQ_val, Jf_val, JTV_val = \
                    self.compute_sensitivities(xi_full, U, T)
            except Exception as e:
                print(f"  [Iter {self.iteration}] WARNING: sensitivity solve failed ({e}) -- reusing cache")
                self.iteration -= 1
                return
            print(f"  [Iter {self.iteration}] Sensitivities done in {time.perf_counter() - tt0:.2f}s")

            if (not np.isfinite(JQ_val) or not np.isfinite(Jf_val) or not np.isfinite(JTV_val) or
                not np.all(np.isfinite(dJQ_full)) or not np.all(np.isfinite(dJf_full))):
                print(f"  [Iter {self.iteration}] WARNING: NaN in objectives/gradients -- reusing cache")
                self.iteration -= 1
                return

            # ── Blob penalty ──────────────────────────────────────────────────
            s = 1.0 - x_phys
            s_filt = self.H_filter.dot(s)
            J_blob = np.sum(s * (s_filt ** 2))
            d_J_blob_dx_phys = -(s_filt ** 2) - 2.0 * self.H_filter.transpose().dot(s * s_filt)

            # ── Chain rule: design-element -> filter -> projection -> half ────
            dJQ_de  = dJQ_full[self.design_elements]
            dJf_de  = dJf_full[self.design_elements]
            dJTV_de = dJTV_full[self.design_elements]

            def chain_rule_half(d_dx_phys):
                """Full chain: d/d(x_half) through projection, filter, and symmetry."""
                # d/d(x_design_full) = H^T @ (d/d(x_phys) * d_phys/d_tilde)
                d_design_full = self.H_filter.transpose().dot(d_dx_phys * d_phys_d_tilde)
                # Sum paired elements for symmetry reduction
                return self.reduce_gradient(d_design_full)

            # Volume gradient (design-element level, positive = more fluid)
            grad_vol_de = np.full(self.n_design, 1.0 / self.n_design)

            # ── Commit to cache ───────────────────────────────────────────────
            self.cached_x = x.copy()
            self.cached_x_phys = x_phys
            self.cached_d_phys = d_phys_d_tilde
            self.cached_JQ  = JQ_val
            self.cached_Jf  = Jf_val
            self.cached_JTV = JTV_val
            self.cached_J_blob = J_blob

            self.cached_grad_JQ     = chain_rule_half(dJQ_de)
            self.cached_grad_Jf     = chain_rule_half(dJf_de)
            self.cached_grad_JTV    = chain_rule_half(dJTV_de)
            self.cached_grad_J_blob = chain_rule_half(d_J_blob_dx_phys)
            self.cached_grad_vol    = chain_rule_half(grad_vol_de)

            # Print summary
            J_total = (JQ_val / self.base_JQ) + self.cfg.w_blob * J_blob
            print(f"Iter {self.iteration:>3d}: J = {J_total:.5e} "
                  f"(JQ = {JQ_val:.3e}, Jf = {Jf_val:.3e}, JTV = {JTV_val:.3e}, "
                  f"Jblob = {J_blob:.3e}, Vf_solid = {solid_frac:.3f}) "
                  f"| Da = {Da:.1e}, beta = {beta:.1f}")

            # ── Convergence check ─────────────────────────────────────────────
            obj_history.append(J_total)
            if len(obj_history) >= _CONV_WINDOW:
                recent = obj_history[-_CONV_WINDOW:]
                rel_change = (max(recent) - min(recent)) / (abs(np.mean(recent)) + 1e-12)
                if rel_change < _CONV_TOL and beta >= self.cfg.filter_beta_max * 0.9:
                    print(f"  Converged: rel_change = {rel_change:.2e}. Stopping.")
                    raise nlopt.ForcedStop()

        # ── Objective: minimise JQ/base_JQ + w_blob * J_blob ─────────────────
        def objective(x, grad):
            _obj_call_count[0] += 1
            if _obj_call_count[0] > _MAX_STALLS:
                print(f"  MMA stall detected ({_obj_call_count[0]} calls). Stopping.")
                raise nlopt.ForcedStop()

            evaluate_physics(x)
            J_total = (self.cached_JQ / self.base_JQ) + self.cfg.w_blob * self.cached_J_blob
            if grad.size > 0:
                grad[:] = (self.cached_grad_JQ / self.base_JQ) + self.cfg.w_blob * self.cached_grad_J_blob
            return float(J_total)

        # ── Volume constraint UPPER: fluid_frac <= (1 - vf) + vf_tol ─────────
        def volume_constraint_upper(x, grad):
            evaluate_physics(x)
            if self.cached_x_phys is None:
                if grad.size > 0:
                    grad[:] = 0.0
                return 0.0
            current_fluid = np.mean(self.cached_x_phys)
            if grad.size > 0:
                grad[:] = self.cached_grad_vol
            target_upper = (1.0 - self.cfg.volume_fraction) + vf_tol
            return float(current_fluid - target_upper)

        # ── Volume constraint LOWER: fluid_frac >= (1 - vf) - vf_tol ─────────
        def volume_constraint_lower(x, grad):
            """Enforces: (1-vf) - vf_tol - current_fluid <= 0"""
            evaluate_physics(x)
            if self.cached_x_phys is None:
                if grad.size > 0:
                    grad[:] = 0.0
                return 0.0
            current_fluid = np.mean(self.cached_x_phys)
            if grad.size > 0:
                grad[:] = -self.cached_grad_vol
            target_lower = (1.0 - self.cfg.volume_fraction) - vf_tol
            return float(target_lower - current_fluid)

        # ── Flow dissipation constraint ──────────────────────────────────────
        def flow_constraint(x, grad):
            evaluate_physics(x)
            if grad.size > 0:
                grad[:] = self.cached_grad_Jf / self.base_Jf
            return float((self.cached_Jf / self.base_Jf) - self.cfg.eps_flow)

        # ── Temperature variance constraint ──────────────────────────────────
        def tv_constraint(x, grad):
            evaluate_physics(x)
            if grad.size > 0:
                grad[:] = self.cached_grad_JTV / self.base_JTV
            return float((self.cached_JTV / self.base_JTV) - self.cfg.eps_tv)

        # ── Register with NLopt ───────────────────────────────────────────────
        opt.set_min_objective(objective)
        opt.add_inequality_constraint(volume_constraint_upper, 1e-4)
        opt.add_inequality_constraint(volume_constraint_lower, 1e-4)
        opt.add_inequality_constraint(flow_constraint, 1e-3)
        opt.add_inequality_constraint(tv_constraint, 1e-3)

        # ── Execute ───────────────────────────────────────────────────────────
        print(f"Starting MMA (eps-constraint), {self.n_var} vars (half-plane symmetry)...")
        try:
            x_opt = opt.optimize(x_init)
            print("Optimization Complete.")
        except nlopt.ForcedStop:
            print(f"Optimization stopped at iteration {self.iteration} "
                  f"(cap: {self.cfg.max_opt_iter}).")
            x_opt = self.cached_x.copy() if self.cached_x is not None else x_init.copy()
        except nlopt.RoundoffLimited:
            print("Optimization halted (roundoff limited / converged).")
            x_opt = self.cached_x.copy() if self.cached_x is not None else x_init.copy()
        except Exception as e:
            print(f"Optimization terminated with error: {e}")
            x_opt = self.cached_x.copy() if self.cached_x is not None else x_init.copy()

        if np.isscalar(x_opt):
            x_opt = self.cached_x.copy() if self.cached_x is not None else x_init.copy()

        return x_opt