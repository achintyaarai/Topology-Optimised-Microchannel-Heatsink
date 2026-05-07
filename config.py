"""
config.py — Microchannel Heatsink Topology Optimizer
=====================================================
Centralizes all material properties, dimensionless numbers,
geometry proportions, and hardware-specific solver tolerances.

No executable heavy logic lives here — only validated data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Material databases (SI units throughout)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AluminiumProps:
    """Thermophysical properties of Aluminium (pure, 20 °C)."""
    # Thermal
    thermal_conductivity: float = 237.0      # k  [W m⁻¹ K⁻¹]
    specific_heat: float = 897.0             # cₚ [J kg⁻¹ K⁻¹]
    density: float = 2700.0                  # ρ  [kg m⁻³]

    # Mechanical
    youngs_modulus: float = 70.0e9           # E  [Pa]
    poisson_ratio: float = 0.33             # ν  [—]

    # Derived
    @property
    def thermal_diffusivity(self) -> float:
        """α = k / (ρ cₚ)  [m² s⁻¹]"""
        return self.thermal_conductivity / (self.density * self.specific_heat)


@dataclass(frozen=True)
class WaterProps:
    """Thermophysical properties of liquid Water (20 °C, 1 atm)."""
    # Thermal
    thermal_conductivity: float = 0.598      # k  [W m⁻¹ K⁻¹]
    specific_heat: float = 4182.0            # cₚ [J kg⁻¹ K⁻¹]
    density: float = 998.2                   # ρ  [kg m⁻³]

    # Viscous
    dynamic_viscosity: float = 1.002e-3      # μ  [Pa s]

    # Derived
    @property
    def kinematic_viscosity(self) -> float:
        """ν = μ / ρ  [m² s⁻¹]"""
        return self.dynamic_viscosity / self.density

    @property
    def thermal_diffusivity(self) -> float:
        """α = k / (ρ cₚ)  [m² s⁻¹]"""
        return self.thermal_conductivity / (self.density * self.specific_heat)

    @property
    def prandtl_number(self) -> float:
        """Pr = ν / α  [—]"""
        return self.kinematic_viscosity / self.thermal_diffusivity


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """
    All dimensionless numbers, material property ratios, and
    optimization hyper-parameters for the topology optimizer.

    Includes the ε-constraint formulation parameters from
    Wang et al. (2025) for fractal microchannel generation.
    """

    # -- Material properties (immutable) ------------------------------------
    aluminium: AluminiumProps = field(default_factory=AluminiumProps)
    water: WaterProps = field(default_factory=WaterProps)

    # -- Dimensionless numbers ----------------------------------------------
    reynolds_number: float = 800.0           # Re  [—]
    peclet_number: float = 685.8             # Pe = Re · Pr_water(20°C)
    conductivity_ratio: float = 396.7        # κ  = k_Al / k_water  [—]
    diffusivity_ratio: float = 82.4          # α_Al / α_water  [—]

    # -- Solver Parameters --------------------------------------------------
    volumetric_heat_source: float = 5.0e6    # [W/m³]
    k_penalty_power: float = 3.0             # SIMP-style k-interpolation power

    # -- ε-constraint formulation (Wang et al. 2025) -------------------------
    # Primary objective: minimise JQ (p-norm peak temperature) + w_blob*J_blob
    # Constraint 1:  Jf  / base_Jf   ≤  eps_flow   (flow dissipation budget)
    # Constraint 2:  JTV / base_JTV  ≤  eps_tv      (temperature-variance budget)
    eps_flow: float = 1.5    # Allowed flow-dissipation multiplier vs initial baseline
    eps_tv:   float = 2.0    # Allowed temp-variance multiplier vs initial baseline
    w_blob:   float = 0.1    # Solid-agglomeration blob penalty weight

    # -- Adjoint objective parameters ----------------------------------------
    p_norm_exponent: float = 8.0             # P-norm hotspot penalty exponent
    t_clip_floor: float = 1e-6               # Floor for temperature clipping

    # -- MMA optimisation bounds --------------------------------------------
    mma_move_limit: float = 0.20             # Δx_max per iteration
    mma_asymptote_init: float = 0.50         # s₀
    mma_asymptote_decrease: float = 0.65     # γ⁻
    mma_asymptote_increase: float = 1.05     # γ⁺
    volume_fraction: float = 0.40            # V_f  [—]

    # -- Density / Heaviside filter -----------------------------------------
    filter_radius: float = 1.5e-3            # r_min [m]  (≈ 3 element widths)
    filter_beta_init: float = 1.0            # β₀
    filter_beta_max: float = 256.0           # β∞ (near-perfect binary projection)
    filter_beta_scale: float = 2.0           # continuation doubling factor

    # -- Iterative Krylov solver tolerances ---------------------------------
    krylov_tol: float = 1e-5
    krylov_atol: float = 1e-10
    krylov_max_iter: int = 500
    krylov_restart: int = 50

    # -- Outer optimisation loop --------------------------------------------
    max_opt_iter: int = 300
    convergence_tol: float = 1e-4

    # -- Post-construction validation ---------------------------------------
    def __post_init__(self) -> None:
        if not (0.0 < self.volume_fraction <= 1.0):
            raise ValueError(
                f"volume_fraction must be in (0, 1], got {self.volume_fraction}"
            )
        if self.reynolds_number <= 0:
            raise ValueError(f"Reynolds number must be positive, got {self.reynolds_number}")
        if self.filter_radius <= 0:
            raise ValueError(f"filter_radius must be positive, got {self.filter_radius}")
        if self.filter_beta_init >= self.filter_beta_max:
            raise ValueError(
                "filter_beta_init must be strictly less than filter_beta_max"
            )
        # Recompute Pe from Re if it was left at default (allow user override)
        object.__setattr__(self, "_pe_auto", self.reynolds_number * self.water.prandtl_number)

    @property
    def peclet_auto(self) -> float:
        """Pe recomputed from Re · Pr_water; use for cross-checking."""
        return self.reynolds_number * self.water.prandtl_number

    @property
    def conductivity_ratio_computed(self) -> float:
        """k_Al / k_water recomputed from current material records."""
        return self.aluminium.thermal_conductivity / self.water.thermal_conductivity

    @property
    def diffusivity_ratio_computed(self) -> float:
        """α_Al / α_water recomputed from current material records."""
        return self.aluminium.thermal_diffusivity / self.water.thermal_diffusivity


# ---------------------------------------------------------------------------
# Geometry configuration — pipe nozzle, no funnels
# ---------------------------------------------------------------------------

@dataclass
class GeoConfig:
    """
    Defines the 2-D domain geometry for the microchannel heatsink.

    Domain layout (x → flow direction, y → height):

        ┌──────────┬───────────────┬──────────┐
        │  L_pipe   │   L_square    │  L_pipe   │
        │ (inlet)   │ (topo region) │ (outlet)  │
        └──────────┴───────────────┴──────────┘
         ←————————— L_total ——————————→

    The pipe regions have a narrow aperture W_pipe centred at y = H/2.
    Elements outside the pipe aperture are INACTIVE (wall).
    The central square spans the full height H and is FREE (topology zone).
    No funnels — abrupt expansion from pipe to heatsink.

    Attributes
    ----------
    L_total : float   Total domain length             [m]
    H       : float   Total domain height             [m]
    L_pipe  : float   Inlet / outlet pipe length      [m]
    W_pipe  : float   Inlet / outlet pipe width       [m]
    L_square: float   Central square side length      [m]  (topology zone)
    nx      : int     Number of elements in x         [—]
    ny      : int     Number of elements in y         [—]
    """

    # -- Primary proportions ------------------------------------------------
    L_total: float = 80.0e-3              # 80 mm total length
    H: float = 40.0e-3                     # 40 mm full channel height
    L_pipe: float = 20.0e-3               # 20 mm inlet/outlet stub
    W_pipe: float = 10.0e-3               # 10 mm pipe (nozzle) width
    L_square: float = 40.0e-3             # 40 mm optimizable square

    # -- Mesh resolution ----------------------------------------------------
    nx: int = 160                           # elements along x
    ny: int = 80                            # elements along y

    def __post_init__(self) -> None:
        self._validate_proportions()

    # -- Derived geometry (read-only properties) ----------------------------

    @property
    def element_size_x(self) -> float:
        """Uniform element width Δx = L_total / nx  [m]."""
        return self.L_total / self.nx

    @property
    def element_size_y(self) -> float:
        """Uniform element height Δy = H / ny  [m]."""
        return self.H / self.ny

    @property
    def aspect_ratio(self) -> float:
        """Element aspect ratio Δx / Δy (should be close to 1)."""
        return self.element_size_x / self.element_size_y

    @property
    def x_square_start(self) -> float:
        """x-coordinate where the optimizable zone begins."""
        return self.L_pipe

    @property
    def x_square_end(self) -> float:
        """x-coordinate where the optimizable zone ends."""
        return self.L_pipe + self.L_square

    @property
    def square_bounds(self) -> Tuple[float, float, float, float]:
        """
        Axis-aligned bounding box of the topology optimisation zone.

        Returns
        -------
        (x_min, x_max, y_min, y_max) in metres.
        """
        return (
            self.x_square_start,      # x_min
            self.x_square_end,        # x_max
            0.0,                      # y_min (bottom wall)
            self.H,                   # y_max (top wall)
        )

    @property
    def n_design_elements(self) -> int:
        """
        Approximate number of finite elements inside the topology zone.
        """
        frac_x = self.L_square / self.L_total
        frac_y = 1.0
        return round(self.nx * frac_x * self.ny * frac_y)

    @property
    def hydraulic_diameter_pipe(self) -> float:
        """
        Hydraulic diameter of the rectangular inlet/outlet pipe (2-D proxy).
        Dₕ = 2 · W_pipe for a channel of width W_pipe [m].
        """
        return 2.0 * self.W_pipe

    # -- Internal validation ------------------------------------------------

    def _validate_proportions(self) -> None:
        """Ensure geometry segments are self-consistent."""
        assembled = 2 * self.L_pipe + self.L_square
        if abs(assembled - self.L_total) > 1e-12:
            raise ValueError(
                f"Geometry segments do not sum to L_total.\n"
                f"  2·L_pipe + L_square = {assembled * 1e6:.1f} µm\n"
                f"  L_total             = {self.L_total * 1e6:.1f} µm\n"
                "Adjust one of the segment lengths."
            )
        if self.W_pipe >= self.H:
            raise ValueError(
                f"W_pipe ({self.W_pipe*1e6:.1f} µm) must be smaller than H "
                f"({self.H*1e6:.1f} µm) so the pipe is a genuine nozzle."
            )
        for name, val in [
            ("L_pipe", self.L_pipe),
            ("W_pipe", self.W_pipe),
            ("L_square", self.L_square),
            ("H", self.H),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}.")
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive integers.")

    def summary(self) -> str:
        """Return a human-readable geometry summary."""
        x0, x1, y0, y1 = self.square_bounds
        lines = [
            "GeoConfig -- Microchannel Heatsink Domain (pipe nozzle, no funnels)",
            "=" * 66,
            f"  L_total          : {self.L_total*1e6:>8.1f} um",
            f"  H                : {self.H*1e6:>8.1f} um",
            f"  L_pipe           : {self.L_pipe*1e6:>8.1f} um",
            f"  W_pipe           : {self.W_pipe*1e6:>8.1f} um",
            f"  L_square         : {self.L_square*1e6:>8.1f} um",
            f"  Mesh (nx x ny)   : {self.nx} x {self.ny}",
            f"  dx               : {self.element_size_x*1e6:>8.2f} um",
            f"  dy               : {self.element_size_y*1e6:>8.2f} um",
            f"  Aspect ratio     : {self.aspect_ratio:>8.3f}",
            f"  Design elements  : {self.n_design_elements:>8d}",
            f"  Topo zone x      : [{x0*1e6:.1f}, {x1*1e6:.1f}] um",
            f"  Topo zone y      : [{y0*1e6:.1f}, {y1*1e6:.1f}] um",
            f"  Dh (pipe)        : {self.hydraulic_diameter_pipe*1e6:>8.1f} um",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: ready-to-use singleton defaults
# ---------------------------------------------------------------------------

DEFAULT_SIM  = SimConfig()
DEFAULT_GEO  = GeoConfig()


# ---------------------------------------------------------------------------
# Quick self-test (python config.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sim = DEFAULT_SIM
    geo = DEFAULT_GEO

    print("SimConfig -- Key quantities")
    print("=" * 44)
    print(f"  Re               : {sim.reynolds_number}")
    print(f"  Pe (stored)      : {sim.peclet_number}")
    print(f"  Pe (Re*Pr)       : {sim.peclet_auto:.2f}")
    print(f"  k_Al / k_water   : {sim.conductivity_ratio_computed:.1f}  "
          f"(stored: {sim.conductivity_ratio})")
    print(f"  a_Al / a_water   : {sim.diffusivity_ratio_computed:.1f}  "
          f"(stored: {sim.diffusivity_ratio})")
    print(f"  P-norm exponent  : {sim.p_norm_exponent}")
    print(f"  Filter radius    : {sim.filter_radius*1e6:.1f} um")
    print(f"  Volume fraction  : {sim.volume_fraction}")
    print(f"  eps-flow         : {sim.eps_flow}")
    print(f"  eps-TV           : {sim.eps_tv}")
    print(f"  w_blob           : {sim.w_blob}")
    print()
    print(geo.summary())