"""Sharp-interface geometry screening for three-particle no-sink loading.

This module contains geometry and metrology only.  It deliberately has no
phase-field operator, time integrator, stochastic clock, or source-event code.

The center particle occupies a fixed axial interval of length ``Lc``.  Its
axisymmetric cross-sectional squared radius ``g(z)=r(z)**2`` follows

    g(z, x) = g(z, 0) - x Vc0 / (pi Lc).

Consequently its volume is exactly ``Vc0*(1-x)``, both grain-boundary planes
remain fixed, and the contact radius, slope, angle, and meridional curvature
are analytic.  The two outer particles receive ``x*Vc0/2`` each.  A quintic
particle-plus-neck profile is reconstructed for each outer body with the same
contact jet and the required volume.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SharpDesign:
    Ro_m: float
    Rc_over_Ro: float
    Lc_m: float
    rTJ_over_Rc: float
    theta_center_deg: float
    theta_outer_deg: float
    kRc_center: float
    kRo_outer: float
    gamma_s: float = 1.0
    W_m: float = 4e-9

    @property
    def Rc_m(self) -> float:
        return self.Ro_m * self.Rc_over_Ro

    @property
    def rTJ0_m(self) -> float:
        return self.Rc_m * self.rTJ_over_Rc

    @property
    def Vc0_m3(self) -> float:
        return 4.0 * math.pi * self.Rc_m**3 / 3.0

    @property
    def Vo0_m3(self) -> float:
        return 4.0 * math.pi * self.Ro_m**3 / 3.0

    @property
    def k_center0_per_m(self) -> float:
        return self.kRc_center / self.Rc_m

    @property
    def k_outer0_per_m(self) -> float:
        return self.kRo_outer / self.Ro_m


def contact_slope(theta_deg: float) -> float:
    """Magnitude of dr/dz for the production angle convention."""
    theta = math.radians(theta_deg)
    if not 0.0 < theta < math.pi:
        raise ValueError("contact angle must lie between zero and 180 degrees")
    return 1.0 / math.tan(0.5 * theta)


def _initial_g_jet(radius: float, theta_deg: float, curvature: float,
                   slope_sign: float) -> tuple[float, float]:
    slope = slope_sign * contact_slope(theta_deg)
    second = -curvature * (1.0 + slope*slope)**1.5
    return 2.0*radius*slope, 2.0*(slope*slope + radius*second)


def center_squared_radius_coefficients(design: SharpDesign) -> np.ndarray:
    """Even sextic coefficients in u=2z/Lc, including exact volume."""
    r = design.rTJ0_m
    a = 0.5*design.Lc_m
    gp, gpp = _initial_g_jet(
        r, design.theta_center_deg, design.k_center0_per_m, -1.0)
    matrix = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 2.0, 4.0, 6.0],
        [0.0, 2.0, 12.0, 30.0],
        [1.0, 1.0/3.0, 1.0/5.0, 1.0/7.0],
    ])
    rhs = np.array([
        r*r, gp*a, gpp*a*a,
        design.Vc0_m3/(math.pi*design.Lc_m),
    ])
    return np.linalg.solve(matrix, rhs)


def evaluate_even_squared_radius(coeff: np.ndarray, u: np.ndarray,
                                 derivative: int = 0) -> np.ndarray:
    powers = np.array([0, 2, 4, 6], dtype=int)
    out = np.zeros_like(np.asarray(u, dtype=float))
    for c, p in zip(coeff, powers):
        if derivative == 0:
            out += c*np.asarray(u)**p
        elif derivative == 1 and p >= 1:
            out += c*p*np.asarray(u)**(p-1)
        elif derivative == 2 and p >= 2:
            out += c*p*(p-1)*np.asarray(u)**(p-2)
    return out


def outer_squared_radius_coefficients(
        design: SharpDesign, x: float) -> tuple[np.ndarray, float]:
    """Quintic squared-radius profile from one GB plane to the outer tip."""
    state = contact_state(design, x)
    r = state["r_TJ_m"]
    volume = design.Vo0_m3 + 0.5*x*design.Vc0_m3
    Req = (3.0*volume/(4.0*math.pi))**(1.0/3.0)
    length = 2.0*Req
    gp0, gpp0 = _initial_g_jet(
        design.rTJ0_m, design.theta_outer_deg,
        design.k_outer0_per_m, +1.0)
    coeff = np.zeros(6)
    coeff[0] = r*r
    coeff[1] = gp0*length
    coeff[2] = 0.5*gpp0*length*length
    # Unknown cubic through quintic terms impose tip closure, sphere-like tip
    # slope, and exact volume.
    matrix = np.array([
        [1.0, 1.0, 1.0],
        [3.0, 4.0, 5.0],
        [1.0/4.0, 1.0/5.0, 1.0/6.0],
    ])
    known_tip = coeff[0] + coeff[1] + coeff[2]
    known_slope = coeff[1] + 2.0*coeff[2]
    known_integral = coeff[0] + coeff[1]/2.0 + coeff[2]/3.0
    rhs = np.array([
        -known_tip,
        -2.0*Req*length-known_slope,
        volume/(math.pi*length)-known_integral,
    ])
    coeff[3:] = np.linalg.solve(matrix, rhs)
    return coeff, length


def evaluate_polynomial(coeff: np.ndarray, u: np.ndarray,
                        derivative: int = 0) -> np.ndarray:
    poly = np.polynomial.Polynomial(coeff)
    return poly.deriv(derivative)(u)


def _evolved_local_geometry(r0: float, r: float, theta0_deg: float,
                            curvature0: float, slope_sign: float) -> Mapping[str, float]:
    gp, gpp = _initial_g_jet(r0, theta0_deg, curvature0, slope_sign)
    slope = gp/(2.0*r)
    second = gpp/(2.0*r) - gp*gp/(4.0*r**3)
    curvature = -second/(1.0+slope*slope)**1.5
    theta = 2.0*math.atan2(1.0, abs(slope))
    return {"slope": slope, "curvature_per_m": curvature,
            "theta_rad": theta, "theta_deg": math.degrees(theta)}


def contact_state(design: SharpDesign, x: float) -> dict[str, float]:
    if not 0.0 <= x < 1.0:
        raise ValueError("center-volume loss x must lie in [0,1)")
    decrement = x*design.Vc0_m3/(math.pi*design.Lc_m)
    r2 = design.rTJ0_m**2-decrement
    if r2 <= 0.0:
        raise ValueError("contact radius has reached topology loss")
    r = math.sqrt(r2)
    center = _evolved_local_geometry(
        design.rTJ0_m, r, design.theta_center_deg,
        design.k_center0_per_m, -1.0)
    outer = _evolved_local_geometry(
        design.rTJ0_m, r, design.theta_outer_deg,
        design.k_outer0_per_m, +1.0)
    mean_k = 0.5*(center["curvature_per_m"]+outer["curvature_per_m"])
    sine_sum = math.sin(0.5*center["theta_rad"])+math.sin(0.5*outer["theta_rad"])
    sigma_k = -design.gamma_s*mean_k
    sigma_tj = 1.5*design.gamma_s*sine_sum/r
    sigma_center = design.gamma_s*(-center["curvature_per_m"]
        + 3.0*math.sin(0.5*center["theta_rad"])/r)
    sigma_outer = design.gamma_s*(-outer["curvature_per_m"]
        + 3.0*math.sin(0.5*outer["theta_rad"])/r)
    return {
        "x": x, "r_TJ_m": r, "neck_radius_m": r,
        "center_curvature_per_m": center["curvature_per_m"],
        "outer_curvature_per_m": outer["curvature_per_m"],
        "center_theta_deg": center["theta_deg"],
        "outer_theta_deg": outer["theta_deg"],
        "sigma_center_side_Pa": sigma_center,
        "sigma_outer_side_Pa": sigma_outer,
        "sigma_kappa_Pa": sigma_k, "sigma_TJ_Pa": sigma_tj,
        "sigma_GB_Pa": sigma_k+sigma_tj,
        "Vc_m3": design.Vc0_m3*(1.0-x),
        "Vo_m3": design.Vo0_m3+0.5*x*design.Vc0_m3,
    }


def admissibility(design: SharpDesign, x_max: float = 0.20,
                  samples: int = 401) -> dict[str, float | bool | str]:
    """Check positivity, simple topology, and conservative W=4 resolution."""
    reasons: list[str] = []
    u = np.linspace(0.0, 1.0, samples)
    try:
        center_coeff = center_squared_radius_coefficients(design)
    except np.linalg.LinAlgError:
        return {"admissible": False, "reason": "singular center profile"}
    g0 = evaluate_even_squared_radius(center_coeff, u)
    dg0 = evaluate_even_squared_radius(center_coeff, u, 1)
    if np.min(g0) <= 0.0:
        reasons.append("negative center squared radius")
    if np.max(dg0) > max(g0)*1e-7:
        reasons.append("center profile is not monotone from midplane to GB")
    try:
        terminal = contact_state(design, x_max)
    except ValueError as exc:
        reasons.append(str(exc))
        terminal = None
    if terminal is not None:
        decrement = x_max*design.Vc0_m3/(math.pi*design.Lc_m)
        if np.min(g0-decrement) <= 0.0:
            reasons.append("center topology fails before requested loss")
        for x in (0.0, x_max):
            coeff, _ = outer_squared_radius_coefficients(design, x)
            go = evaluate_polynomial(coeff, u)
            dgo = evaluate_polynomial(coeff, u, 1)
            if np.min(go[:-1]) < -1e-8*np.max(abs(go)):
                reasons.append("negative outer squared radius")
                break
            signs = np.sign(dgo[1:-1])
            changes = np.count_nonzero(signs[1:]*signs[:-1] < 0)
            if changes > 1:
                reasons.append("outer profile has multiple radial extrema")
                break
        rmin = terminal["r_TJ_m"]
        if rmin < 6.0*design.W_m:
            reasons.append("TJ radius below 6W")
    if design.Lc_m < 8.0*design.W_m:
        reasons.append("fixed GB separation below 8W")
    if design.rTJ0_m >= 1.25*design.Ro_m:
        reasons.append("contact exceeds outer-body support")
    r_terminal = terminal["r_TJ_m"] if terminal else 0.0
    margin = min(design.Lc_m/(8.0*design.W_m),
                 r_terminal/(6.0*design.W_m) if r_terminal else 0.0)
    return {"admissible": not reasons, "reason": "; ".join(dict.fromkeys(reasons)),
            "resolution_margin": margin,
            "rTJ_terminal_over_W": r_terminal/design.W_m if terminal else 0.0,
            "center_min_radius_over_W": math.sqrt(max(0.0, np.min(g0-
                x_max*design.Vc0_m3/(math.pi*design.Lc_m))))/design.W_m}


def inverse_target_radius(target_stress_pa: float, mean_curvature_per_m: float,
                          theta_negative_deg: float, theta_positive_deg: float,
                          gamma_s: float = 1.0) -> float:
    sine_sum = (math.sin(math.radians(theta_negative_deg)/2.0)
                + math.sin(math.radians(theta_positive_deg)/2.0))
    denominator = target_stress_pa + gamma_s*mean_curvature_per_m
    if denominator <= 0.0:
        raise ValueError("target and curvature give a non-positive denominator")
    return 1.5*gamma_s*sine_sum/denominator


def half_chain_profile(design: SharpDesign, x: float,
                       points_per_branch: int = 801) -> dict[str, np.ndarray]:
    """Return center-midplane -> right TJ -> outer-tip meridional profiles."""
    if points_per_branch < 21:
        raise ValueError("at least 21 points per branch are required")
    u=np.linspace(0.0,1.0,points_per_branch)
    center_coeff=center_squared_radius_coefficients(design).copy()
    center_coeff[0]-=x*design.Vc0_m3/(math.pi*design.Lc_m)
    center_r=np.sqrt(np.maximum(0.0,evaluate_even_squared_radius(center_coeff,u)))
    center_z=0.5*design.Lc_m*u
    outer_coeff,length=outer_squared_radius_coefficients(design,x)
    outer_r=np.sqrt(np.maximum(0.0,evaluate_polynomial(outer_coeff,u)))
    outer_z=0.5*design.Lc_m+length*u
    return {"center_parameter":u,"center_z_m":center_z,"center_r_m":center_r,
            "outer_parameter":u,"outer_z_m":outer_z,"outer_r_m":outer_r}


def _surface_area(z: np.ndarray, r: np.ndarray) -> float:
    ds=np.hypot(np.diff(z),np.diff(r))
    return float(2.0*math.pi*np.sum(0.5*(r[:-1]+r[1:])*ds))


def sharp_free_energy(design: SharpDesign, x: float, gamma_gb: float,
                      points_per_branch: int = 801) -> dict[str, float]:
    """Surface plus two fixed-plane GB disks for the full symmetric chain."""
    p=half_chain_profile(design,x,points_per_branch)
    area_center=2.0*_surface_area(p["center_z_m"],p["center_r_m"])
    area_outer=2.0*_surface_area(p["outer_z_m"],p["outer_r_m"])
    r=contact_state(design,x)["r_TJ_m"]
    area_gb=2.0*math.pi*r*r
    return {"surface_area_m2":area_center+area_outer,"GB_area_m2":area_gb,
            "surface_energy_J":design.gamma_s*(area_center+area_outer),
            "GB_energy_J":gamma_gb*area_gb,
            "free_energy_J":design.gamma_s*(area_center+area_outer)+gamma_gb*area_gb}


def _branch_kinetic_terms(z: np.ndarray, r: np.ndarray,
                          zm: np.ndarray, rm: np.ndarray,
                          zp: np.ndarray, rp: np.ndarray, dx: float,
                          Q_start_m3: float, mobility_m6_per_J_s: float
                          ) -> tuple[float,float,float]:
    """Return friction, terminal r*j, and absolute-volume velocity scale."""
    dzdu=np.gradient(z);drdu=np.gradient(r)
    norm=np.hypot(dzdu,drdu)
    # Either normal orientation is valid because the friction is quadratic;
    # use the outward meridional normal for a z-increasing upper profile.
    nz=-drdu/norm;nr=dzdu/norm
    dzdx=(zp-zm)/(2.0*dx);drdx=(rp-rm)/(2.0*dx)
    un=dzdx*nz+drdx*nr
    ds=np.hypot(np.diff(z),np.diff(r))
    integrand=r*un
    increments=-0.5*(integrand[:-1]+integrand[1:])*ds
    Q=np.r_[Q_start_m3,Q_start_m3+np.cumsum(increments)]
    rmid=0.5*(r[:-1]+r[1:]);Qmid=0.5*(Q[:-1]+Q[1:])
    flux=Qmid/rmid
    friction=float(np.sum(2.0*math.pi*rmid*flux*flux*ds/mobility_m6_per_J_s))
    volume_scale=float(np.sum(2.0*math.pi*0.5*(integrand[:-1]+integrand[1:])*ds))
    return friction,float(Q[-1]),volume_scale


def surface_diffusion_projection(
        design: SharpDesign, x: float, *, gamma_gb: float,
        mobility_m6_per_J_s: float, derivative_step: float = 1e-5,
        points_per_branch: int = 801) -> dict[str, float]:
    """Onsager projection for the single no-sink loading coordinate ``x``.

    ``mobility_m6_per_J_s`` is the PF sharp-surface volume mobility.  In the
    atomic-flux notation it equals ``Omega**2*M_atom``; the explicit atomic
    volume therefore cancels from the generalized friction.
    """
    if mobility_m6_per_J_s <= 0.0:
        raise ValueError("surface mobility must be positive")
    h=min(derivative_step,0.49*x if x>0 else derivative_step,
          0.49*(1.0-x))
    if x-h < 0.0:
        x0,x1,x_eval=x,x+2*h,x+h
        # Evaluate the kinetic metric at the midpoint and report it at x=0;
        # this is a second-order one-sided estimate.
    else:
        x0,x1,x_eval=x-h,x+h,x
    base=half_chain_profile(design,x_eval,points_per_branch)
    minus=half_chain_profile(design,x0,points_per_branch)
    plus=half_chain_profile(design,x1,points_per_branch)
    span=x1-x0
    fc,Q,vc=_branch_kinetic_terms(
        base["center_z_m"],base["center_r_m"],
        minus["center_z_m"],minus["center_r_m"],plus["center_z_m"],plus["center_r_m"],
        0.5*span,0.0,mobility_m6_per_J_s)
    fo,Qend,vo=_branch_kinetic_terms(
        base["outer_z_m"],base["outer_r_m"],
        minus["outer_z_m"],minus["outer_r_m"],plus["outer_z_m"],plus["outer_r_m"],
        0.5*span,Q,mobility_m6_per_J_s)
    em=sharp_free_energy(design,x0,gamma_gb,points_per_branch)["free_energy_J"]
    ep=sharp_free_energy(design,x1,gamma_gb,points_per_branch)["free_energy_J"]
    dFdx=(ep-em)/span
    friction=2.0*(fc+fo)
    xdot=-dFdx/friction
    volume_rate_residual=vc+vo
    scale=max(abs(vc),abs(vo),design.Vc0_m3)
    return {"x":x,"evaluation_x":x_eval,"free_energy_J":sharp_free_energy(
        design,x,gamma_gb,points_per_branch)["free_energy_J"],
        "dF_dx_J":dFdx,"minus_dF_dx_J":-dFdx,
        "zeta_J_s":friction,"xdot_per_s":xdot,
        "terminal_r_times_volume_flux_m3":Qend,
        "half_chain_volume_rate_residual_m3":volume_rate_residual,
        "relative_half_chain_closure":abs(volume_rate_residual)/scale}
