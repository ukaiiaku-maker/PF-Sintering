"""Milestone-1 directional-state diagnostic for the coarsening-stress campaign.

This module is read-only with respect to physics: it inspects `f`/`eta`/`Sink`/
`Stress` state and reports the quantities needed to classify the sign of the
coarsening -> local-stress response over a short trajectory (see
CODEX_PHYSICS_SEQUENCE.md, Milestone 1). It does not alter dynamics.

Quantity definitions (documented here because they are diagnostic conventions,
not existing production quantities):

- ``separation`` (L): distance from the shrinking particle's x-centroid
  (``center(e2, p)``) to the fixed substrate-wall reference
  ``wall_x0 = (substrate_wall_frac - 0.5) * Nx * dx``. ``wall_x0`` is a
  geometric constant fixed by the run configuration (the same constant
  ``compute_stress`` uses for its "particle burrowed into substrate" check),
  not a field-derived quantity, so it cannot itself drift.
- ``x_neck`` (neck/contact width): ``Stress.x_neck`` from
  ``model.compute_stress`` (unchanged, existing rigorous neck measure).
- ``A_GB`` (GB/contact measure): ``sum(4*e1*e2)*dx**2``, the same ``4*e1*e2``
  weighting ``model.contact_width`` uses, integrated over the domain so it has
  units of area. This is a load-bearing-contact proxy independent of the
  single-column neck-width measure.
- ``sigma`` (local sintering stress) and its decomposition ``sigma_lt``
  (line-tension/Cahn-Hoffman neck-force term) and ``sigma_curv``
  (``gamma_s * kappa`` curvature term): ``Stress.sigma`` /
  ``Stress.sigma_lt`` / ``Stress.sigma_curv`` from ``compute_stress``,
  unchanged.
- ``E_surf`` (total free-surface energy): ``gamma_s`` times a total-variation
  estimate of the ``f=0.5`` interface length, ``gamma_s * sum(|grad f|) * dx**2``.
- ``E_gb`` (total GB energy): effective GB energy density times the GB/contact
  area measure, ``effective_gamma(s, p) * A_GB``.
- ``G_interface`` (total interfacial energy): ``E_surf + E_gb``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .model import Sink, Stress, center, effective_gamma


def wall_x0(p) -> float:
    """Fixed substrate-wall x-reference (geometric constant, not field-derived)."""
    return (p.substrate_wall_frac - 0.5) * p.Nx * p.dx


def contact_area(e1, e2, p) -> float:
    """GB/contact measure: domain integral of the same 4*e1*e2 weighting used
    by ``model.contact_width``. Units of area (m^2 in physical presets)."""
    return float(np.sum(4.0 * e1 * e2)) * p.dx * p.dx


def _grad_mag(f, p) -> np.ndarray:
    gx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * p.dx)
    gy = np.zeros_like(f)
    gy[1:-1] = (f[2:] - f[:-2]) / (2 * p.dx)
    gy[0] = (f[1] - f[0]) / p.dx
    gy[-1] = (f[-1] - f[-2]) / p.dx
    return np.hypot(gx, gy)


def surface_energy(f, p) -> float:
    """Total free-surface energy proxy: gamma_s * interfacial length."""
    return float(p.gamma_s * np.sum(_grad_mag(f, p)) * p.dx * p.dx)


def gb_energy(e1, e2, s, p) -> float:
    """Total GB energy proxy: effective GB energy density * GB/contact area."""
    return float(effective_gamma(s, p) * contact_area(e1, e2, p))


def sample(f, e1, e2, e3, s: Sink, st: Stress, p, step: int, time_s: float, v20: float) -> dict:
    """Build one Milestone-1 diagnostic sample. Pure inspection; no side effects."""
    v2 = float(e2.sum() * p.dx * p.dx)
    a_gb = contact_area(e1, e2, p)
    e_surf = surface_energy(f, p)
    e_gb = gb_energy(e1, e2, s, p)
    return dict(
        step=step,
        time_s=time_s,
        V2=v2,
        V2_ratio=v2 / v20 if v20 else math.nan,
        separation_m=center(e2, p) - wall_x0(p),
        rbm_disp_m=s.cumulative_disp,
        strain=s.cumulative_strain,
        x_neck_m=st.x_neck,
        A_GB_m2=a_gb,
        sigma_Pa=st.sigma,
        sigma_lt_Pa=st.sigma_lt,
        sigma_curv_Pa=st.sigma_curv,
        psi_rad=st.psi,
        psi_eq_rad=st.psi_eq,
        kappa_1pm=st.kappa,
        Fx_cahn_hoffman=st.Fx_cahn_hoffman,
        gamma_gb_eff=st.gamma_gb_eff,
        E_surf_J=e_surf,
        E_gb_J=e_gb,
        G_interface_J=e_surf + e_gb,
        sink_active=int(s.active),
        hazard=s.hazard,
        hazard_threshold=s.threshold,
        quota_progress=min(1.0, s.current_disp / max(p.b, 1e-30)),
        stress_exists=int(st.exists),
    )


_DELTA_KEYS = (
    "V2", "V2_ratio", "separation_m", "strain", "x_neck_m", "A_GB_m2",
    "sigma_Pa", "sigma_lt_Pa", "sigma_curv_Pa", "E_surf_J", "E_gb_J", "G_interface_J",
)


def summarize(samples: list[dict]) -> dict:
    """Reduce a diagnostic trajectory to first/last values, deltas, and the
    short-trajectory directional derivatives requested by Milestone 1."""
    if len(samples) < 2:
        raise ValueError("need at least two samples to summarize a trajectory")
    first, last = samples[0], samples[-1]
    out = dict(n_samples=len(samples), step0=first["step"], step1=last["step"])
    for k in _DELTA_KEYS:
        out[f"d_{k}"] = last[k] - first[k]
    dv2 = last["V2"] - first["V2"]
    denom = dv2 if abs(dv2) > 0 else math.nan
    out["dx_neck_dV2"] = (last["x_neck_m"] - first["x_neck_m"]) / denom
    out["dA_GB_dV2"] = (last["A_GB_m2"] - first["A_GB_m2"]) / denom
    out["dsigma_dV2"] = (last["sigma_Pa"] - first["sigma_Pa"]) / denom
    out["dG_interface_dV2"] = (last["G_interface_J"] - first["G_interface_J"]) / denom
    out["V20"] = first["V2"]
    out["dV2_over_V20"] = dv2 / first["V2"] if first["V2"] else math.nan
    return out
