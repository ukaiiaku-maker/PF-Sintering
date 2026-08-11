"""Milestone 15I Section 6: mobility-independent thermodynamic driving
strengths D_eta, D_surf, reusing EXISTING qualified machinery only (no
new energy decomposition):

  D_eta = -Fdot_eta/M_eta, Fdot_eta = dx^2*sum_i sum_grid(g_i*eta_dot_i),
  g_i = constrained_eta.structural_thermodynamic_force (the EXACT
  variational derivative constrained_tangent_cone_eta_update itself
  uses), eta_dot_i measured from an ACTUAL short GB-only trial (so it
  is the real tangent-cone-projected + reproject-corrected velocity the
  production update applies, not an idealized unconstrained guess).

  D_surf = D_h/M_s, D_h = dx^2*(sum(Dx)+sum(Dy)) from
  surface_transport.exact_dissipation_face_projected(fp) (the ALREADY
  machine-precision-verified identity Fdot_chain=-D_h) -- Dx/Dy already
  carry the physical M_s tensor factor, so dividing by M_s gives the
  mobility-independent driving strength.

Both are >=0 by construction (D_eta from the projection-onto-cone
argument in constrained_eta.py's own docstring; D_h from
exact_dissipation_face_projected's own global-PSD verification, Milestone
13E Section 6).
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15_gb_surface_rate_competition import BC_X, BC_Y  # noqa: E402
from m15g_mechanism_lib import _mu_field, trial_evolve  # noqa: E402
from pf_sintering.constrained_eta import local_wc, structural_thermodynamic_force  # noqa: E402
from pf_sintering.surface_transport import exact_dissipation_face_projected, m_s_ref, surface_flux_face_projected  # noqa: E402


def D_eta(f, e1, e2, e3, s, p, n_steps=1, bc_x=BC_X, bc_y=BC_Y):
    """Mobility-independent GB/eta driving strength at the given state."""
    Wc = local_wc(f, e1, e2, e3, s, p)
    g1 = structural_thermodynamic_force(e1, f, Wc, p.dx, p.k_eta, bc_x, bc_y)
    g2 = structural_thermodynamic_force(e2, f, Wc, p.dx, p.k_eta, bc_x, bc_y)
    g3 = structural_thermodynamic_force(e3, f, Wc, p.dx, p.k_eta, bc_x, bc_y) if p.use_eta3 else np.zeros_like(e1)

    f1, e1_1, e2_1, e3_1 = trial_evolve(f, e1, e2, e3, s, p, "GB", n_steps, bc_x=bc_x, bc_y=bc_y)
    Delta_t = n_steps * p.dt
    eta_dot_1 = (e1_1 - e1) / Delta_t
    eta_dot_2 = (e2_1 - e2) / Delta_t
    eta_dot_3 = (e3_1 - e3) / Delta_t if p.use_eta3 else np.zeros_like(e1)

    Fdot_eta = p.dx * p.dx * float(np.sum(g1 * eta_dot_1) + np.sum(g2 * eta_dot_2) + np.sum(g3 * eta_dot_3))
    return -Fdot_eta / p.M_eta, Fdot_eta


def D_surf(f, e1, e2, e3, s, p, bc_x=BC_X, bc_y=BC_Y):
    """Mobility-independent surface-diffusion driving strength at the
    given state (instantaneous, no trial evolution needed -- exact
    dissipation identity is evaluated directly on the current mu/flux
    fields)."""
    mu = _mu_field(f, e1, e2, e3, s, p, bc_x, bc_y)
    M_s = m_s_ref(p.M_f, p.interface_width)
    fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, bc_x, bc_y)
    Dx, Dy = exact_dissipation_face_projected(fp)
    D_h = p.dx * p.dx * (float(np.sum(Dx)) + float(np.sum(Dy)))
    return D_h / M_s, -D_h
