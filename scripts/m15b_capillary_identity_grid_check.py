"""Milestone 15B Section 8: does the ~13% capillary-force curvature-form
vs. endpoint-form discrepancy converge with grid refinement at a genuinely
DYNAMIC (deformed, not just t=0) state? Runs the GB-REF case to a bounded
t=0.02s at dx=5, 2.5, 1.25nm and reports force_rel_err at the end."""

from __future__ import annotations

import json
import math
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m12b_grid_convergence import classify_branches  # noqa: E402
from m15_gb_surface_rate_competition import M_ETA_HISTORICAL_REF, BC_X, BC_Y, build_state  # noqa: E402
from pf_sintering.capillary_stress import (  # noqa: E402
    capillary_force_curvature_form,
    capillary_force_endpoint_form,
    force_validation_relative_error,
    trace_particle_arc,
)
from pf_sintering.ch_exact_energy import mu_isotropic  # noqa: E402
from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update  # noqa: E402
from pf_sintering.gb_obstacle_energy import m_gb_from_m_eta  # noqa: E402
from pf_sintering.model import Sink  # noqa: E402
from pf_sintering.surface_transport import m_s_ref, variational_surface_diffusion_step  # noqa: E402
from pf_sintering.tj_force import compute_tj_force, locate_neck_tjs  # noqa: E402

M_GB_ref = m_gb_from_m_eta(M_ETA_HISTORICAL_REF, 20e-9)
s = Sink(threshold=math.inf)

out = {}
for dx in (5.0, 2.5, 1.25):
    p, f, e1, e2, e3 = build_state(dx_nm=dx, M_GB=M_GB_ref, surface_mobility_scale=0.3)
    M_s = m_s_ref(p.M_f, p.interface_width)
    dt = p.dt
    n_steps = round(0.02 / dt)
    for _ in range(n_steps):
        mu = mu_isotropic(f, e1, e2, e3, s, p)
        f, _ = variational_surface_diffusion_step(f, mu, p.dx, dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
        e1, e2, e3, _ = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, dt=dt, use_eta3=False,
                                                              bc_x=BC_X, bc_y=BC_Y)

    tjs = locate_neck_tjs(f, e1, e2, p)
    top = compute_tj_force(f, e1, e2, tjs["tj_top"], s, p)
    bottom = compute_tj_force(f, e1, e2, tjs["tj_bottom"], s, p)
    wall_mean = p.substrate_wall_frac * p.Nx * p.dx
    tpd, tsd = classify_branches(f, p, tjs["tj_top"], top.v_s1, top.v_s2, wall_mean)
    bpd, bsd = classify_branches(f, p, tjs["tj_bottom"], bottom.v_s1, bottom.v_s2, wall_mean)
    arc = trace_particle_arc(f, p, tjs["tj_top"], tjs["tj_bottom"], tpd, bpd)
    if not arc.get("resolved"):
        print(dx, "arc not resolved:", arc.get("reason"))
        out[str(dx)] = dict(resolved=False)
        continue
    F_curv = capillary_force_curvature_form(arc, p.gamma_s)
    F_ep = capillary_force_endpoint_form(tpd, bpd, p.gamma_s)
    rel_err = force_validation_relative_error(F_curv, F_ep)
    print(f"dx={dx}nm t=0.02s F_curv={F_curv} F_ep={F_ep} rel_err={rel_err:.5f} "
          f"arc_len={arc['arc_length']*1e9:.2f}nm closest_approach={arc['closest_approach_dist']*1e9:.3f}nm")
    out[str(dx)] = dict(resolved=True, F_curv=list(F_curv), F_ep=list(F_ep), rel_err=rel_err,
                         arc_length=arc["arc_length"], closest_approach_dist=arc["closest_approach_dist"])

with open("/tmp/m15b_capillary_identity_grid_check.json", "w") as fh:
    json.dump(out, fh, default=str)
print("wrote /tmp/m15b_capillary_identity_grid_check.json")
