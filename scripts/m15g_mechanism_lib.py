"""Milestone 15G Sections 8-11: contact-broadening mechanism audit.

Given a saved state (f,e1,e2,e3,p), performs three SHORT, NON-COMMITTED
trial evolutions (SURF-only, GB-only, FULL) over the same physical
Delta_t, and reports:

  - operator-level changes in L_contact, TJ x-positions, F_cap_n, sigma,
    L_GB, and total/component free energy (Section 8/11), reusing
    m15f_campaign_lib.sample_state_f/energy_ledger_f on the before/after
    states rather than re-deriving those quantities;
  - a direct KINEMATIC TJ velocity from the two level sets A=f-0.5,
    B=eta1-eta2 (Section 10), solving grad(A).v=-dA/dt,
    grad(B).v=-dB/dt locally at the TJ, for each operator trial
    separately;
  - a surface-flux budget (Section 9): mu(s)/J_tangent(s) along each
    TJ's particle-side branch (production-consistent flux, anisotropy-
    aware), sampled at control sections 1W/2W/3W/4W from the TJ.

Trial branches never modify the canonical trajectory -- callers must
pass in copies (trial_evolve already copies internally, but the
canonical state held by the caller is never touched here).
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/..")

from m15_gb_surface_rate_competition import BC_X, BC_Y, sample_state  # noqa: E402
from m15f_campaign_lib import energy_ledger_f, sample_state_f  # noqa: E402
from pf_sintering.aniso_flux import mu_anisotropic  # noqa: E402
from pf_sintering.ch_exact_energy import mu_isotropic  # noqa: E402
from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update  # noqa: E402
from pf_sintering.curvature_extraction import branch_mu_J_profile  # noqa: E402
from pf_sintering.surface_transport import m_s_ref, surface_flux_face_projected, variational_surface_diffusion_step  # noqa: E402
from pf_sintering.tj_force import _local_gradient, _sample_bilinear, locate_neck_tjs  # noqa: E402


def _mu_field(f, e1, e2, e3, s, p, bc_x=BC_X, bc_y=BC_Y):
    if p.use_aniso_surface:
        return mu_anisotropic(f, e1, e2, e3, s, p, bc_x, bc_y)
    return mu_isotropic(f, e1, e2, e3, s, p)


def trial_evolve(f, e1, e2, e3, s, p, mode, n_steps, bc_x=BC_X, bc_y=BC_Y):
    """mode in {'SURF','GB','FULL'}. Operates on and returns COPIES; the
    caller's arrays are never mutated."""
    assert mode in ("SURF", "GB", "FULL")
    f = f.copy(); e1 = e1.copy(); e2 = e2.copy(); e3 = e3.copy()
    dt = p.dt
    M_s = m_s_ref(p.M_f, p.interface_width)
    for _ in range(n_steps):
        if mode in ("SURF", "FULL"):
            mu = _mu_field(f, e1, e2, e3, s, p, bc_x, bc_y)
            f, _ = variational_surface_diffusion_step(f, mu, p.dx, dt, p.interface_width, M_s,
                                                        bc_x=bc_x, bc_y=bc_y)
        if mode in ("GB", "FULL"):
            e1, e2, e3, _ = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, dt=dt, use_eta3=False,
                                                                  bc_x=bc_x, bc_y=bc_y)
    return f, e1, e2, e3


def tj_velocity_kinematic(f0, e1_0, e2_0, f1, e1_1, e2_1, tj_xy, p, dt):
    """Section 10: direct kinematic TJ velocity from the two level sets
    A=f-0.5 (free surface), B=eta1-eta2 (GB ownership), solving
    grad(A).v=-dA/dt, grad(B).v=-dB/dt locally at tj_xy. Gradients are
    evaluated on the MIDPOINT field (average of before/after) for
    second-order-ish accuracy; time derivatives are Eulerian finite
    differences at the fixed point tj_xy, sampled bilinearly on each
    field. Returns (vx,vy,ok) -- ok=False if the local level-set
    intersection is degenerate (near-singular 2x2 system, e.g. the two
    contours nearly tangent rather than transverse)."""
    f_mid = 0.5 * (f0 + f1)
    d0 = e1_0 - e2_0
    d1 = e1_1 - e2_1
    d_mid = 0.5 * (d0 + d1)
    gA = _local_gradient(f_mid, tj_xy, p)
    gB = _local_gradient(d_mid, tj_xy, p)
    fA0 = float(_sample_bilinear(f0, [tj_xy[0]], [tj_xy[1]], p)[0])
    fA1 = float(_sample_bilinear(f1, [tj_xy[0]], [tj_xy[1]], p)[0])
    At = (fA1 - fA0) / dt
    dB0 = float(_sample_bilinear(d0, [tj_xy[0]], [tj_xy[1]], p)[0])
    dB1 = float(_sample_bilinear(d1, [tj_xy[0]], [tj_xy[1]], p)[0])
    Bt = (dB1 - dB0) / dt
    M = np.array([[gA[0], gA[1]], [gB[0], gB[1]]])
    det = np.linalg.det(M)
    if abs(det) < 1e-6 * max(np.linalg.norm(gA), 1e-30) * max(np.linalg.norm(gB), 1e-30):
        return math.nan, math.nan, False
    v = np.linalg.solve(M, np.array([-At, -Bt]))
    return float(v[0]), float(v[1]), True


def flux_budget(f, e1, e2, e3, s, p, tj_xy, branch_dir, radii_in_W=(1.0, 2.0, 3.0, 4.0)):
    """Section 9: mu(s)/J_tangent(s) along the particle-side branch from
    tj_xy, using the SAME (anisotropy-aware) mu/flux the production
    f-step actually consumes, sampled at control sections
    radii_in_W*interface_width from the TJ. J_tangent>0 means flux AWAY
    from the TJ along branch_dir (branch_mu_J_profile's own convention);
    positive values here mean mass leaving the contact region along this
    branch, i.e. consistent with L_contact BROADENING."""
    mu = _mu_field(f, e1, e2, e3, s, p)
    M_s = m_s_ref(p.M_f, p.interface_width)
    fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
    Jx = 0.5 * (fp["Jx_face"] + np.roll(fp["Jx_face"], 1, axis=1))
    Jy = np.zeros_like(Jx)
    Jy[1:] = 0.5 * (fp["Jy_face"][1:] + fp["Jy_face"][:-1])
    W = p.interface_width
    prof = branch_mu_J_profile(f, mu, Jx, Jy, p, tj_xy, branch_dir, max_arclength=6.0 * W)
    if prof is None:
        return None
    s_arr, mu_arr, Jt_arr, Jn_arr = prof["s"], prof["mu"], prof["J_tangent"], prof["J_normal"]
    out = {"resolved": True}
    for r_W in radii_in_W:
        s_target = r_W * W
        if s_target > s_arr[-1]:
            out[f"{r_W:g}W"] = dict(mu=math.nan, J_tangent=math.nan, J_normal=math.nan, out_of_range=True)
            continue
        mu_s = float(np.interp(s_target, s_arr, mu_arr))
        Jt_s = float(np.interp(s_target, s_arr, Jt_arr))
        Jn_s = float(np.interp(s_target, s_arr, Jn_arr))
        out[f"{r_W:g}W"] = dict(mu=mu_s, J_tangent=Jt_s, J_normal=Jn_s, out_of_range=False)
    return out


def operator_audit(f, e1, e2, e3, s, p, label, n_steps=100, verbose=True):
    """Section 8/10/11: run SURF/GB/FULL trials from (f,e1,e2,e3), report
    operator-level Delta(L_contact), Delta(x_TJ), Delta(F_cap_n),
    Delta(sigma), Delta(L_GB), Delta(F_total and components), and
    kinematic TJ velocities. Returns a dict; does not mutate inputs."""
    dt = p.dt
    Delta_t = n_steps * dt
    mass0 = float(f.sum()) * p.dx * p.dx
    F0 = energy_ledger_f(f, e1, e2, e3, s, p)["F_total"]
    before = sample_state_f(f, e1, e2, e3, s, p, 0, 0.0, mass0, F0, 0.0, 0.0)
    tjs = locate_neck_tjs(f, e1, e2, p)
    result = dict(label=label, Delta_t=Delta_t, n_steps=n_steps, before=before, tjs=None, modes={})
    if tjs is None:
        if verbose:
            print(f"[{label}] TJs not resolved, skipping operator audit")
        return result
    tj_top, tj_bottom = tjs["tj_top"], tjs["tj_bottom"]
    result["tjs"] = dict(tj_top=[float(v) for v in tj_top], tj_bottom=[float(v) for v in tj_bottom])

    flux_before = dict(
        top=flux_budget(f, e1, e2, e3, s, p, tj_top, np.array(before.get("top_particle_tangent", [1.0, 0.0]))),
        bottom=flux_budget(f, e1, e2, e3, s, p, tj_bottom, np.array(before.get("bot_particle_tangent", [1.0, 0.0]))),
    )
    result["flux_before"] = flux_before

    for mode in ("SURF", "GB", "FULL"):
        f1, e1_1, e2_1, e3_1 = trial_evolve(f, e1, e2, e3, s, p, mode, n_steps)
        after = sample_state_f(f1, e1_1, e2_1, e3_1, s, p, n_steps, Delta_t, mass0, F0, 0.0, 0.0)
        led_after = energy_ledger_f(f1, e1_1, e2_1, e3_1, s, p)

        vtop = tj_velocity_kinematic(f, e1, e2, f1, e1_1, e2_1, tj_top, p, Delta_t)
        vbot = tj_velocity_kinematic(f, e1, e2, f1, e1_1, e2_1, tj_bottom, p, Delta_t)

        def d(key, bkey=None):
            bkey = bkey or key
            a = after.get(key)
            b = before.get(bkey)
            if a is None or b is None or not (math.isfinite(a) if isinstance(a, float) else True):
                return None
            try:
                return float(a) - float(b)
            except (TypeError, ValueError):
                return None

        sigma_key = "sigma_sint_app_endpoint_form_aniso" if p.use_aniso_surface else "sigma_sint_app_endpoint_form"
        Fcap_key = "F_cap_n_endpoint_form_aniso" if p.use_aniso_surface else "F_cap_n_endpoint_form"
        entry = dict(
            Delta_L_contact=d("L_contact"),
            Delta_L_GB=d("L_GB"),
            Delta_sigma=d(sigma_key),
            Delta_F_cap_n=d(Fcap_key),
            Delta_F_total=led_after["F_total"] - F0,
            Delta_E_surface=led_after["E_surface"] - before["E_surface"],
            Delta_E_coupling=led_after["E_coupling"] - before["E_coupling"],
            Delta_E_GB=led_after["E_GB"] - before["E_GB"],
            v_TJ_top=vtop, v_TJ_bottom=vbot,
            after=after,
        )
        result["modes"][mode] = entry
        if verbose:
            print(f"  [{label}][{mode}] dL_contact={entry['Delta_L_contact']} dsigma={entry['Delta_sigma']} "
                  f"dF_total={entry['Delta_F_total']:.3e} v_TJ_top={vtop[:2]} v_TJ_bot={vbot[:2]}")

    m = result["modes"]
    if all(k in m for k in ("SURF", "GB", "FULL")) and m["SURF"]["Delta_L_contact"] is not None \
            and m["GB"]["Delta_L_contact"] is not None and m["FULL"]["Delta_L_contact"] is not None:
        result["R_coupled_L_contact"] = (m["FULL"]["Delta_L_contact"] - m["SURF"]["Delta_L_contact"]
                                          - m["GB"]["Delta_L_contact"])
    return result
