"""Milestone 16B Section 20: Cartesian-vs-axisymmetric same-profile
comparison.

Per Section 17's own finding: the M15 series' "contact broadening vs
narrowing" result (M15G-M15I) was produced on the SINUSOIDAL substrate,
which has NO unique axisymmetric interpretation (a corrugated plate
with straight Z-invariant ridges cannot be recovered by revolving any
single axis). That specific comparison is therefore out of scope here.
What CAN be compared on a matched, physically well-posed footing is the
FLAT-substrate case (model.py's geometry="substrate", i.e.
sinusoid_amplitude=0): a particle sitting on a flat wall, run in (a)
the Cartesian per-unit-depth production path
(surface_transport.variational_surface_diffusion_step +
constrained_eta.constrained_tangent_cone_eta_update, the exact
m15_gb_surface_rate_competition.py stepping pattern) and (b) the
axisymmetric analogue (axisym_gb_face_projected_step on
axisym_m15_flat_geometry), with gamma_s, gamma_gb, W, M_s (the
INTEGRATED tangential mobility m_s_ref(M_f,W)), and M_eta all matched
by direct construction (not by matching the config-level scale knobs,
which have different unit conventions in the two paths -- p.M_f/p.M_eta
are overridden directly after build_state so the actual numbers used
in the dynamics are identical to the axisymmetric run's).

Reports contact-width evolution (dL_contact/dt sign), neck-radius
evolution, energy evolution, and states plainly whether the two
geometries broaden or narrow -- this is a test of azimuthal curvature's
effect on a flat-substrate spheroidal particle, not a replication of
the M15 GB-vs-surface rate-competition finding.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import (  # noqa: E402
    axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_m15_flat_geometry, axisym_reproject,
    axisym_volume, r_centers_faces,
)
from pf_sintering.ch_exact_energy import mu_isotropic  # noqa: E402
from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients, m_gb_from_m_eta  # noqa: E402
from pf_sintering.model import reproject  # noqa: E402
from pf_sintering.surface_transport import SECH8_INTEGRAL, variational_surface_diffusion_step  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m15_gb_surface_rate_competition import BC_X, BC_Y, build_state, energy_ledger  # noqa: E402
from m16b_com_no_sink_test import P as AxP  # noqa: E402
from m16b_com_no_sink_test import find_stable_dt, measure_neck_radius  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16b_campaign")

W_NM = 10.0
GAMMA_GB = 0.6
M_S_TARGET = 1e-33
M_ETA_TARGET = 1e-33 / (W_NM * 1e-9 * SECH8_INTEGRAL)


def cartesian_contact_width(f, e1, e2, p):
    """Contact-width proxy matching the axisymmetric neck-radius
    measurement's spirit: the full (both-sides) Y-extent of the region
    where min(e1,e2) exceeds a threshold, at the substrate-surface
    X-column, located via SUB-GRID linear interpolation of the
    min(e1,e2)=thresh crossing (matching measure_R_of_z's own
    interpolation convention) -- NOT a raw grid-cell count, which is
    quantized to multiples of dx and insensitive to sub-cell change.

    AXIS CONVENTION (found by direct inspection, cost two earlier
    broken versions of this function): model.py's initialize_fields
    builds `X,Y=np.meshgrid(x,y)` with the numpy DEFAULT 'xy' indexing,
    so field arrays are shaped (Ny,Nx) with axis 0 = Y (row), axis 1 =
    X (column) -- i.e. a FIXED-X slice is `field[:, j]`, not
    `field[j]`. An earlier version of this function used `field[j]`
    (a fixed-Y row) with `j` computed from an X-position formula,
    silently reading a nonsensical slice -- it happened to return
    exactly 8.0nm forever (t=0 to t=1) as an artifact, not a genuine
    "no change" physics finding."""
    wall = (p.substrate_wall_frac - 0.5) * p.Nx * p.dx
    j_wall = int(round(wall / p.dx + p.Nx / 2))
    j_wall = max(0, min(p.Nx - 1, j_wall))
    overlap = np.minimum(e1[:, j_wall], e2[:, j_wall])
    thresh = 0.05
    n = len(overlap)
    center = n // 2
    if overlap[center] <= thresh:
        return 0.0
    # walk outward from center to find the crossing on the + side
    i = center
    while i < n - 1 and overlap[i + 1] > thresh:
        i += 1
    if i >= n - 1:
        half_width = (n - 1 - center) * p.dx
    else:
        f0, f1 = overlap[i], overlap[i + 1]
        frac = (f0 - thresh) / (f0 - f1) if f0 != f1 else 0.0
        half_width = (i - center + frac) * p.dx
    return 2.0 * half_width


def run_cartesian(t_target, n_sample, dx_nm=2.0):
    p, f, e1, e2, e3 = build_state(dx_nm=dx_nm, W_nm=W_NM, gamma_gb=GAMMA_GB, A_nm=0.0,
                                    R2_nm=80.0, aspect_ratio=2.0, overlap_nm=20.0)
    from pf_sintering.model import Sink
    s = Sink(threshold=math.inf)
    # override to the EXACT matched integrated mobilities (bypassing
    # build_config's own m_s_ref/M_GB scale-knob conventions)
    p.M_f = M_S_TARGET / (p.interface_width * SECH8_INTEGRAL)
    p.M_eta = M_ETA_TARGET
    M_s = M_S_TARGET
    dt = p.dt

    n_steps_total = max(1, int(t_target / dt))
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    mass0 = float(f.sum()) * p.dx * p.dx
    F0 = energy_ledger(f, e1, e2, e3, s, p)["F_total"]
    contact0 = cartesian_contact_width(f, e1, e2, p)
    print(f"[cartesian] dx={dx_nm}nm dt={dt:.4e} n_steps={n_steps_total} M_f={p.M_f:.4e} "
          f"M_eta={p.M_eta:.4e} M_s(check)={M_s:.4e} contact0={contact0*1e9:.4f}nm F0={F0:.4e}")

    rows = []
    step = 0
    for target in sample_steps:
        while step < target:
            mu = mu_isotropic(f, e1, e2, e3, s, p)
            f, _ = variational_surface_diffusion_step(f, mu, p.dx, dt, p.interface_width, M_s,
                                                        bc_x=BC_X, bc_y=BC_Y)
            e1, e2, e3, _ = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, dt=dt,
                                                                  use_eta3=False, bc_x=BC_X, bc_y=BC_Y)
            step += 1
        led = energy_ledger(f, e1, e2, e3, s, p)
        contact = cartesian_contact_width(f, e1, e2, p)
        V = float(f.sum()) * p.dx * p.dx
        rows.append(dict(step=step, t=step * dt, contact_nm=contact * 1e9, F=led["F_total"],
                          mass_drift=(V - mass0) / mass0))
        print(f"  [cartesian] step={step} t={rows[-1]['t']:.3e} contact={rows[-1]['contact_nm']:.4f}nm "
              f"F={led['F_total']:.4e} mass_drift={rows[-1]['mass_drift']:.2e}")
    return dict(dx_nm=dx_nm, dt=dt, n_steps_total=n_steps_total, contact0_nm=contact0 * 1e9, F0=F0, rows=rows)


def run_axisym(t_target, n_sample, dx_nm=2.0):
    Nz, Nr = 200, 120
    dr = dz = dx_nm * 1e-9
    W = W_NM * 1e-9
    p = AxP(gamma_s=1.0, gamma_gb=GAMMA_GB, W=W)
    Wc = gb_obstacle_coefficients(GAMMA_GB, W)["Wc"]
    geom = axisym_m15_flat_geometry(Nz, Nr, dz, dr, W, Rz_nm=113.137, Rr_nm=56.569, overlap_nm=20.0,
                                     wall_z_frac=0.2)
    f = geom["f"]
    e1, e2 = axisym_reproject(f, geom["e1_raw"].copy(), geom["e2_raw"].copy())
    r_c, r_f = r_centers_faces(Nr, dr)
    z = (np.arange(Nz) + 0.5) * dz
    M_s = M_S_TARGET
    M_eta = M_ETA_TARGET

    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100)
    dt *= 0.4
    n_steps_total = max(1, int(t_target / dt))
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    V0 = axisym_volume(f, r_c, dr, dz)
    F0 = axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)
    neck0 = measure_neck_radius(f, r_c, z, geom["wall_z"], dz)
    print(f"[axisym] dx={dx_nm}nm dt={dt:.4e} n_steps={n_steps_total} M_s={M_s:.4e} M_eta={M_eta:.4e} "
          f"neck0={neck0*1e9:.4f}nm F0={F0:.4e}")

    rows = []
    step = 0
    for target in sample_steps:
        while step < target:
            f, e1, e2, diag = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
            step += 1
            if not np.all(np.isfinite(f)):
                raise RuntimeError(f"blew up at step {step}")
        F = axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)
        V = axisym_volume(f, r_c, dr, dz)
        neck = measure_neck_radius(f, r_c, z, geom["wall_z"], dz)
        rows.append(dict(step=step, t=step * dt, neck_nm=neck * 1e9, F=F, mass_drift=(V - V0) / V0))
        print(f"  [axisym] step={step} t={rows[-1]['t']:.3e} neck={rows[-1]['neck_nm']:.4f}nm "
              f"F={F:.4e} mass_drift={rows[-1]['mass_drift']:.2e}")
    return dict(dx_nm=dx_nm, dt=dt, n_steps_total=n_steps_total, neck0_nm=neck0 * 1e9, F0=F0, rows=rows)


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "cartesian_vs_axisym.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            result = json.load(fh)
    else:
        t_target = 1.0
        n_sample = 20
        cart = run_cartesian(t_target, n_sample)
        axi = run_axisym(t_target, n_sample)
        result = dict(cartesian=cart, axisym=axi, t_target=t_target)
        with open(out_path, "w") as fh:
            json.dump(result, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Section 20 summary ---")
    cart, axi = result["cartesian"], result["axisym"]
    c0, cf = cart["contact0_nm"], cart["rows"][-1]["contact_nm"]
    a0, af = axi["neck0_nm"], axi["rows"][-1]["neck_nm"]
    print(f"Cartesian contact width: {c0:.4f}nm -> {cf:.4f}nm  (d/dt sign: {'BROADEN' if cf>c0 else 'NARROW'})")
    print(f"Axisymmetric neck radius: {a0:.4f}nm -> {af:.4f}nm  (d/dt sign: {'BROADEN' if af>a0 else 'NARROW'})")
    same_dir = (cf > c0) == (af > a0)
    print(f"SAME DIRECTION: {same_dir}")
    if not same_dir:
        print("Cartesian and axisymmetric disagree in sign -- definitive azimuthal-curvature effect.")
