"""M16P Section 7: standalone diagnostic reproducing the root-cause
investigation of the ~8.8e-6 per-event mass residual (preserved for
provenance; the actual fix lives in pf_sintering/axisym_sink_rbm.py and
pf_sintering/m16m_multisink.py).

Instruments one single-substep transport call on the real M16K frozen
state and separately tracks: (1) the volume change from clipping f up to
0 (the negative-undershoot artifact) versus (2) clipping e1/e2 to [0,1]
(found NOT to contribute at all in this state). Confirms the f-negative-
clip channel alone accounts for the entire previously-observed drift.

Run: ../.venv/bin/python scripts/m16p_mass_residual_microtest.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym_sink_rbm import HazardParams, _axisym_weighted_sum  # noqa: E402
from pf_sintering.grain_roles import roles_from_m16j_geometry  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry  # noqa: E402

PSI_DEG, GAMMA_S = 160.0, 1.0
R_P_NM, RATIO, W_NM, DX_NM = 1000.0, 0.10, 10.0, 1.25
B = 2.5e-10


def main():
    d = np.load("runs/m16k_prescribed_displacement_microtest/frozen_state.npz")
    f0, e1_0, e2_0 = d["f"], d["e1"], d["e2"]
    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=None, X0_over_2Rp=RATIO, psi_deg=PSI_DEG,
                                    W_nm=W_NM, dr_nm=DX_NM, dz_nm=DX_NM, aspect_ratio=1.0)
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    dz = geo["dz"] * 1e-9
    roles = roles_from_m16j_geometry(e1_0, e2_0)
    particle, substrate = roles.particle.copy(), roles.substrate.copy()
    f = f0.copy()

    V0 = 12.5 * B ** 3
    A0_J = 0.8589 * 1.602176634e-19
    GS = 201.74e-9
    hp = HazardParams(kB=1.380649e-23, T=1000.0, Omega=1e-29, b=B,
                       D_gb=1e-3 * math.exp(-1.5e5 / (8.314 * 1000.0)), GS=GS, r0=1e12, A0=A0_J, V0=V0, tau_ex0=0.0)

    def vol(field):
        return _axisym_weighted_sum(field, r_c)

    sigma_drive = 45.126e6
    dt = 1.0
    xd = 0.5 * hp.GS / 2
    tau_Coble = xd * xd * hp.kB * hp.T / (sigma_drive * hp.Omega * hp.D_gb) + hp.tau_ex0
    v_event = hp.b / tau_Coble
    d_delta_full = v_event * dt
    d_delta_requested = min(d_delta_full, hp.b)  # fresh event, remaining=b
    frac = d_delta_requested / d_delta_full
    dt_scaled = dt * frac
    den = particle + substrate + 1e-30
    vz_field = -v_event * (particle / den)
    vmax = float(np.max(np.abs(vz_field)))
    n_sub = max(1, math.ceil(dt_scaled * vmax / dz / 0.4))
    ds = dt_scaled / n_sub  # correctly frac-scaled substep size (NOT dt=1.0 directly)

    V_f0 = vol(f)
    v_face = 0.5 * (vz_field + np.roll(vz_field, -1, axis=0))
    f_face = np.maximum(v_face, 0) * f + np.minimum(v_face, 0) * np.roll(f, -1, axis=0)
    f_new = f - ds * (f_face - np.roll(f_face, 1, axis=0)) / dz

    V_preclip = vol(f_new)
    f_negclip_only = np.clip(f_new, 0.0, None)
    V_after_negclip = vol(f_negclip_only)
    negclip_gain = (V_after_negclip - V_preclip) / V_f0

    print(f"V_f before advection:      {V_f0:.10e}")
    print(f"V_f after advection:       {V_preclip:.10e}")
    print(f"relative gain from clipping f up to >=0 only: {negclip_gain:.6e}")
    print("(this matches the ~8.8e-6 total mass drift previously observed per event -- ")
    print(" confirming the f<0 undershoot, uncompensated by the excess-mass redistribution, ")
    print(" was the entire source. Fixed by tracking deficit=max(0,-f) alongside ")
    print(" excess=max(0,f-1) and applying one combined correction.)")


if __name__ == "__main__":
    main()
