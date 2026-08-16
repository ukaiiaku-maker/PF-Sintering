"""M16N Section G: transport-only microtest (PF evolution DISABLED).

On a frozen field, at a FIXED driving stress, apply the corrected
one-b sink-transport contract (Section F: `active_sink_transport_step`
now accumulates `delta_sink` == the analytic Coble-rate quantity the
field is actually advected by, not the measured COM response) with NO
intervening PF stepping (`axisym_gb_face_projected_step` is never
called), and verifies:

  1. cumulative delta_sink -> b EXACTLY (not overshooting, not
     undershooting once completed);
  2. no further field mutation occurs after completion (a subsequent
     transport call is a strict no-op);
  3. delta_COM (measured particle-relative-substrate COM response) is
     recorded at every step, independently, and is NOT required to
     reach b;
  4. mass is conserved (mass_conservation_residual and total volume
     drift stay at floating-point precision).

Then, from the resulting POST-EVENT field state, runs PF-ONLY (no
hazard, no RBM -- pure capillary/surface-diffusion relaxation) for a
further stretch and tracks how the particle-relative-substrate COM
continues to evolve with NO further sink transport applied. This
isolates how much of the delta_COM<delta_sink gap seen during the event
is attributable to concurrent natural capillary relaxation (which keeps
moving COM on its own, with or without any sink activity) versus
anything specific to the transport substep loop itself.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_gb_face_projected_step, axisym_volume  # noqa: E402
from pf_sintering.axisym_sink_rbm import AxisymSink, HazardParams, active_sink_transport_step, particle_com_z  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.grain_roles import roles_from_m16j_geometry  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry  # noqa: E402

sys.path.insert(0, "scripts")
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402

PSI_DEG, GAMMA_S = 160.0, 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
R_P_NM, RATIO, W_NM, DX_NM = 1000.0, 0.10, 10.0, 1.25
B = 2.5e-10
V0 = 12.5 * B ** 3
A0_J = 0.8589 * 1.602176634e-19
GS = 201.74e-9
R0 = 1e12
T = 1000.0
SIGMA_FIXED = 45.126e6  # matches the M16K/M16L/M16M activation-point stress


def main():
    d = np.load("runs/m16k_prescribed_displacement_microtest/frozen_state.npz")
    f0, e1_0, e2_0 = d["f"], d["e1"], d["e2"]
    dz_grid = float(d["dz_grid"])

    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=None, X0_over_2Rp=RATIO, psi_deg=PSI_DEG,
                                    W_nm=W_NM, dr_nm=DX_NM, dz_nm=DX_NM, aspect_ratio=1.0)
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    dr, dz = geo["dr"] * 1e-9, geo["dz"] * 1e-9
    r_f = geo["r_f"] * 1e-9
    W = W_NM * 1e-9
    assert f0.shape == (len(z), len(r_c))

    p = P(gamma_s=GAMMA_S, gamma_gb=GAMMA_GB, W=W)
    Wc = gb_obstacle_coefficients(GAMMA_GB, W)["Wc"]
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    hp = HazardParams(kB=1.380649e-23, T=T, Omega=1e-29, b=B, D_gb=1e-3 * math.exp(-1.5e5 / (8.314 * T)),
                       GS=GS, r0=R0, A0=A0_J, V0=V0, tau_ex0=0.0)

    roles = roles_from_m16j_geometry(e1_0, e2_0)
    f, particle, substrate = f0.copy(), roles.particle.copy(), roles.substrate.copy()
    sink = AxisymSink(active=True, current_disp=0.0)
    V_ref = axisym_volume(f, r_c, dr, dz)

    print("=== PHASE 1: transport-only (NO PF evolution), INCREMENTAL (dt matches real PF stepping) ===")
    dt = 4.8828e-05  # realistic PF dt -- exercises MANY small increments toward b, not one giant step
    step = 0
    cum_delta_sink = 0.0
    cum_delta_COM = 0.0
    completed = False
    while not completed and step < 5000:
        f, particle, substrate, completed, diag = active_sink_transport_step(
            f, particle, substrate, sink, hp, SIGMA_FIXED, dt, dz, r_c, z, 0.0)
        cum_delta_sink += diag["delta_sink_this_step"]
        cum_delta_COM += diag["delta_COM_this_step"]
        step += 1
        if step <= 5 or completed:
            print(f"  step={step} delta_sink_step={diag['delta_sink_this_step']:.4e} "
                  f"delta_COM_step={diag['delta_COM_this_step']:.4e} cum_delta_sink={cum_delta_sink:.6e} "
                  f"cum_delta_COM={cum_delta_COM:.6e} sink.current_disp={sink.current_disp:.4e} completed={completed}")

    print(f"\nTotal steps to complete: {step}")
    print(f"cumulative delta_sink = {cum_delta_sink:.10e} (target b={B:.10e}, ratio={cum_delta_sink/B:.8f})")
    print(f"cumulative delta_COM  = {cum_delta_COM:.10e} (ratio to b={cum_delta_COM/B:.6f})")
    assert abs(cum_delta_sink - B) < 1e-15, "delta_sink must reach b EXACTLY"
    assert sink.active is False, "sink must be deactivated on completion"
    assert sink.current_disp == 0.0, "current_disp must reset for next event"

    V_post_event = axisym_volume(f, r_c, dr, dz)
    print(f"mass drift after event: {(V_post_event - V_ref)/V_ref:.4e}")

    print("\n=== PHASE 1b: no-op verification (transport call after completion) ===")
    f_before, particle_before, substrate_before = f.copy(), particle.copy(), substrate.copy()
    f, particle, substrate, completed2, diag2 = active_sink_transport_step(
        f, particle, substrate, sink, hp, SIGMA_FIXED, dt, dz, r_c, z, 0.0)
    print(f"  completed2={completed2} active={diag2.get('active')} "
          f"fields_unchanged={np.array_equal(f, f_before) and np.array_equal(particle, particle_before)}")
    assert completed2 is False
    assert np.array_equal(f, f_before) and np.array_equal(particle, particle_before)

    print("\n=== PHASE 2: PF-ONLY continuation from post-event state (no hazard, no RBM) ===")
    com_particle_0 = particle_com_z(particle, z, r_c, dr, dz)
    com_substrate_0 = particle_com_z(substrate, z, r_c, dr, dz)
    dt_pf = 4.8828e-05
    n_pf_steps = 500
    for i in range(n_pf_steps):
        f, particle, substrate, _ = axisym_gb_face_projected_step(f, particle, substrate, p, Wc, dr, dz, r_c, r_f,
                                                                    dt_pf, M_s, M_eta, W, bc_z="noflux")
        if not np.all(np.isfinite(f)):
            print(f"  BLOWUP at PF-only step {i}"); break
    com_particle_1 = particle_com_z(particle, z, r_c, dr, dz)
    com_substrate_1 = particle_com_z(substrate, z, r_c, dr, dz)
    natural_relative_drift = (com_particle_0 - com_particle_1) - (com_substrate_0 - com_substrate_1)
    print(f"  after {n_pf_steps} PF-only steps (dt={dt_pf:.4e}, t={n_pf_steps*dt_pf:.4f}):")
    print(f"  natural relative COM drift (no sink activity) = {natural_relative_drift*1e9:.6f}nm "
          f"({natural_relative_drift/B*100:.3f}% of b)")
    V_final = axisym_volume(f, r_c, dr, dz)
    print(f"  mass drift after PF-only continuation: {(V_final - V_ref)/V_ref:.4e}")
    print("\nDONE")


if __name__ == "__main__":
    main()
