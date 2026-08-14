"""Milestone 16K Sections 15-19: physical activation-volume ladder and
cumulative-hazard calibration against an ALREADY-COMPUTED deterministic
sink-OFF sigma(t) trajectory (no PF re-run for this screen -- Section 16).

For each V0 in a small physically-motivated ladder, solves for A0 such
that the CUMULATIVE hazard integrated along the deterministic trajectory
reaches ln(2) (50% cumulative activation probability) at the moment the
trajectory first reaches sigma_target=50 MPa -- not a hard threshold, an
Arrhenius first-passage calibration point (Section 14, 16).

Then reports the full predicted nucleation distribution
P(event before X MPa) = 1-exp(-H(t_X)) for X in {30,40,50,60,75,100} MPa
(Section 17), and the resulting tau_sink(sigma) ladder (Section 24).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, ".")
from pf_sintering.axisym_sink_rbm import HazardParams  # noqa: E402

EV = 1.602176634e-19


def load_trajectory(history_csv, sigma_col="sigma_Hussein_MPa", time_col="time"):
    with open(history_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([float(r[time_col]) for r in rows])
    sigma_MPa = np.array([float(r[sigma_col]) for r in rows])
    order = np.argsort(t)
    return t[order], sigma_MPa[order]


def cumulative_hazard(t, sigma_Pa, hp: HazardParams):
    """H(t_i) = integral_0^{t_i} r_nuc(sigma(t')) dt', trapezoidal, using
    the SAME rate formula as hazard_step (sigma clipped at 0 from below)."""
    sigma_pos = np.clip(sigma_Pa, 0.0, None)
    r_nuc = hp.r0 * (hp.b / hp.GS) ** 3 * np.exp(-np.clip(hp.A0 - sigma_pos * hp.V0, 0.0, None) / (hp.kB * hp.T))
    H = np.concatenate([[0.0], np.cumsum(0.5 * (r_nuc[1:] + r_nuc[:-1]) * np.diff(t))])
    return H, r_nuc


def time_to_stress(t, sigma_MPa, target_MPa):
    """First interpolated time the trajectory reaches target_MPa, or None."""
    above = sigma_MPa >= target_MPa
    if not np.any(above):
        return None
    i = np.argmax(above)
    if i == 0:
        return t[0]
    t0, t1 = t[i - 1], t[i]
    s0, s1 = sigma_MPa[i - 1], sigma_MPa[i]
    if s1 == s0:
        return t1
    frac = (target_MPa - s0) / (s1 - s0)
    return t0 + frac * (t1 - t0)


def calibrate_A0_for_ln2_at_target(t, sigma_MPa, GS, V0, sigma_target_MPa=50.0, r0=1e12, T=1000.0, b=2.5e-10):
    hp_base = HazardParams(T=T, GS=GS, r0=r0, b=b, V0=V0, A0=0.0)
    sigma_Pa = sigma_MPa * 1e6
    t_target = time_to_stress(t, sigma_MPa, sigma_target_MPa)
    if t_target is None:
        return None

    def resid(A0):
        hp = HazardParams(T=T, GS=GS, r0=r0, b=b, V0=V0, A0=A0)
        mask = t <= t_target
        H, _ = cumulative_hazard(t[mask], sigma_Pa[mask], hp)
        return H[-1] - math.log(2.0)

    # bracket: A0 too small -> huge H (resid>0); A0 too large -> H~0 (resid<0)
    lo, hi = 0.0, 20.0 * EV
    f_lo, f_hi = resid(lo), resid(hi)
    tries = 0
    while f_lo * f_hi > 0 and tries < 50:
        hi *= 1.5
        f_hi = resid(hi)
        tries += 1
    if f_lo * f_hi > 0:
        return None
    A0 = brentq(resid, lo, hi, xtol=1e-25)
    return A0, t_target


def tau_sink_ladder(hp: HazardParams, sigma_MPa_list=(10, 25, 50, 75, 100)):
    out = []
    xd = 0.5 * hp.GS / 2
    for sMPa in sigma_MPa_list:
        sigma = sMPa * 1e6
        tau = (xd * xd * hp.kB * hp.T) / (sigma * hp.Omega * hp.D_gb) + hp.tau_ex0
        v = hp.b / tau
        out.append(dict(sigma_MPa=sMPa, tau_sink_s=tau, velocity_m_per_s=v))
    return out


def main(history_csv, a_contact_nm, out_json):
    t, sigma_MPa = load_trajectory(history_csv)
    print(f"trajectory: t=[{t[0]:.3f},{t[-1]:.3f}], sigma=[{sigma_MPa.min():.2f},{sigma_MPa.max():.2f}] MPa")

    GS = 2 * a_contact_nm * 1e-9  # same convention as M16H/M16I: GS = 2*contact_radius
    b = 2.5e-10
    V0_over_b3_ladder = [12.5, 30.0, 100.0, 300.0]
    results = []

    for v0_over_b3 in V0_over_b3_ladder:
        V0 = v0_over_b3 * b ** 3
        calib = calibrate_A0_for_ln2_at_target(t, sigma_MPa, GS, V0, sigma_target_MPa=50.0)
        if calib is None:
            print(f"V0/b^3={v0_over_b3}: could not calibrate (trajectory never reaches 50 MPa, or no valid A0 bracket)")
            continue
        A0, t50 = calib
        hp = HazardParams(GS=GS, r0=1e12, T=1000.0, b=b, V0=V0, A0=A0)
        sigma_Pa = sigma_MPa * 1e6
        H_full, r_nuc_full = cumulative_hazard(t, sigma_Pa, hp)

        probs = {}
        for target in (30, 40, 50, 60, 75, 100):
            t_x = time_to_stress(t, sigma_MPa, target)
            if t_x is None:
                probs[f"P_before_{target}MPa"] = None
                continue
            mask = t <= t_x
            H_x, _ = cumulative_hazard(t[mask], sigma_Pa[mask], hp)
            probs[f"P_before_{target}MPa"] = 1.0 - math.exp(-H_x[-1])

        tau_ladder = tau_sink_ladder(hp)

        result = dict(V0_over_b3=v0_over_b3, V0_m3=V0, A0_J=A0, A0_eV=A0 / EV, t50=t50, GS_nm=GS * 1e9,
                      probs=probs, tau_sink_ladder=tau_ladder)
        results.append(result)
        print(f"\nV0/b^3={v0_over_b3}  A0={A0/EV:.4f} eV  t(50MPa)={t50:.3f}")
        for k, v in probs.items():
            print(f"   {k}: {v}")
        print("   tau_sink ladder:")
        for row in tau_ladder:
            print(f"     sigma={row['sigma_MPa']}MPa  tau_sink={row['tau_sink_s']:.4e}s  v={row['velocity_m_per_s']:.4e}m/s")

    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nwrote {out_json}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-csv", required=True)
    ap.add_argument("--a-contact-nm", type=float, required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    main(args.history_csv, args.a_contact_nm, args.out_json)
