from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d

from .model import (
    ModelConfig,
    Sink,
    SinteringModel as _BaseSinteringModel,
    compute_stress,
    evolve_eta,
    evolve_f,
    hazard_step,
    ostwald_substrate,
    rbm,
)


def selective_reproject_eta_to_f(f, eta1, eta2, eta3, params):
    """Match MATLAB reproject_eta_to_f in its default repair-only mode.

    This operation is deliberately *not* a global eta<=f or sum(eta)<=f
    projection.  It removes labels only in true void and repairs solid pixels
    that have nearly lost all grain ownership.  The hard consistency repair is
    a separate post-Allen-Cahn operation.
    """
    fb = np.clip(f, 0.0, 1.0)
    e1 = np.maximum(eta1, 0.0).copy()
    e2 = np.maximum(eta2, 0.0).copy()
    use_eta3 = bool(params.use_eta3)
    e3 = np.maximum(eta3, 0.0).copy() if use_eta3 else np.zeros_like(eta3)

    true_void = fb <= 0.005
    e1[true_void] = 0.0
    e2[true_void] = 0.0
    e3[true_void] = 0.0

    esum = e1 + e2 + e3
    repair = (fb > 0.02) & (esum < 0.02)
    if np.any(repair):
        kernel = np.ones((3, 3), dtype=float) / 9.0
        # MATLAB conv2(..., 'same') uses zero padding outside the domain.
        w1 = convolve2d(e1 + 1e-16, kernel, mode="same", boundary="fill")
        w2 = convolve2d(e2 + 1e-16, kernel, mode="same", boundary="fill")
        if use_eta3:
            w3 = convolve2d(e3 + 1e-16, kernel, mode="same", boundary="fill")
        else:
            w3 = np.zeros_like(e3)
        wsum = w1 + w2 + w3 + 1e-30
        e1[repair] = fb[repair] * w1[repair] / wsum[repair]
        e2[repair] = fb[repair] * w2[repair] / wsum[repair]
        e3[repair] = fb[repair] * w3[repair] / wsum[repair]

    return tuple(np.clip(e, 0.0, 1.0) for e in (e1, e2, e3))


def hard_eta_consistency(f, eta1, eta2, eta3, params):
    """Match the hard consistency repair at the end of MATLAB evolve_eta."""
    fb = np.clip(f, 0.0, 1.0)
    e1 = np.clip(eta1, 0.0, fb)
    e2 = np.clip(eta2, 0.0, fb)
    e3 = np.clip(eta3, 0.0, fb) if params.use_eta3 else np.zeros_like(eta3)

    esum = e1 + e2 + e3
    over = esum > fb + 1e-12
    if np.any(over):
        scale = fb[over] / esum[over]
        e1[over] *= scale
        e2[over] *= scale
        e3[over] *= scale

    return tuple(np.clip(e, 0.0, 1.0) for e in (e1, e2, e3))


class SinteringModel(_BaseSinteringModel):
    """v64-ordered integrator using the existing low-level Python kernels."""

    def run(self):
        p = self.p
        last = self.step0 - 1
        stop = False
        reason = ""

        if self.c.status_prints:
            print(
                f"grid={p.Nx}x{p.Ny} dx={p.dx*1e9:.2f} nm "
                f"R2={p.R2*1e9:.1f} nm Rx/Ry={p.Rx*1e9:.1f}/{p.Ry*1e9:.1f} nm\n"
                f"dt={p.dt:.3e}s Nt={p.Nt} target={p.sigma_target/1e6:g} MPa"
            )

        for t in range(self.step0, p.Nt + 1):
            last = t

            # 1. Conservative CH evolution of f.
            self.f = evolve_f(self.f, self.e1, self.e2, self.e3, self.s, Sink(), p)

            # MATLAB v64: selective repair only after CH.  Do not globally
            # force eta_i<=f here, because the temporary eta/f mismatch carries
            # boundary-motion memory into the subsequent relaxation.
            self.e1, self.e2, self.e3 = selective_reproject_eta_to_f(
                self.f, self.e1, self.e2, self.e3, p
            )

            # 2. Explicit Ostwald transfer.  No immediate reprojection.
            if p.geometry == "substrate":
                self.f, self.e1, self.e2, self.e3 = ostwald_substrate(
                    self.f, self.e1, self.e2, self.e3, p
                )

            # 3. AC smoothing, followed by the hard consistency repair that is
            # part of MATLAB evolve_eta.
            self.e1, self.e2, self.e3 = evolve_eta(self.e1, self.e2, self.e3, p)
            self.e1, self.e2, self.e3 = hard_eta_consistency(
                self.f, self.e1, self.e2, self.e3, p
            )

            # 4. Stress and integrated hazard.
            if t == 1 or (t - 1) % p.hazard_every == 0:
                self.st, stop, reason = compute_stress(
                    self.f, self.e1, self.e2, self.e3, self.s, p
                )
                if not stop:
                    on = hazard_step(
                        self.s,
                        self.st,
                        p,
                        p.dt if t == 1 else p.dt * p.hazard_every,
                        self.rng,
                    )
                    if on and self.c.event_prints:
                        print(
                            f"t={t*p.dt*1e3:.3f} ms activation "
                            f"sigma={self.st.sigma/1e6:.1f} MPa"
                        )

            # 5. RBM, then the same selective eta-to-f repair used by MATLAB.
            if self.s.active:
                self.f, self.e1, self.e2, self.e3, _ = rbm(
                    self.f, self.e1, self.e2, self.e3, self.s, p
                )
                self.e1, self.e2, self.e3 = selective_reproject_eta_to_f(
                    self.f, self.e1, self.e2, self.e3, p
                )

            if t == 1 or t % p.diag_every == 0:
                self.diag.append(
                    dict(
                        step=t,
                        time_s=t * p.dt,
                        sigma_Pa=self.st.sigma,
                        x_neck_m=self.st.x_neck,
                        hazard=self.s.hazard,
                        hazard_threshold=self.s.threshold,
                        r_nuc=self.s.r_nuc,
                        tau_sink_s=self.s.tau_sink,
                        sink_active=int(self.s.active),
                        quota_progress=min(1.0, self.s.current_disp / max(p.b, 1e-30)),
                        strain=self.s.cumulative_strain,
                        V2_ratio=float(self.e2.sum() * p.dx**2 / p.V2_initial),
                        Vsolid_ratio=float(self.f.sum() * p.dx**2 / p.V_solid_initial),
                    )
                )

            if t == 1 or t % p.save_interval == 0:
                np.savez_compressed(
                    self.c.output_dir / f"sintering_{self.c.out_tag}_frame{t:09d}.npz",
                    f=self.f.astype("f4"),
                    eta1=self.e1.astype("f4"),
                    eta2=self.e2.astype("f4"),
                    eta3=self.e3.astype("f4"),
                    step=t,
                    dt=p.dt,
                )
                if self.c.status_prints:
                    print(
                        f"t={t*p.dt*1e3:.3f} ms sigma={self.st.sigma/1e6:.1f} MPa "
                        f"V2/V20={self.e2.sum()*p.dx**2/p.V2_initial:.4f} "
                        f"Vs/Vs0={self.f.sum()*p.dx**2/p.V_solid_initial:.8f} "
                        f"H={self.s.hazard:.3g}/{self.s.threshold:.3g} "
                        f"active={int(self.s.active)}"
                    )

            if self.c.checkpoint and t % p.checkpoint_interval == 0:
                self.save(
                    self.c.output_dir / f"sintering_{self.c.out_tag}_ckpt_step{t:09d}.h5",
                    t,
                )

            if stop:
                break

        if self.diag:
            with (
                self.c.output_dir / f"sintering_{self.c.out_tag}_diagnostics.csv"
            ).open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.diag[0])
                writer.writeheader()
                writer.writerows(self.diag)

        final = self.c.output_dir / f"sintering_{self.c.out_tag}_final.h5"
        self.save(final, last)
        out = dict(
            final_step=last,
            final_time_s=last * p.dt,
            stop_requested=stop,
            stop_reason=reason,
            nucleations_gb1=self.s.nucleations,
            climbs_gb1=self.s.climbs,
            sink_active=bool(self.s.active),
            quota_progress=min(1.0, self.s.current_disp / max(p.b, 1e-30)),
            strain_gb1=self.s.cumulative_strain,
            V2_ratio=float(self.e2.sum() * p.dx**2 / p.V2_initial),
            Vsolid_ratio=float(self.f.sum() * p.dx**2 / p.V_solid_initial),
            final_file=str(final),
        )
        (
            self.c.output_dir / f"sintering_{self.c.out_tag}_summary.json"
        ).write_text(json.dumps(out, indent=2))
        return out
