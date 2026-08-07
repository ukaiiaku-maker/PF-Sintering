#!/usr/bin/env python3
"""Diagnose which physical/numerical operator changes eta2-defined grain volume.

The updated physics contract is:

  * CH changes conserved solid geometry f but not integrated grain ownership.
  * Structural eta relaxation is regularization only and is mass-neutral.
  * Explicit Ostwald transfer is allowed to change grain volume.
  * RBM translates/deforms grains but is mass-neutral after transport correction.

The ledger must therefore attribute secular no-event V2 change to Ostwald only.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from pf_sintering.model import (
    ModelConfig,
    Sink,
    build_params,
    compute_stress,
    evolve_eta,
    evolve_f,
    hazard_step,
    initialize_fields,
    ostwald_substrate,
    rbm,
)
from pf_sintering.structural_projection import eta_masses, project_eta_mass_preserving

NM = 1e-9
MPA = 1e6
MS = 1e-3


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-operator eta2/V2 ledger")
    p.add_argument("--preset", choices=["dev", "v64"], default="dev")
    p.add_argument("--nx", type=int)
    p.add_argument("--ny", type=int)
    p.add_argument("--dx-nm", type=float)
    p.add_argument("--r1-nm", type=float)
    p.add_argument("--r2-nm", type=float)
    p.add_argument("--r3-nm", type=float)
    p.add_argument("--interface-cells", type=float, default=4.0)
    p.add_argument("--aspect-ratio", type=float, default=2.0)
    p.add_argument(
        "--contact-orientation",
        choices=["short_plane", "long_plane"],
        default="short_plane",
    )
    p.add_argument("--sigma-target-mpa", type=float, default=75.0)
    p.add_argument("--time-ms", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-hazard-rbm",
        action="store_true",
        help="Disable hazard/RBM to isolate CH/Ostwald/structural relaxation",
    )
    return p


def _nm_or_none(value: float | None) -> float | None:
    return None if value is None else value * NM


def v2(e2: np.ndarray) -> float:
    return float(np.sum(e2, dtype=np.float64))


def main() -> None:
    a = parser().parse_args()
    cfg = ModelConfig(
        preset=a.preset,
        geometry="substrate",
        nx=a.nx,
        ny=a.ny,
        dx=_nm_or_none(a.dx_nm),
        r1=_nm_or_none(a.r1_nm),
        r2=_nm_or_none(a.r2_nm),
        r3=_nm_or_none(a.r3_nm),
        interface_cells=a.interface_cells,
        aspect_ratio=a.aspect_ratio,
        contact_orientation=a.contact_orientation,
        sigma_target=a.sigma_target_mpa * MPA,
        t_total=a.time_ms * MS,
        seed=a.seed,
        status_prints=False,
        event_prints=False,
        checkpoint=False,
    )
    p = build_params(cfg)
    rng = np.random.default_rng(cfg.seed)
    f, e1, e2, e3 = initialize_fields(p)
    s = Sink(threshold=float(rng.exponential()))
    null_sink = Sink()

    v0 = v2(e2)
    ledger = {
        "post_ch_constrained": 0.0,
        "ostwald": 0.0,
        "ac_raw": 0.0,
        "post_ac_constrained": 0.0,
        "rbm_raw": 0.0,
        "post_rbm_constrained": 0.0,
    }
    first = {k: 0.0 for k in ledger}
    nucleations = 0
    st = None

    for t in range(1, p.Nt + 1):
        # CH changes f only.  Constrained reconciliation preserves all eta masses.
        ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
        f = evolve_f(f, e1, e2, e3, s, null_sink, p)
        before = v2(e2)
        e1, e2, e3 = project_eta_mass_preserving(
            f, e1, e2, e3, p, target_masses=ch_targets
        )
        d = v2(e2) - before
        ledger["post_ch_constrained"] += d
        if t == 1:
            first["post_ch_constrained"] += d

        # Explicit Ostwald transfer is the physical grain-volume-change channel.
        before = v2(e2)
        f, e1, e2, e3 = ostwald_substrate(f, e1, e2, e3, p)
        d = v2(e2) - before
        ledger["ostwald"] += d
        if t == 1:
            first["ostwald"] += d

        # Structural smoothing must be mass neutral after constrained projection.
        ac_targets = eta_masses(e1, e2, e3, p.use_eta3)
        before = v2(e2)
        e1, e2, e3 = evolve_eta(e1, e2, e3, p)
        d = v2(e2) - before
        ledger["ac_raw"] += d
        if t == 1:
            first["ac_raw"] += d

        before = v2(e2)
        e1, e2, e3 = project_eta_mass_preserving(
            f, e1, e2, e3, p, target_masses=ac_targets
        )
        d = v2(e2) - before
        ledger["post_ac_constrained"] += d
        if t == 1:
            first["post_ac_constrained"] += d

        if not a.no_hazard_rbm:
            st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
            if stop:
                raise RuntimeError(f"diagnostic stopped at step {t}: {reason}")
            on = hazard_step(s, st, p, p.dt, rng)
            nucleations += int(on)

            if s.active:
                rbm_targets = eta_masses(e1, e2, e3, p.use_eta3)
                before = v2(e2)
                f, e1, e2, e3, _ = rbm(f, e1, e2, e3, s, p)
                d = v2(e2) - before
                ledger["rbm_raw"] += d
                if t == 1:
                    first["rbm_raw"] += d

                before = v2(e2)
                e1, e2, e3 = project_eta_mass_preserving(
                    f, e1, e2, e3, p, target_masses=rbm_targets
                )
                d = v2(e2) - before
                ledger["post_rbm_constrained"] += d
                if t == 1:
                    first["post_rbm_constrained"] += d

    vf = v2(e2)
    measured = vf - v0
    ledger_sum = sum(ledger.values())
    theoretical_ostwald = math.exp(-(p.Nt * p.dt) / p.tau_ripening) - 1.0

    print("\n=== eta2 / V2 PHYSICS OPERATOR LEDGER ===")
    print(f"preset:               {a.preset}")
    print(f"grid:                 {p.Nx} x {p.Ny}")
    print(f"dx:                   {p.dx*1e9:.6f} nm")
    print(
        f"R1/R2/R3:             {p.R1*1e9:.3f} / {p.R2*1e9:.3f} / "
        f"{p.R3*1e9:.3f} nm"
    )
    print(f"interface width W:    {p.interface_width*1e9:.3f} nm")
    print(f"R2/dx:                {p.R2/p.dx:.3f} cells")
    print(f"W/R2:                 {p.interface_width/p.R2:.6f}")
    print(f"dt:                   {p.dt:.9e} s")
    print(f"steps:                {p.Nt}")
    print(f"physical time:         {p.Nt*p.dt*1e3:.6f} ms")
    print(f"sigma target:          {p.sigma_target/1e6:.3f} MPa")
    print(f"hazard/RBM enabled:    {not a.no_hazard_rbm}")
    print(f"nucleations:           {nucleations}")
    if st is not None:
        print(f"final stress:          {st.sigma/1e6:.6f} MPa")
    print(f"final V2/V20:          {vf/v0:.12f}")
    print(f"net dV2/V20:           {measured/v0:+.12e}")
    print(f"ideal Ostwald-only:    {theoretical_ostwald:+.12e}")
    print("\nCumulative operator contributions (normalized by initial V2):")
    for k, value in ledger.items():
        print(f"  {k:26s} {value/v0:+.12e}")
    print(f"  {'LEDGER SUM':26s} {ledger_sum/v0:+.12e}")
    print(f"  {'closure error':26s} {(ledger_sum-measured)/v0:+.12e}")

    nonphysical = (
        ledger["post_ch_constrained"]
        + ledger["ac_raw"]
        + ledger["post_ac_constrained"]
        + ledger["rbm_raw"]
        + ledger["post_rbm_constrained"]
    )
    print(f"  {'NON-OSTWALD NET':26s} {nonphysical/v0:+.12e}")

    print("\nFirst-step contributions (normalized by initial V2):")
    for k, value in first.items():
        print(f"  {k:26s} {value/v0:+.12e}")


if __name__ == "__main__":
    main()
