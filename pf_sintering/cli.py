from __future__ import annotations

import argparse
from pathlib import Path

from .model import ModelConfig
from .runner import SinteringModel

NM = 1e-9
MPA = 1e6
US = 1e-6
MS = 1e-3


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-field sintering model (Python v64 port)")
    p.add_argument("--preset", choices=["dev", "v64"], default="dev", help="Development or full v64 dimensional defaults")
    p.add_argument("--geometry", choices=["substrate", "threeparticle"], default="substrate")
    p.add_argument("--nx", type=int, help="Grid columns; omit for automatic domain sizing")
    p.add_argument("--ny", type=int, help="Grid rows; omit for automatic domain sizing")
    p.add_argument("--dx-nm", type=float, help="Grid spacing in nm")
    p.add_argument("--r1-nm", type=float, help="Grain/substrate reference radius R1 in nm")
    p.add_argument("--r2-nm", type=float, help="Central/particle reference radius R2 in nm")
    p.add_argument("--r3-nm", type=float, help="Grain 3 radius R3 in nm")
    p.add_argument("--interface-cells", type=float, default=4.0, help="Diffuse interface width in grid cells")
    p.add_argument("--aspect-ratio", type=float, default=2.0)
    p.add_argument("--contact-orientation", choices=["short_plane", "long_plane"], default="short_plane")
    p.add_argument("--wall-frac", type=float, help="Substrate wall location as fraction of Nx")
    p.add_argument("--initial-overlap-nm", type=float)
    p.add_argument("--domain-margin-r2", type=float, default=3.0, help="Automatic-domain margin measured in R2")
    p.add_argument("--theta-mis-deg", type=float, default=30.0)
    p.add_argument("--sigma-target-mpa", type=float, default=75.0)
    p.add_argument("--temperature-k", type=float, default=1000.0)
    p.add_argument("--time-ms", type=float, help="Total simulated time; preset default if omitted")
    p.add_argument("--dt-us", type=float, help="Override stable timestep in microseconds (advanced)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, help="Frame cadence in fine steps")
    p.add_argument("--diag-every", type=int, help="Diagnostic cadence in fine steps")
    p.add_argument("--checkpoint-every", type=int, help="Checkpoint cadence in fine steps")
    p.add_argument("--hazard-every", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path("runs/dev"))
    p.add_argument("--out-tag", default="dev")
    p.add_argument("--restart", type=Path)
    p.add_argument("--overwrite", action="store_true", help="Allow a fresh run to replace an existing final output")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-event-prints", action="store_true")
    p.add_argument("--no-checkpoint", action="store_true")
    return p


def config_from_args(a: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        preset=a.preset, geometry=a.geometry,
        nx=a.nx, ny=a.ny,
        dx=None if a.dx_nm is None else a.dx_nm*NM,
        r1=None if a.r1_nm is None else a.r1_nm*NM,
        r2=None if a.r2_nm is None else a.r2_nm*NM,
        r3=None if a.r3_nm is None else a.r3_nm*NM,
        interface_cells=a.interface_cells,
        aspect_ratio=a.aspect_ratio, contact_orientation=a.contact_orientation,
        substrate_wall_frac=a.wall_frac,
        initial_overlap=None if a.initial_overlap_nm is None else a.initial_overlap_nm*NM,
        domain_margin_r2=a.domain_margin_r2,
        theta_mis_deg=a.theta_mis_deg, sigma_target=a.sigma_target_mpa*MPA,
        temperature=a.temperature_k,
        t_total=None if a.time_ms is None else a.time_ms*MS,
        dt_override=None if a.dt_us is None else a.dt_us*US,
        seed=a.seed,
        save_interval_steps=a.save_every, diag_every_steps=a.diag_every,
        checkpoint_interval_steps=a.checkpoint_every, hazard_every=a.hazard_every,
        output_dir=a.output_dir, out_tag=a.out_tag, restart_file=a.restart,
        status_prints=not a.quiet, event_prints=not a.no_event_prints,
        checkpoint=not a.no_checkpoint,
    )


def main() -> None:
    ap = parser()
    args = ap.parse_args()
    cfg = config_from_args(args)

    final = cfg.output_dir / f"sintering_{cfg.out_tag}_final.h5"
    if args.restart is None and final.exists() and not args.overwrite:
        ap.error(
            f"refusing to overwrite existing final output: {final}\n"
            "Use a distinct --output-dir/--out-tag for a new case, or pass --overwrite intentionally."
        )

    summary = SinteringModel(cfg).run()
    if not args.quiet:
        print("\nRun complete")
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
