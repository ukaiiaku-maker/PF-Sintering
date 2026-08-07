#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pf_sintering.model import ModelConfig, SinteringModel


def main() -> None:
    ap=argparse.ArgumentParser(description="Short-contact activation-stress sweep")
    ap.add_argument("--preset",choices=["dev","v64"],default="dev")
    ap.add_argument("--sigmas-mpa",type=float,nargs="+",default=[0.1,75.0,150.0])
    ap.add_argument("--time-ms",type=float)
    ap.add_argument("--nx",type=int);ap.add_argument("--ny",type=int);ap.add_argument("--dx-nm",type=float)
    ap.add_argument("--r2-nm",type=float);ap.add_argument("--aspect-ratio",type=float,default=2.0)
    ap.add_argument("--contact-orientation",choices=["short_plane","long_plane"],default="short_plane")
    ap.add_argument("--theta-mis-deg",type=float,default=30.0)
    ap.add_argument("--output-root",type=Path,default=Path("runs/short_activation_sweep"))
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=True)
    for sigma in args.sigmas_mpa:
        tag=(f"{sigma:g}MPa").replace(".","p")
        out=args.output_root/tag
        cfg=ModelConfig(preset=args.preset,geometry="substrate",nx=args.nx,ny=args.ny,dx=None if args.dx_nm is None else args.dx_nm*1e-9,r2=None if args.r2_nm is None else args.r2_nm*1e-9,aspect_ratio=args.aspect_ratio,contact_orientation=args.contact_orientation,theta_mis_deg=args.theta_mis_deg,sigma_target=sigma*1e6,t_total=None if args.time_ms is None else args.time_ms*1e-3,seed=args.seed,output_dir=out,out_tag=f"short_{tag}",live_plot=False)
        print(f"\n=== sigma_target={sigma:g} MPa -> {out} ===")
        SinteringModel(cfg).run()


if __name__=="__main__":main()
