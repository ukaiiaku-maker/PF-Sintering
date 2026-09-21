# C2 single-event thermodynamic qualification

## Decision

`C2_PF_AND_SHARP_FORCE_DISAGREE`

The selected RIGHT-contact positive displacement is not initially downhill on the fully relaxed, fixed-individual-volume PF branch. The stationary-envelope force at the preserved root is `-6.04736e-8 N`, whereas the KKT-pressure and global-CMC Cannon–Carter forces are both positive and approximately `4.451e-7 N`. The independent analytic PF configurational derivative agrees with the stationary-envelope force to a maximum relative discrepancy of `5.50e-9`; the inconsistency is therefore between the stationary PF branch and its sharp-interface reduction, not between the two PF force calculations.

The PF force crosses zero at the refined barrier maximum `u/b = 0.36702`, where `F_u = -7.40e-12 N`. The barrier is `4.41253e-18 J` above the stationary `u=0` reference. The `u=b` state is `4.58232e-18 J` below the reference, but reaching it requires traversing the initial uphill segment. A nucleated root therefore cannot yet be asserted to drive this path without an additional, physically specified coupling or energy contribution.

## Numerical qualification

All 21 coarse states from `u/b=0` to `1` and four force-zero refinements use the full 706,752-cell domain, fixed individual grain volumes, no moment constraints, no clipping, and no topology projection. Every archived state passed the `1e-7` projected KKT gate and the established volume tolerance. The largest relative grain-volume change is `3.33e-13`.

The trapezoidal PF work over the 21-point branch is `4.61824e-18 J`, compared with `-Delta G*=4.58232e-18 J`; the residual is `3.59e-20 J` (0.78%). The right-particle centroid advances `0.10250 nm` while the prescribed right GB frame advances `0.12500 nm`, demonstrating the required distinction between the coordinate and the mass centroid.

The selected neck changes by `-0.0732 nm`, the remote neck by `+0.0463 nm`, the local activation stress by `+0.0721 MPa`, and the remote pressure difference by `+0.00759 MPa`. These are legitimate full-field responses; no zero-remote-response constraint was imposed.

The historical stochastic trajectory remains unchanged. No descendant kinetics, avalanche, source-retention dynamics, or stochastic continuation was launched, and the repaired operator was not promoted to production.

## Artifacts

- `stationary_branch.csv`: 21-state PF/sharp force, pressure, stress, geometry, volume, centroid, and work ledger.
- `refined_force_zero.csv`: stationary refinement around the internal force zero.
- `summary.json`: machine-readable decision and headline metrics.
- `runs/c2_thermodynamic_single_event_first_right_root/`: full stationary fields, checkpoints, refined states, and per-state convergence metadata.
