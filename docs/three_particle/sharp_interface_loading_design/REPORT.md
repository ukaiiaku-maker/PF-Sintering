# Sharp-interface three-particle loading design

**Status: analytical geometry campaign complete. No phase-field or stochastic trajectory was run.**

The campaign preserves the production bicrystal contact stress and the root barrier, site count, formation penalty, temperature, clock conversion, and 1.25 threshold multiplier from the authoritative manifest. It introduces no cluster stress, barrier fit, or event physics.

## Construction

The fixed center GB planes are separated by `Lc`. Writing the axisymmetric cross-sectional squared radius as `g=r^2`, the no-sink path uses `g(z,x)=g(z,0)-x Vc0/(pi Lc)`. Therefore `Vc(x)=Vc0(1-x)` exactly and `rTJ`, the contact slope, angle, and curvature follow analytically. Each outer particle receives `x Vc0/2`; a smooth quintic particle-plus-neck profile enforces its contact value, tangent, curvature, tip closure, and exact volume. This is a deliberately non-CMC kinematic family, suitable for screening rather than an equilibrium claim.

The exact contact expression is

`sigma_GB = -gamma_s (k_-+k_+)/2 + 3 gamma_s [sin(theta_-/2)+sin(theta_+/2)]/(2 r_TJ)`.

The inverse requirement is

`r_TJ,target = (3 gamma_s/2)[sin(theta_-/2)+sin(theta_+/2)] / [sigma_target + gamma_s (k_-+k_+)/2]`.

## Search and exclusions

The grid evaluated **14400** designs over `Ro=80-150 nm`, `Rc/Ro=0.45-0.85`, `Lc=8-24W`, `rTJ/Rc=0.60-1.60`, angles `120-170 deg`, and normalized local curvature `kR=0-0.75`. **5909** were geometrically admissible with positive loading through 20% loss; **2954** also began in the 10-30 s characteristic-wait screening band. Rejections preserve their explicit geometry, topology, monotonicity, or 6W contact-resolution reason.

## Kinetic finding

The unchanged kinetics do not identify 20-30 MPa with the proposed 0.5-3 s high-rate band. The TJ site count decreases as the contact shrinks, partially opposing stress activation. The admissible positive-loading designs actually span initial waits of 10.00-14.57 s inside the requested band; none reaches a 3 s characteristic wait by 20% center loss.

A stronger bound is independent of the chosen polynomial shape. For a flat or convex meridional jet, the most favorable possible contact has both angles at 180 degrees and zero curvature penalty, so `sigma <= 3 gamma/rTJ`. At the conservative resolved limit `rTJ=6W=24 nm`, this gives at most 125 MPa and the unchanged root law gives a minimum characteristic wait of 6.43 s. Even at `3W=12 nm`, the favorable bound is about 3.92 s. Entering the 3 s band requires roughly a 2W contact, which is not resolved for production metrology. A sufficiently negative meridional curvature could evade this bound, but that would be a qualitatively concave groove and is outside the screened convex particle-plus-neck family.

For the requested representative values `theta_-=theta_+=160 deg` and mean curvature `3.36 /um`, the inverse formula gives `rTJ=126.47, 104.18, 88.56 nm` at 20, 25, and 30 MPa respectively.

## Current 0.65 reference

The retained symmetry-released PF reference covers 0.986-2.125% loss. It moves from 18.018 to 18.564 MPa and its characteristic wait changes from 9.600 to 9.444 s. Its large ~137.6 nm contact explains the modest stress reserve; the analytical analogue is included to 20% only as a kinematic extrapolation.

## Discussion candidates

No single scalar objective selected a winner. These are distinct representatives from the Pareto map and the admissible screening pool:

| Role | Geometry | sigma0 | sigma20 | reserve | tau0 | tau20 | rTJ20/W | Δsigma radius / angle / curvature |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| largest reserve | `Ro150_q0.75_L24W_r0.60_th170_k0.75` | 38.44 MPa | 102.14 MPa | 63.70 MPa | 10.10 s | 8.49 s | 6.13 | 77.05 / -2.16 / -11.05 MPa |
| largest resolution margin | `Ro120_q0.75_L24W_r1.00_th170_k0.50` | 28.35 MPa | 32.67 MPa | 4.33 MPa | 10.12 s | 10.26 s | 19.49 | 5.13 / -0.05 / -0.76 MPa |
| slowest admissible initial clock | `Ro80_q0.65_L8W_r1.40_th120_k0.75` | 23.79 MPa | 26.05 MPa | 2.26 MPa | 14.57 s | 15.28 s | 16.06 | 4.67 / -1.28 / -1.13 MPa |
| greatest kinetic acceleration | `Ro150_q0.55_L10W_r0.80_th170_k0.75` | 38.24 MPa | 98.74 MPa | 60.51 MPa | 10.39 s | 8.83 s | 6.19 | 74.95 / -2.00 / -12.32 MPa |
| least loss to cross 30 MPa from below | `Ro100_q0.75_L8W_r1.00_th150_k0.75` | 29.89 MPa | 45.65 MPa | 15.76 MPa | 11.58 s | 12.42 s | 11.48 | 23.87 / -2.66 / -5.45 MPa |

For each candidate, `candidate_loading_paths.csv` records all requested milestones, both one-sided curvatures and angles, one-sided and production stresses, unchanged root rate and barrier, exact volume ledger, and the radius/angle/curvature contributions to `d sigma/dx`.

## Decision

These candidates justify discussion of a later PF initialization study, but they do not authorize one. The analytical family shows how smaller, contracting contacts increase stress while also showing the countervailing loss of TJ sites and resolution margin. Review the five geometries and choose the acceptable balance of body shape, contact resolution, and loading reserve before constructing any diffuse field.
