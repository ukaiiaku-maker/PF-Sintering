# Ripening-versus-nucleation analytical design

**Status: analytical campaign complete. No PF field was initialized or evolved and no stochastic threshold was drawn.**

## Reduced kinetics and validation

The full-chain sharp energy is `F=gamma_s A_s + gamma_GB A_GB`. On each symmetric half-chain, the unit-x normal velocity determines `d(r Jx)/ds=-r u_n/Omega`. Using `M_atom=M_surface/Omega^2` makes the atomic and PF volume-flux dissipations identical, so the measured PF surface mobility is used without an adjustable factor. The resulting `xdot=-F'/zeta` is accepted only when `F'<0` throughout 0-20% loss.

Against the existing 0.65 post-transient-to-30-s interval, the reduced model predicts 31.672 s for the observed 0.8701% relative loss; PF took 29.597 s. The ratio is 1.070. No mobility or stress parameter was fitted.

## Search result

Of 14,400 signed-curvature designs, 5145 pass geometry and 6W curvature/contact resolution. Only **10** have spontaneous center shrinkage (`dF/dx<0`) across the full 0-20% interval; 7 of those are concave. Concavity can raise stress at large TJ radius, but most prescribed concave paths raise total interfacial energy and are rejected.

The root survival is evaluated as `S(x)=exp[-integral 2 Gamma/(1.25 xdot) dx]`. The primary outputs are the 10/50/90% first-root loss, stress, TJ resolution, and elapsed time.

## Prior balanced examples

| Target | Matched old-screen geometry | stress path | r20/W | thermodynamic result |
|---|---|---:|---:|---|
| approximately 18 to 45 MPa, r20/W>12 | `Ro150_q0.85_L10W_r1.00_th170_k0.75` | 18.0->45.3 MPa | 12.3 | reject: prescribed center shrink raises sharp-interface free energy |
| approximately 24 to 55 MPa, r20/W>10 | `Ro150_q0.65_L8W_r1.00_th170_k0.75` | 24.3->55.0 MPa | 10.6 | reject: prescribed center shrink raises sharp-interface free energy |

## Discussion candidates

These candidates are retained for analytical discussion, not PF promotion:

| Role | Geometry | k0 | sigma0 | x50 | sigma(x50) | t50 | Gamma50/Gamma0 | r50/W |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| largest shrinkage before median nucleation | `Ro100_q0.65_L8W_r1.60_th140_km+5` | 5.0/um | 22.11 MPa | 0.358% | 22.14 MPa | 7.51 s | 0.999 | 25.95 |
| highest stress at median nucleation | `Ro100_q0.65_L8W_r1.60_th170_km-20` | -20.0/um | 48.74 MPa | 0.169% | 48.78 MPa | 3.54 s | 1.000 | 25.98 |
| largest hazard amplification at median | `Ro150_q0.65_L12W_r1.60_th170_km-10` | -10.0/um | 29.16 MPa | 0.034% | 29.16 MPa | 3.95 s | 1.000 | 38.99 |
| largest resolution margin | `Ro100_q0.75_L12W_r1.40_th140_km+5` | 5.0/um | 21.85 MPa | 0.257% | 21.87 MPa | 7.51 s | 1.000 | 26.21 |
| least concave thermodynamic candidate | `Ro150_q0.65_L12W_r1.60_th140_km+5` | 5.0/um | 13.07 MPa | 0.066% | 13.07 MPa | 7.35 s | 1.000 | 38.99 |

The full first-root distributions are:

| Candidate | x10 / x50 / x90 | sigma10 / sigma50 / sigma90 | t10 / t50 / t90 | rTJ/W at 10 / 50 / 90% |
|---|---:|---:|---:|---:|
| C1 | 0.055 / 0.358 / 1.176% | 22.112 / 22.142 / 22.223 MPa | 1.141 / 7.510 / 24.964 s | 25.99 / 25.95 / 25.84 |
| C2 | 0.026 / 0.169 / 0.559% | 48.743 / 48.779 / 48.879 MPa | 0.538 / 3.539 / 11.756 s | 26.00 / 25.98 / 25.92 |
| C3 | 0.005 / 0.034 / 0.114% | 29.158 / 29.163 / 29.175 MPa | 0.600 / 3.947 / 13.113 s | 39.00 / 38.99 / 38.98 |
| C4 | 0.039 / 0.257 / 0.845% | 21.852 / 21.874 / 21.931 MPa | 1.141 / 7.509 / 24.956 s | 26.24 / 26.21 / 26.13 |
| C5 | 0.010 / 0.066 / 0.219% | 13.072 / 13.075 / 13.084 MPa | 1.117 / 7.348 / 24.415 s | 39.00 / 38.99 / 38.95 |

## Numerical robustness

Repeating the projected kinetics with 401 and 801 meridional points changes `dF/dx` by at most 0.0031 fJ and `xdot` by less than 0.10% over the ten accepted paths. Half-chain flux closure improves to below 2.6e-6 at 801 points. These errors are small relative to the retained thermodynamic driving.

## Decision

The unchanged root process is likely to fire after only a small fraction of center-volume loss. The thermodynamically allowed paths therefore realize little of their nominal 20% stress reserve before median nucleation. Concavity raises the absolute initial stress while preserving sites, but within this fixed signed-curvature family it does not create large pre-root stress amplification. No geometry is selected and no PF run is authorized.
