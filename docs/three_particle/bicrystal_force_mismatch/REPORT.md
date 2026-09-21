# Full-domain bicrystal force-mismatch audit

**Decision: `CANNON_CARTER_METROLOGY_WAS_THE_MISMATCH`**

**Current status: `FULL_DOMAIN_PF_FORCE_RESOLVED_SHARP_INTERFACE_CONJUGACY_RESTORED`**

The stationary envelope force is `3.125914971e-07 N`. An
independent analytic shape derivative of the implemented diffuse PF
functional gives `3.125914971e-07 N`, a
`2.202e-11` relative
difference. It integrates `-(delta L/delta phi_i) phi_i,u` and independently
recovers the KKT pressures; it does not call the envelope derivative,
neighboring minimizations, or Cannon--Carter.

As a second check, the continuum axial Eshelby/Korteweg control-surface
resultant is `3.221519373e-07 N`, differing
from the discrete envelope by
`3.06%`.
Its bulk-side plateau is stable across the declared `2W` through `10W`
windows; the small offset is consistent with evaluating continuum gradients
on the finite grid rather than the exact face-energy ledger.

## What `local_reference` measured

`met["local_reference"]["s_local_cap"]` combines quadratic meridional
curvatures fitted over 15 nm on each side of the diffuse TJ with
`sin(160 deg/2)/r_neck`. Multiplying it by the contact area is algebraically
the Cannon--Carter line term minus a pressure term formed from that local
TJ-adjacent total curvature. It does not use the measured angle and it does
not use either grain's far-field constant-mean-curvature pressure.

The converged neck radius is `103.480960 nm`; the fitted left and right
slopes are `-0.088168446` and `0.142132393`. The
measured dihedral angle is `166.862596 deg`, versus the
prescribed `160 deg`. Complete local meridional, azimuthal, total-curvature,
area, line-force, and pressure-force values are archived in the CSV and JSON
tables.

The old local-reference construction gives
`4.040177507e-07 N`. Replacing its
TJ-adjacent curvature with the independently fitted far-field CMC pressure
difference gives `3.199581863e-07 N` with the 160-degree
angle and `3.255677977e-07 N` with the measured angle.
Their differences from the PF envelope are
`2.36%` and
`4.15%`,
respectively; both pass the 5% gate.

## Pressure check

The volume multipliers convert to capillary pressures with the exact solver
normalization `p_i=-lambda_i/(2*pi*dr*dz*L)`. They give
`18.839196` and `9.330246 MPa`. Area-weighted CMC
fits over `3W` to `8W` from the TJ give
`18.832682` and
`9.309998 MPa`. Thus the
stationary multipliers agree with the global surface curvatures while the
local TJ curvatures do not. The pressure difference used by the corrected
sharp-interface comparison is `9.522685 MPa`.

## PF and sharp-interface decompositions

For the implemented normalized-ownership functional, the ownership-dependent
terms are

```text
G_phi = integral [f Wc sum_(i<j)(phi_i phi_j)
                  + (k_eta/2) f sum_i |grad phi_i|^2] dV.
```

Their analytic configurational derivatives are

```text
delta L/delta phi_i
  = f Wc sum_(j!=i) phi_j - k_eta div(f grad phi_i) - p_i f,

F_config
  = -integral sum_i [(delta L/delta phi_i) phi_i,u] dV.
```

The report evaluates these expressions directly with the implemented
face-consistent divergence. This supplies the independent PF force quoted
above and establishes the limiting correspondence without assigning names by
analogy.

The PF envelope terms are `F_G=-G_u=1.649123105e-07 N` and
`F_C=-lambda^T C_u=1.476791866e-07 N`. They are both required, but they are not
separately the Cannon--Carter line and pressure forces. In the sharp ownership
limit the displaced midplane moves by `u/2`, so

```text
dV_0/du -> A/2,
F_C -> Delta-p A/2.
```

The explicit ownership derivative contains the complementary TJ/GB shape
term and the remaining pressure contribution. Integrating the stationary
Euler--Lagrange balance through the diffuse GB/TJ gives

```text
F_G -> F_line - 3 Delta-p A/2,
F_G + F_C -> F_line - Delta-p A = F_CC.
```

This also explains why assigning `F_G` to line tension and `F_C` to the full
pressure force would be incorrect. Numerically, the fixed-field reassignment
area is `1.553054627e-14 m^2`, approaching the sharp value
`A/2=1.682057273e-14 m^2`.

The Cannon--Carter component tables report line and pressure forces
separately for local measured-angle, local 160-degree, global-CMC, and KKT
pressure choices. Only the totals should be compared with the PF envelope.

## Decision and scope

The original 22.6% discrepancy came from using diffuse-TJ local curvature as
the pressure curvature. Global CMC curvature, KKT pressure, the independent
PF configurational force, and the stationary envelope now close within the
prescribed 5% tolerance. The conditional interface-width campaign was not
triggered because a residual PF-versus-sharp-interface mismatch is no longer
present after correcting the metrology.

No C2 state was minimized or advanced. No clipping, barrier change, root-law
change, or stochastic parameter change was made.
