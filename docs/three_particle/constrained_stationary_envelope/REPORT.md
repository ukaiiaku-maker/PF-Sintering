# Stationary-Lagrangian envelope-force qualification

**Decision: `ACTIVE_DOMAIN_CONTAMINATES_STATIONARY_FORCE`**

The force was evaluated at one stationary state from the complete constrained
Lagrangian,

```text
F_u = -(G_u + lambda^T C_u),
```

with both derivatives taken through the prescribed ownership frame at fixed
total-solid field. There are no first-moment constraints. The volume
multipliers were recovered on the same free set used by the box-constrained
stationary solver.

The implementation passes the independent 28 x 16 full-domain envelope
identity benchmark. Across four centered steps, the largest difference from
the derivative of independently minimized neighboring states is
`5.773e-06`.

At `u/b = 0.5`, fixed-field ownership steps from `0.1b` through `0.00625b`
give an effectively invariant full-domain force; their relative spread is
`5.297e-10`. Expanding the
production-sized stationary domain from the 25,242-cell `h16_t1e-4` mask to
all `76,590` cells changes the envelope force by
`2.807%`. The full-domain KKT
residual is `6.565e-08`.

The full-domain envelope force is
`3.125914971e-07 N`. The independent
Cannon--Carter force measured on the same converged morphology is
`4.040177507e-07 N`, a
`22.63%` mismatch. This exceeds
the prescribed 5% conjugacy gate.

The active cutoff contaminates the truncated stationary force, which is the
specified classification for the nonconverged active-domain sequence. The
full-domain reference removes that cutoff and supplies a stable numerical
derivative, but it still does not restore bicrystal conjugacy. Its secondary
decision is `FAILED_5_PERCENT_CANNON_CARTER_GATE`. The full-domain force is
therefore not accepted as the physical event force, and the conditional
force-versus-displacement trace was not run.

The earlier projected-Hessian diagnostics remain archived in
`../constrained_volume_only_bicrystal/hessian_spectra.csv`; all reported
minima there are positive. No C2 field was minimized or advanced. Root
kinetics, barriers, ordinary PF evolution, and the completed stochastic
trajectory are unchanged.
