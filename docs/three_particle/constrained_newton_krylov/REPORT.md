# Constrained Newton--Krylov bicrystal hard gate

**Decision: `MULTIPLE_METASTABLE_REDISPOSITION_BRANCHES`**

The new solver is a reduced-space projected L-BFGS method followed by a
damped Newton--Krylov polish. Its analytic axisymmetric Hessian action agrees
with a centered finite difference of the implemented gradient to
`9.203e-10` relative Linf error. It
uses an explicit box active set and never clips the phase field.

Every archived state closes the volume/first-moment constraints near machine
precision and reaches the declared projected KKT tolerance of `1e-7` on its
recorded active domain. The four prescribed redistribution guesses at
`u=0.5b` converge on one common active mask with an energy spread of
`2.974e-20 J`, below the `1e-19 J` independence tolerance.

The bicrystal hard gate nevertheless fails. Two `u=0` initial morphologies,
solved on the identical 45,427-cell union domain, remain separated by
`1.365e-18 J` after both satisfy the KKT tolerance. The local derivative
is also inconsistent with the independent capillary force: the
`Delta u=0.05b` estimate is `1.692737e-08 N`, whereas Cannon--Carter is
`1.253725e-07 N` (relative mismatch `86.50%`, tolerance `5%`).
The `0.10b`, `0.05b`, and `0.025b` estimates do not form a convergent sequence;
the finest pair switches among nearby stationary branches.

These are converged stationary states, so this outcome is not another
projected-gradient iteration-limit result. It shows that the presently defined
fixed-ownership, fixed-volume/first-moment manifold is multivalued and does not
yet supply a unique work-conjugate bicrystal branch. Per the hard-gate rule, no
C2 stationary solve, C2 continuation, energy derivative, stochastic run, or
production-operator promotion was performed.
