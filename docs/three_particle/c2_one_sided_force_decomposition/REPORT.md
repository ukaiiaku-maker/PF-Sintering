# C2 unilateral RIGHT-event force decomposition

**Decision: `C2_EVENT_FORCE_DEPENDS_ON_TRANSPORT_RESERVOIR`**

No finite event or descendant was constructed.  Every calculation starts from
the untouched first stochastic RIGHT-root snapshot at 2.772296142578125 s.

## Artifact reconciliation

The retained event operator is the literal rigid-frame construction archived
with commit `00154de`: (U_L=U_C=0), (U_R=-u), (q_L=0), and (q_R=u).
Its negative branch opens internal free surface and remains archived only as a
contact-cusp diagnostic.  It is not used here.

The event-direction force is the unilateral limit

\[
F_R^+=-\left.\frac{dG}{du}\right|_{0^+}.
\]

The earlier projected simplex-tangent value is not retained or reported as an
event force. The current implementation, audit script, CSV files, and summary
all use the literal positive event operator.

## Exact PF derivative

The exact discrete derivatives of the archived multigrain energy were
contracted with the literal positive direction:

\[
F_f=-\int \mu_f f_{,u}\,dV,\qquad
F_\phi=-\sum_i\int\frac{\delta G}{\delta\phi_i}\phi_{i,u}\,dV.
\]

For the existing (W)-wide TJ Gaussian reservoir,

| contribution | force (N) |
|---|---:|
| (F_{\rm swept}) | (+3.72800892\times10^{-7}) |
| (F_{\rm deposit}) | (-1.70743063\times10^{-7}) |
| (F_f) | (+2.02057829\times10^{-7}) |
| (F_\phi) | (-3.62302093\times10^{-8}) |
| (F_R^+=F_f+F_\phi) | (+1.65827619\times10^{-7}) |

The one-sided finite-energy difference converges to the analytic contraction.
At the finest step, (delta u/b=1.5625\times10^{-4}), it is
(1.65756249\times10^{-7}) N, 0.0430% below the exact directional value.
The refinement table is in `one_sided_force_refinement.csv`.

## Reservoir sensitivity and chemical potential

Only the conservative redeposition support was changed.  Rigid translation,
ownership displacement, swept field, and (F_\phi) remained identical.

| reservoir | (F_R^+) (N) | mean source (mu) (MPa) |
|---|---:|---:|
| TJ Gaussian, (W) | (1.65827619\times10^{-7}) | 14.7809 |
| TJ Gaussian, (2W) | (1.65572554\times10^{-7}) | 14.8030 |
| TJ Gaussian, (3W) | (1.67417092\times10^{-7}) | 14.6433 |
| distributed nearby free surface | (1.78344619\times10^{-7}) | 13.6973 |
| low-(mu)-weighted nearby free surface | (1.79317176\times10^{-7}) | 13.6131 |

The force range is (1.37446\times10^{-8}) N, or 8.02% of the five-support
mean.  This dependence is larger than the numerical refinement errors and is
caused entirely by the redeposition chemical potential.

With

\[
\dot V_{\rm swept}=-\int f_{,u}^{\rm swept}dV
=\int f_{,u}^{\rm deposit}dV,
\]

the sign and normalization identities are

\[
F_{\rm swept}=\dot V_{\rm swept}\bar\mu_{\rm swept},\qquad
F_{\rm deposit}=-\dot V_{\rm swept}\bar\mu_{\rm source},
\]

and therefore

\[
F_f=\dot V_{\rm swept}
(\bar\mu_{\rm swept}-\bar\mu_{\rm source}).
\]

Here (dot V_{\rm swept}=1.15516233\times10^{-14} {m m^2}) and
(ar\mu_{\rm swept}=32.2726) MPa for every reservoir.  Both identities
close to the printed floating-point precision in `reservoir_sensitivity.csv`.
Thus the transport affinity is the swept/contact-to-reservoir chemical
potential difference; it is not fixed by rigid-body kinematics alone.

## Dynamic configurational balance

The canonical PF resultant was evaluated on cross-sections at
(2W,3W,5W,8W,) and (12W) on both sides of the RIGHT contact.  No KKT
pressure or stationarity assumption was introduced.  The bulk
configurational-force density was calculated directly from the Euler--Lagrange
residual,

\[
f_z^{\rm config}=\mu_f f_{,z}
+\sum_i\frac{\delta G}{\delta\phi_i}\phi_{i,z}.
\]

For all five control volumes, the measured surface-resultant difference agrees
with the integrated bulk residual within 1.01--4.15%.  The control-surface
dependence is therefore quantitatively explained by the nonstationary bulk
residual.  The PF configurational balance is consistent at the resolution of
this continuum-gradient diagnostic.

## Curvature audit

Total curvature was measured in (3W)-(5W), (5W)-(8W), and
(8W)-(12W) windows.  The center-particle branches meet the declared 5%
plateau criterion.  The two outer-particle branches do not: their within-window
relative standard deviations reach 15.86%, and their window means vary by
79.19%.  Consequently there is no qualified particle-pair CMC pressure
difference at this dynamic root.  A Cannon--Carter number may be quoted only
as a local-equilibrium reference, not as an instantaneous conjugacy gate.

## Interpretation

The unilateral PF directional derivative is well resolved for any specified
transport reservoir, and its analytic decomposition agrees with direct
one-sided energy differences.  The event operator nevertheless lacks a unique
transport path: changing only where the swept material is redeposited changes
the driving force by 8.02%.  A physical event model must specify that transport
reservoir or solve the transport path dynamically before a unique event force
or finite event can be assigned.

```text
C2_EVENT_FORCE_DEPENDS_ON_TRANSPORT_RESERVOIR
finite event: NOT LAUNCHED
descendants: NOT CONSTRUCTED
```
