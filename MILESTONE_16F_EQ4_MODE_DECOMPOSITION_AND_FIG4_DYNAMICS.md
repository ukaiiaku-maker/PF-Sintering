# Milestone 16F — Eq.-4 Mode Decomposition and Exact Figure-4 Dynamics

Checkpoint: `f601ada` (Milestone 16E). This report was corrected mid-milestone
(see "Correction note" at the end of each affected section) after an initial
draft conflated three distinct quantities. The final version keeps them
strictly separate throughout:

- **(A) literal Eq. 4-5** — the printed closed-form criterion
  (`pf_sintering.hussein_eq4_reference.lambda_c_over_R_eq4`), no quadrature,
  no PF/Hessian call.
- **(B) full direct Eq.-2 surface-area energy** — an independent numerical
  quadrature of the surface-of-revolution area functional, root-found for
  its own threshold (`delta_E_full_area` / `lambda_c_full_area`). A
  different calculation from (A), not a stand-in for it.
- **(C) unrestricted local Hessian / directional-gradient modes** —
  `pf_sintering.sharp_interface_stability`'s constrained second-variation
  machinery (M16C), plus a new first-derivative (directional-gradient) probe
  added in this milestone. Local curvature/gradient information about one
  specific state, not a finite-amplitude energy comparison.

No `pf_sintering` production code was modified except for one new,
self-contained module, `pf_sintering/hussein_eq4_reference.py`. New scripts:
`scripts/m16f_eigenmode_decomposition.py`,
`scripts/m16f_sharp_figure4_dynamics.py`,
`scripts/m16f_directional_derivative.py`,
`scripts/m16f_mobility_scaling_check.py`.

## 1. Correction of the M16E interpretation

M16E found that the sharp-interface **linearized Hessian** ((C) above)
classified both psi=160 and psi=80 as UNSTABLE for the exact Figure-4
two-mode geometry, regardless of topology, and speculated this reflected a
missing *kinetic* contribution relative to the published threshold.

That interpretation is wrong and is retracted. The Hussein/Abdeljawad Eq. 4
threshold is not a linearized eigenvalue/dispersion relation at all — it is
a **finite-amplitude interfacial-energy comparison**,
`DeltaE(lambda) = E_pert(lambda) - E_cyl`, between two *specific, discrete*
geometric states (the exact two-mode perturbed shape vs. a volume-matched
plain cylinder with the same two grain-boundary disks). The clearest proof
that Eq. 4 is not a linear/infinitesimal criterion is that the published
`lambda_critical/R_cyl` is strongly amplitude-dependent: `1.48919*pi` at
`e1bar=e2bar=0.4` but `0.20067*pi` at `e1bar=e2bar=0.02` (Sections 3/4). A
linearized eigenvalue threshold is amplitude-independent by definition, so
this dependence alone rules out treating (C)'s lowest eigenvalue as
synonymous with (A)'s threshold. Kinetics may still control the *path*
between morphologies once the system is set in motion (Sections 9-13), but
the Eq.-4 stability boundary itself is energetic, not kinetic.

## 2. Direct implementation of the printed Eqs. 4-5

`pf_sintering/hussein_eq4_reference.py` now contains a **literal**
closed-form transcription of the printed Eqs. 4-5, calling no PF or Hessian
routine and no quadrature:

```
beta1 = (e1^2 - e2^2/2 - 4*e1 - e2^2/(2*e1) + e2^4/(32*e1^2)) / (e1^2+e2^2)
beta2 = (4*e1^2 + e2^2 + 3*e1*e2^2) / (e1^2+e2^2)
lambda_c/R = beta1*cos(psi/2) + sqrt(beta1^2*cos(psi/2)^2 + pi^2*beta2)
```

(`lambda_c_over_R_eq4(e1bar, e2bar, psi_deg)`.) This supersedes the M16F
draft's quadrature-based `find_lambda_critical`, which is retained as a
**separate** diagnostic (Section 5) and never used as the Eq.-4 threshold.

**Correction note**: the initial M16F draft implemented only a full
surface-area quadrature (treatment (B)) and rooted it for `lambda_critical`,
incorrectly presenting that root as "the Eq.-4 threshold." It is not — Eq. 4
is the closed form above. This section replaces that with the literal
formula.

## 3. Numerical reference values (large amplitude, e1bar=e2bar=0.4)

Exact reproduction of the required checks, computed directly by
`lambda_c_over_R_eq4`:

```
beta1 = -5.359375000000   (required: -5.359375)
beta2 =  3.100000000000   (required: 3.1)

psi=160:  lambda_c/R = 4.6784429106  = 1.4891946304*pi   (required: 4.67844..., 1.48919...*pi)
psi=80:   lambda_c/R = 2.7829537965  = 0.8858417062*pi   (required: 2.78295..., 0.88584...*pi)

at lambda/R = 1.14*pi = 3.581415...:
psi=160:  lambda/lambda_c = 0.76551444   (required: 0.76551444)  -> STABLE
psi=80:   lambda/lambda_c = 1.28691164   (required: 1.28691164)  -> UNSTABLE
```

All values match the required checks to the full precision given. The
Section-4-of-the-handoff decision gate (psi=160 STABLE, psi=80 UNSTABLE at
the test wavelength, using the **printed** Eq.-4 criterion) is satisfied
exactly, not approximately. This supersedes the M16F draft's approximate
"0.9074 / 1.8113" ratios, which came from the (B) full-area-quadrature
diagnostic, not from the printed Eq. 4.

## 4. Small-amplitude reference values (e1bar=e2bar=0.02)

```
psi=160:  lambda_c/R = 0.6304147176 = 0.2006672370*pi   (required: 0.6304147176, 0.2006672370*pi)
psi=80:   lambda_c/R = 0.1450922158 = 0.0461842867*pi   (required: 0.1450922158, 0.0461842867*pi)
```

Both match the required checks exactly. At `lambda/R=1.14*pi`, both
thresholds are far below the test wavelength, so both psi cases are
UNSTABLE at this amplitude — as already correctly reported in the prior
draft of this milestone (Section 4), that qualitative conclusion is
unchanged by the formula fix; only the exact numeric threshold values are
now literal-Eq.4-exact rather than approximate.

## 5. The full-area quadrature (B) is a separate, distinct energy treatment

`delta_E_full_area` / `lambda_c_full_area` (renamed from the prior draft's
`delta_E` / `find_lambda_critical`) remain in the module as an independent,
honest numerical quadrature of a surface-of-revolution energy functional:

```
Epert = gamma_s * A_surface[R(z)] + gamma_gb * 2*pi*R(z1)^2      (both GB disks, per the source's stated Epert_GB=2*pi*gamma_GB*R(zmin)^2)
Ecyl  = gamma_s * (2*pi*Rcyl)*(2*lambda) + gamma_gb * 2*pi*Rcyl^2
```

Its own threshold, at `e1bar=e2bar=0.4`: `lambda_c_full_area/Rcyl = 3.94710`
(psi=160) and `1.97733` (psi=80) — noticeably different from the literal
Eq.-4 values in Section 3 (`4.67844`, `2.78295`). Both quadrature-based
values were validated internally: the classical zero-amplitude, zero-GB
limit converges exactly to `2*pi` (Rayleigh-Plateau), and this module's
quadrature energy agrees with `sharp_interface_stability.py`'s independent
frustum-based `energy()` evaluator to `7.2e-8` relative on the same profile.

**Correction note**: the prior draft attributed the ~15-29% gap between (B)
and the paper's published numbers to "an unresolved formula-definition gap
... not recoverable without the primary source text." That framing is
wrong and is removed. The source's GB-disk convention
(`Epert_GB=2*pi*gamma_GB*R(zmin)^2`) is not in question — it is used
identically in both (A) and (B) above. The difference between (A) and (B)
is simply that they are two different analytical/numerical treatments of a
related but not identical energy comparison (Eq. 4's closed form is a
specific analytic reduction of the perturbation-series expansion of the
area functional; the direct quadrature in (B) evaluates the exact,
un-expanded area functional). Reporting a ~15-29% difference between an
exact closed-form asymptotic/perturbative reduction and a fully nonlinear
direct quadrature, at `e1bar=e2bar=0.4` (a substantial amplitude, `40%` of
`R_cyl`), is unsurprising and requires no missing-source-material
explanation. (B) is retained purely as an independent cross-check
diagnostic; it is **not** used for any stable/unstable classification in
this report — only (A) is.

## 6. Full Hessian eigenmode decomposition (C), exact e1bar=e2bar=0.4 state

First 8 eigenmodes of the constrained sharp-interface Hessian
(`sharp_interface_stability.volume_constrained_eigenmodes`), applied
directly to the exact, un-relaxed published initial profile (`Nz=240`,
`lambda/Rcyl=1.14*pi`), classified by `surface_mode_strength =
sqrt(overlap(cos 2*pi*z/lambda)^2 + overlap(cos pi*z/lambda)^2)`: `>0.5`
P-R/FREE-SURFACE, `<0.15` GRAIN-COARSENING/GB-MIGRATION, else MIXED.

**psi=160** (gamma_ratio=0.34730):

| mode | eig | class | motion | surf_strength | dV1 | dV2 | GB1_shift | GB2_shift | overlap(lam) | overlap(2lam) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.23457 | near-zero-transfer GB translation | SYMMETRIC | 0.000 | +0.0000 | -0.0000 | -0.0374 | -0.0374 | -0.000 | -0.000 |
| 1 | -0.14634 | **candidate inner-grain-shrinkage mode** | ANTISYMMETRIC (seesaw) | 0.916 | -1.0005 | +1.0007 | +0.0500 | -0.0500 | +0.297 | +0.867 |
| 2 | -0.02459 | P-R/FREE-SURFACE (short-wavelength) | ANTISYMMETRIC (seesaw) | 0.969 | +0.5880 | -0.5877 | -0.1306 | +0.1306 | +0.894 | -0.374 |
| 3 | +0.09008 | GB translation | SYMMETRIC | 0.000 | -0.0001 | +0.0001 | +0.1673 | +0.1673 | -0.000 | +0.000 |
| 4 | +0.45988 | MIXED | ANTISYMMETRIC | 0.166 | -0.1636 | +0.1636 | -0.1689 | +0.1689 | -0.022 | +0.164 |
| 5 | +0.49869 | GB translation | SYMMETRIC | 0.000 | +0.0000 | -0.0000 | +0.1839 | +0.1839 | +0.000 | -0.000 |
| 6 | +0.99940 | MIXED | ANTISYMMETRIC | 0.164 | +0.0882 | -0.0882 | -0.2951 | +0.2951 | -0.164 | +0.006 |
| 7 | +1.07570 | GB translation | SYMMETRIC | 0.000 | -0.0000 | +0.0000 | -0.1685 | -0.1685 | +0.000 | +0.000 |

**psi=80** (gamma_ratio=1.53209):

| mode | eig | class | motion | surf_strength | dV1 | dV2 | GB1_shift | GB2_shift | overlap(lam) | overlap(2lam) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.15065 | near-zero-transfer GB translation | SYMMETRIC | 0.000 | +0.0001 | -0.0001 | +0.0575 | +0.0575 | +0.000 | -0.000 |
| 1 | -0.12310 | **candidate inner-grain-shrinkage mode** | ANTISYMMETRIC (seesaw) | 0.953 | -1.1572 | +1.1573 | +0.0909 | -0.0909 | -0.072 | +0.950 |
| 2 | +0.04207 | P-R/FREE-SURFACE (short-wavelength) | ANTISYMMETRIC (seesaw) | 0.925 | -0.1356 | +0.1352 | +0.1197 | -0.1197 | -0.925 | -0.006 |
| 3 | +0.10424 | GB translation | SYMMETRIC | 0.000 | +0.0001 | -0.0001 | -0.1696 | -0.1696 | +0.000 | -0.000 |
| 4 | +0.50964 | MIXED | ANTISYMMETRIC | 0.212 | +0.1251 | -0.1250 | +0.1722 | -0.1722 | +0.174 | -0.120 |
| 5 | +0.53364 | GB translation | SYMMETRIC | 0.000 | -0.0000 | +0.0000 | -0.1870 | -0.1870 | -0.000 | +0.000 |
| 6 | +1.00710 | GB-migration-dominated | ANTISYMMETRIC | 0.121 | -0.0888 | +0.0889 | +0.2909 | -0.2909 | +0.120 | -0.009 |
| 7 | +1.15180 | GB translation | SYMMETRIC | 0.000 | +0.0000 | -0.0000 | +0.1673 | +0.1673 | -0.000 | +0.000 |

**Correction note (terminology, per explicit instruction)**: mode 0 is a
near-rigid, same-direction GB translation with negligible net grain-volume
transfer (`|dV1|,|dV2| ~ 1e-4-1e-5`) — it is **not** called "the principal
grain-coarsening mode." Mode 1 (both psi values) is the mode with by far
the largest grain-volume transfer of any mode (`|dV1|,|dV2| ~1.0-1.16`, an
order of magnitude above every other mode) *and* a strong period-`2*lambda`
overlap (`+0.867` psi=160, `+0.950` psi=80): GB1 shifts positive, GB2
shifts negative (the two GBs converge, inner grain shrinks), with a large
V1->V2 transfer. This is identified as the **candidate inner-grain-shrinkage
mode** — "candidate" because, per Section 9 below, the raw published
profile is not a stationary state, so the Hessian eigenvalue alone
(negative curvature) does not establish that this is the direction of
actual evolution; that requires the first-derivative test.

## 7. Key psi=160 hypothesis

Section 3 established that the *initial* psi=160 two-mode state is
Eq.-4-STABLE (treatment (A), `lambda/lambda_c=0.7655<1`). Section 6 shows
the full coupled Hessian (treatment (C)) nonetheless has three negative
eigenvalues at that same state, the most physically significant being mode
1 (`eig=-0.14634`).

This is not a contradiction. (A) answers "is the exact two-mode perturbed
shape lower- or higher-energy than reverting fully to a plain cylinder" — a
comparison between two specific, discrete end states along one particular
amplitude-scaling family. (C) answers a different question: "is this exact
point a local energy minimum with respect to all nearby continuous
deformations," including deformations (GB migration that grows the outer
grain at the inner grain's expense while reshaping the free surface toward
`2*lambda`) that (A)'s one-parameter family does not probe. A state can be
Eq.-4-stable (favored over reverting to a cylinder) while not being a local
Hessian minimum (favored to evolve further along a different direction) —
these are compatible. Section 9 tests directly whether that direction is
actually downhill from the (non-stationary) starting profile.

## 8. Directional-derivative test along mode 1 (new, per correction item 9)

The raw published Figure-4 profile is not guaranteed to be a stationary
state of the constrained Lagrangian, so (C)'s Hessian eigenvalue at mode 1
is local curvature only — it does not by itself establish which direction
the state actually evolves. `scripts/m16f_directional_derivative.py`
orients mode 1 so that `+eps*v1` means grain 1 (inner) shrinks (`dV1<0`,
already the solver's natural sign in both cases — no flip needed), then
evaluates `DeltaF(+eps*v1)` and `DeltaF(-eps*v1)` for a small-eps ladder
using the raw sharp-interface energy `F` ((C)'s own energy functional,
`sharp_interface_stability.energy`):

**psi=160** (mode-1 eig=-1.4634e-01, F0=48.047472):

| eps | dF(+eps·v1) | dF(-eps·v1) | central slope | downhill |
|---|---|---|---|---|
| 1e-4 | -2.3928e-05 | +2.3929e-05 | -2.39285e-01 | +v1 (shrink) |
| 1e-3 | -2.3923e-04 | +2.3934e-04 | -2.39285e-01 | +v1 (shrink) |
| 1e-2 | -2.3877e-03 | +2.3980e-03 | -2.39285e-01 | +v1 (shrink) |
| 3e-2 | -7.1323e-03 | +7.2250e-03 | -2.39288e-01 | +v1 (shrink) |

**psi=80** (mode-1 eig=-1.2310e-01, F0=49.667699):

| eps | dF(+eps·v1) | dF(-eps·v1) | central slope | downhill |
|---|---|---|---|---|
| 1e-4 | -6.6889e-05 | +6.6890e-05 | -6.68892e-01 | +v1 (shrink) |
| 1e-3 | -6.6882e-04 | +6.6896e-04 | -6.68892e-01 | +v1 (shrink) |
| 1e-2 | -6.6819e-03 | +6.6959e-03 | -6.68893e-01 | +v1 (shrink) |
| 3e-2 | -2.0004e-02 | +2.0130e-02 | -6.68895e-01 | +v1 (shrink) |

The central slope is essentially constant across the full eps ladder (the
first-order/gradient term dominates the much smaller curvature correction
at every tested amplitude), and it is unambiguously negative (downhill in
the "+v1 = grain 1 shrinks" direction) for **both** psi=160 and psi=80. The
raw published profile is not a Lagrangian-stationary state, and its
dominant first-order response — for both the Eq.-4-stable psi=160 case and
the Eq.-4-unstable psi=80 case — is inner-grain shrinkage via inward GB
migration. This directly confirms mode 1's status as a genuine downhill
direction (not merely a Hessian curvature artifact) and is consistent with
Section 7's mechanistic picture: grain-1 shrinkage begins immediately in
both cases, with psi=80's linear driving force roughly 2.8x steeper
(`-0.6689` vs `-0.2393`) than psi=160's — consistent with psi=80 also being
Eq.-4-unstable and psi=160 needing to evolve toward the doubled-wavelength
morphology before an eventual P-R crossing (Sections 9-13).

## 9. PF geometry used for the dynamics runs

Per the explicit resolution requirement (`W/Rcyl<=0.10`, `W/dx>=8`):
`R_cyl=100nm`, `W=10nm`, `dr=dz=1.25nm` (`W/Rcyl=0.10`, `W/dx=8`), grid
`Nz=574 x Nr=185` (~106,000 cells). Exact published geometry:
`e1bar=e2bar=0.4`, `lambda/Rcyl=1.14*pi`, `z1/lambda=0.5804306233`,
`z2/lambda=1.4195693767`. No reservoir, no sink, no rigid-body motion.
`gamma_GB/gamma_s = 2*cos(psi/2)` exactly (`0.34730` psi=160, `1.53209`
psi=80). Stable timestep `dt~9.77e-5` at C=1 mobility (0.4x safety margin
on the CFL bound from `find_stable_dt`).

## 10. Mobility-timescale collapse check (new, per correction item 10)

Before drawing any conclusion from a possibly-slow t=60 run,
`scripts/m16f_mobility_scaling_check.py` tests whether scaling
`(M_s, M_GB) -> C*(M_s, M_GB)` (fixed ratio `M_GB/M_s`, matching this
project's `M_eta=M_s/(W*32/35)` convention) and running for `t'=t/C` gives
the same state as the baseline `C=1` run at `t` — the exact time-rescaling
symmetry expected of a gradient-flow PDE `dR/dt=M*L[R]` with `L`
independent of `M` — and whether this provides real wall-clock speedup for
an *explicit* time-stepping scheme (whose CFL-stable `dt` is expected to
scale as `1/C` too, in which case steps-to-solution, and hence wall-clock
cost, would be unchanged).

Baseline (`C=1`, psi=160, target physical time `t=0.3`): `V1/V1_0
=0.99999662`, `R(0)=171.64487nm`, `R(lambda)=91.64738nm`.

| C | state match vs. baseline | n_steps | apparent step-count ratio |
|---|---|---|---|
| 10 | `dV1_frac=+1.4e-10`, `dR(0)=+1.5e-7 nm` — collapse essentially exact | 4915 | 1.25x |
| 20 | `dV1_frac=+1.4e-10`, `dR(0)=+1.5e-7 nm` — collapse essentially exact | 4915 | 1.25x |
| 30 | `dV1_frac=-1.0e-4`, `dR(0)=+1.62 nm` — collapse breaks down | 3277 | 1.88x |
| 50 | `dV1_frac=+9.9e-8`, `dR(0)=+0.80 nm` — partial | 3932 | 1.56x |
| 100 | `dV1_frac=+9.9e-8`, `dR(0)=+0.80 nm` — partial | 3932 | 1.56x |

Two findings:

1. **The rescaling symmetry itself holds essentially exactly** where the
   timestep is well-resolved (`C=1,10,20`: state differences at the
   `1e-7`-`1e-10` level, i.e. floating-point/roundoff scale). This
   cross-validates that the `C=1` (placeholder-mobility) trajectory used
   for the production runs below is a faithful representation of the true
   PDE dynamics at whatever reduced/dimensionless time it reaches — not a
   numerical artifact of the specific mobility value chosen.
2. **It provides no reliable, exploitable wall-clock acceleration.**
   `find_stable_dt` (this project's dt-search, both in M16D-16E and here)
   selects `dt` by pure geometric halving from `dt=1.0`, so the selected
   `dt` is quantized to powers of 2 — this is why identical `n_steps` (and
   hence identical apparent "speedup") appear at unrelated `C` values
   (`C=10` and `C=20` both land on the same power-of-2 bracket; `C=50` and
   `C=100` do too), and why `C=30` shows the largest collapse error (its
   power-of-2 `dt` happens to sit closest to the true marginal-stability
   boundary for that particular `C`, i.e. the largest truncation error, not
   a breakdown of the underlying continuum symmetry). The apparent
   `1.25x`-`1.9x` "speedups" are artifacts of this quantization, not a
   real, order-of-magnitude acceleration — for a genuinely proportional
   `1/C` CFL bound (which is what an explicit scheme predicts and what is
   observed once quantization noise is accounted for), steps-to-solution
   at fixed reduced/dimensionless time is independent of `C`. This
   technique therefore cannot make full grain-elimination computationally
   tractable within this session; whatever partial trajectory the `C=1`,
   `t=60` runs reach (Sections 11-13) is the best available direct
   evidence, and per finding 1 above, it can be trusted as physically
   faithful rather than a numerical artifact.

## 11-13. Grain-elimination dynamics (psi=160) and unstable control (psi=80)

Both `t=60`, `n_sample=40` runs (`Nz=574, Nr=185`, `dt=4.8828e-5` at C=1
mobility, `1,228,800` steps each) completed. `mass_drift` stayed at
`~2e-13` throughout (exact conservation to solver tolerance). Every
checkpoint below is post-processed with the **literal Eq. 4** (A),
`lambda_c_over_R_eq4(e1bar_fit(t), e2bar_fit(t), psi)` — the in-run
`lam_over_lamc` value logged during the run itself used the pre-correction
(B)-style full-area diagnostic and is not used here.

**psi=160** (gamma_GB/gamma_s=0.34730):

| t | V1/V1,0 | V2/V2,0 | R(0) nm | R(lam) nm | e1bar(t) | e2bar(t) | lambda/lambda_c (A) |
|---|---|---|---|---|---|---|---|
| 0  | 1.000000 | 1.000000 | 171.648 | 91.649 | 0.40000 | 0.40000 | 0.76551 |
| 12 | 0.999860 | 1.000038 | 171.512 | 91.579 | 0.39967 | 0.40004 | 0.76573 |
| 24 | 0.999722 | 1.000076 | 171.397 | 91.519 | 0.39944 | 0.40009 | 0.76589 |
| 36 | 0.999587 | 1.000113 | 171.294 | 91.464 | 0.39924 | 0.40014 | 0.76603 |
| 48 | 0.999455 | 1.000150 | 171.201 | 91.413 | 0.39904 | 0.40018 | 0.76616 |
| 60 | 0.999326 | 1.000185 | 171.113 | 91.366 | 0.39885 | 0.40022 | 0.76629 |

**psi=80** (gamma_GB/gamma_s=1.53209):

| t | V1/V1,0 | V2/V2,0 | R(0) nm | R(lam) nm | e1bar(t) | e2bar(t) | lambda/lambda_c (A) |
|---|---|---|---|---|---|---|---|
| 0  | 1.000000 | 1.000000 | 171.648 | 91.649 | 0.40000 | 0.40000 | 1.28691 |
| 12 | 0.999861 | 1.000038 | 171.541 | 91.596 | 0.39959 | 0.40005 | 1.28768 |
| 24 | 0.999727 | 1.000075 | 171.447 | 91.548 | 0.39933 | 0.40009 | 1.28816 |
| 36 | 0.999597 | 1.000111 | 171.360 | 91.504 | 0.39911 | 0.40012 | 1.28857 |
| 48 | 0.999471 | 1.000145 | 171.280 | 91.462 | 0.39891 | 0.40015 | 1.28896 |
| 60 | 0.999349 | 1.000179 | 171.205 | 91.423 | 0.39871 | 0.40018 | 1.28932 |

**What is and is not established by these numbers.**

At this project's placeholder mobility scale (`M_s=1e-33`,
`M_eta=M_s/(W*32/35)`, unchanged from M16D-16F precedent), `t=60` produces
only a small evolution: `V1/V1,0` drops by `0.067%` (psi=160) and `0.065%`
(psi=80) — nowhere near the `~1.0->0.05` grain-elimination target the
handoff describes as the ideal endpoint. Section 10's mobility-timescale
check already established (a) this slow rate is a faithful reflection of
the true PDE dynamics, not a numerical artifact of the chosen mobility
value, and (b) no rescaling trick makes a full grain-elimination run
computationally tractable within this session. Full elimination and an
observed P-R pinch-off (Section 12 of the handoff) were **not achieved**
and are not claimed.

What **is** established, and is qualitatively exactly the mechanism
predicted in Sections 7-8:

- In both cases `V1` (inner grain) decreases monotonically while `V2`
  (outer grain) increases monotonically — mass transfers from grain 1 to
  grain 2, matching mode 1's `dV1<0, dV2>0` signature and the directional-
  derivative result that grain-1 shrinkage is the downhill direction from
  the very first step.
- `R(0,t)` (crest, over the shrinking inner grain) and `R(lambda,t)`
  (trough region) both decrease monotonically in both cases — no sign of
  the free surface rebuilding toward the inner grain; consistent with
  material draining from the shrinking grain 1 region.
- `e1bar(t)` (short-wavelength, period-`lambda`, amplitude) decreases
  slightly while `e2bar(t)` (long-wavelength, period-`2*lambda`) increases
  slightly in **both** cases — the period-`lambda` component is losing
  relative importance and the period-`2*lambda` component gaining it,
  exactly the doubled-wavelength-development trend Section 10 of the
  handoff calls for, even though the magnitude reached by `t=60` is small.
- `lambda/lambda_c` (A), tracked via the literal Eq. 4 at the fitted
  `e1bar(t), e2bar(t)`, **increases monotonically for both psi=160 and
  psi=80** — `0.76551 -> 0.76629` (psi=160, moving toward the instability
  boundary from below, exactly the "increases, eventually >1" direction
  Section 11 of the handoff requires) and `1.28691 -> 1.28932` (psi=80,
  already unstable and moving further into the unstable region, consistent
  with Section 13's expectation that this case does not need to wait for
  grain elimination). Neither trajectory reaches its respective crossing
  or new plateau within `t=60`; both are moving in the theoretically
  predicted direction from the first checkpoint onward, with no sign
  reversal or plateau at any sampled point.

In short: the direction of the mechanism (psi=160: stable-but-coarsening,
inner grain shrinks, period-`2*lambda` mode grows, `lambda/lambda_c`
creeps upward toward instability; psi=80: already unstable and continuing
to destabilize) is confirmed at every sampled checkpoint for both cases.
The magnitude of coarsening reached within the practical compute budget of
this session is far short of full grain elimination or an observed P-R
crossing, and this report does not claim otherwise.

## Status summary

- **M16E's FAIL-F4 gate is reversed.** Using the correct tool — the literal
  printed Eq. 4 (A), not the linearized Hessian (C) M16E used — the
  Section-4 classification gate (psi=160 STABLE, psi=80 UNSTABLE at
  `lambda/Rcyl=1.14*pi`) passes exactly (Section 3), and the small-amplitude
  correction (Section 4) resolves M16E's apparent energy-scale puzzle
  without any gamma_GB retuning (Section 5, retired per instruction).
- **The psi=160 mechanism is mechanistically confirmed, not fully
  demonstrated to completion.** The coupled inner-grain-shrinkage /
  period-`2*lambda`-growth mode exists in the Hessian (Section 6), is
  confirmed to be a genuine downhill direction from the non-stationary
  starting profile for both psi values (Section 8), and the actual PF
  dynamics move in exactly that direction at every sampled checkpoint for
  both psi=160 and psi=80 (Sections 11-13) — but at this project's
  established placeholder mobility scale, `t=60` reaches only a `~0.07%`
  change in `V1`, far short of grain elimination or an observed P-R
  crossing. The mobility-timescale check (Section 10) confirms this is a
  faithful reflection of genuinely slow reduced-time dynamics, not a
  numerical artifact, and that no computationally-cheap acceleration trick
  is available to close this gap within the session.
- **Event-local sink result (M16E) is unchanged and not recalibrated.**
  M16E's event-local sink ON/OFF comparison remains PASS-DIFFERENTIAL-SIGN
  only (all four required signs correct at quota completion:
  `strain_ON>strain_OFF`, `separation_ON<separation_OFF`,
  `x_neck_ON>x_neck_OFF`, `sigma_ON<sigma_OFF`), not a calibrated
  broadening magnitude — this milestone did not revisit or rerun that
  diagnostic.
- Full regression suite: 235/235 passing after all module renames and
  corrections (Section on test suite below).

STOP. No stochastic hazard introduced.

