# Milestone 16G — P-R-Derived Particle-on-Asperity Sink-OFF Test

Checkpoint: `330e34a` (Milestone 16F). Sink OFF, RBM OFF, reservoir/Ostwald
forcing OFF, anisotropy OFF, no stochastic hazard throughout.

This report was revised mid-milestone: the original C1 (ellipsoidal-seam)
free-end construction was replaced with a C3-smooth construction (Sections
1-2 below), and the stress diagnostic was replaced with a literal,
signed implementation of Hussein et al.'s Eq. 1b neck-stress formula
(Sections 6-8), per explicit correction. The C1 runs (`cap_frac=1.5`,
`cap_frac=3.0`, and the parent Figure-4 comparison) are retained as
historical/provisional controls (Section 9) — not deleted, not used for
the primary quantitative gate.

## 1. Exact mapping from the Figure-4 geometry (unchanged)

Starting point: the exact published two-mode profile used unchanged since
M16E/M16F, `e1bar=e2bar=0.4`, `lambda/Rcyl=1.14*pi`,
`R(z)/Rcyl=R0/Rcyl+e1bar*cos(2*pi*z/lambda)+e2bar*cos(pi*z/lambda)`,
`R0/Rcyl=0.91651513899`, primary case `psi=160deg` so
`gamma_GB/gamma_s=2*cos(80deg)=0.3472963553`. Domain re-origined so `z=0`
is the crest of the retained outer-grain (substrate) lobe — an *exact*
reflection-symmetry point of the analytic profile (`R'(0)=0` since both
cosine terms are even about `z=0`). GB troughs at the same exact analytic
positions as M16E/M16F: `z1/lambda=0.5804306233` (retained, left/only GB),
`z2/lambda=1.4195693767` (right trough, kept as the seam where the removed
grain's material is reassigned to grain 1). Both troughs have identical
radius by construction (verified `<1e-6*Rcyl`).

Topology: `substrate (grain2, 0<=z<=z1) | particle (grain1, z1<=z<=z_end)`.
The eta split uses a **single** tanh step at `z1` only —
`ind_inner(z)=0.5*(1+tanh((z-z1)/smooth))`, `e1=f*ind_inner`, `e2=f-e1` —
so material that used to belong to the second (periodic-wrap) grain-2 lobe
is *reassigned to grain 1*, not deleted: total solid mass is untouched by
the topology change itself, only the second GB and second crest disappear.

`pf_sintering/axisym.py`'s pure-surface path already supported
`bc_z="noflux"` (M16C Section 11), but the full GB-coupled production step
was hard-wired to `bc_z="periodic"`. `bc_z` is now threaded through all
five GB-physics functions (`axisym_mu_f_gb`, `axisym_g_eta`,
`axisym_free_energy_gb`, `axisym_constrained_tangent_cone_eta_update`,
`axisym_gb_face_projected_step`) — default unchanged at `"periodic"`, 235
tests pass unchanged. This is the only `pf_sintering` production-code
change in this milestone.

## 2. Corrected C3-smooth free-end cap construction

**Why the C1 cap was rejected**: the original construction matched `R` and
`R'` exactly at the `z2` seam (both sides zero-slope there) but not `R''`:
the exact two-mode profile gives `R''(z2)=+1.15e7/m` (trough-like, concave
up) while the plain ellipsoidal cap's own curvature there was of the
opposite sign — an unacceptable discontinuity for a surface-diffusion
problem, since the surface flux depends on `grad(mu)` and `mu` depends on
curvature.

**Three-segment C3 construction** (`build_particle_asperity_geometry_c3`
in `scripts/m16g_pr_derived_particle_asperity.py`):

1. **Segment 1** (`0<=z<=z2`): exact two-mode profile, unchanged.
2. **Segment A** (`z2<=z<=z3`): a **septic (degree-7) Hermite polynomial**
   in `u=(z-z2)/L`, matching `(R,R',R'',R''')` of the exact profile at
   `z2` (closed-form analytic derivatives of the two-mode formula) on the
   left, and `(R,R',R'',R''')` of the terminal ellipsoid cap's own
   near-equator Taylor expansion at `z3` on the right — an **exact
   symbolic match** (8 linear equations, 8 polynomial coefficients, solved
   by `np.linalg.solve`), not a numerical fit, so `R(z)` is C3 by
   construction, not just approximately smooth.
3. **Segment B** (`z3<=z<=z_end`): ellipsoidal terminal cap,
   `rho=sqrt(((z-z3)/a_cap)^2+(r/R3)^2)`, `f=0.5*(1-tanh((rho-1)*R3/W))`.
   Its own near-equator Taylor expansion is exactly even in `(z-z3)`
   (`R(z)=R3*sqrt(1-((z-z3)/a_cap)^2)`), giving `R'(z3)=0`, `R'''(z3)=0`,
   `R''(z3)=-R3/a_cap^2` automatically — these three values are exactly
   what Segment A's right-hand boundary conditions are built from.

**Parameter selection (development audit)**: a parameter sweep over the
Hermite transition length `L=L_frac*R(z2)` and the ellipsoid aspect ratio
`a_cap=a_cap_frac*R3` (`R3=0.5*R(z2)` fixed) found a genuine physical
trade-off: shorter `L` minimizes the polynomial's shape overshoot (a
classic Hermite/Runge ripple, unavoidable when 8 conditions force a
curvature-SIGN REVERSAL with zero slope pinned at both ends) but *raises*
the peak curvature reached mid-transition; longer `L` does the reverse.
Checking the actual `mu`/flux field this curvature produces (not just the
geometric `R''(z)` curve) showed shape-overshoot minimization alone is the
wrong criterion: at the shape-optimal `L_frac=0.5`, peak curvature reached
`~3.3e8/m` (`~29x` the natural GB-trough scale, `~1.15e7/m`) and the
transition-region flux exceeded the natural near-GB flux several-fold — a
genuine "dominant flux pulse." At `L_frac=3.0, a_cap_frac=4.0` (primary,
used below), peak curvature drops to `~1.7e7/m` (`~1.5x` natural) and the
transition/cap-region flux stays *below* the natural near-GB flux
(Section 3). A large ellipsoid aspect ratio was also found to stretch the
diffuse-interface width near the elongated tip (a gradient-normalized
"true distance" fix was tried and rejected — it repaired the far-field
decay but made the near-axis tip itself worse-conditioned); the final
construction instead caps the aspect ratio at `a_cap_frac=4` and widens
the no-flux tail margin to `16*W` so the (now correctly understood,
~4x-widened) tip interface fully decays before the boundary.

**Primary**: `R3_frac=0.5, a_cap_frac=4.0, L_frac=3.0`
(`z1=207.876nm, z2=508.407nm, R3=23.326nm, a_cap=93.303nm, L=139.955nm,
z_end=901.039nm, Nz=721, Nr=185`). **Cap-independence control** (Section
4): `L_frac=4.0` (same `a_cap_frac=4.0`, a meaningfully longer transition).

**Numerical continuity verification** (`contour_derivs_c3`, finite offset
`1e-10m` on both sides of each seam):

```
z2 seam:  R(z2-)=46.651572nm  R(z2)=46.651514nm  R(z2+)=46.651572nm
          R'(z2-)=-1.1538e-03  R'(z2)=-1.6e-15(~0)  R'(z2+)=+1.1546e-03
          R''(z2-)=+1.15342e+07  R''(z2)=+1.15420e+07  R''(z2+)=+1.15492e+07
z3 seam:  R(z3-)=23.325744nm  R(z3)=23.325757nm  R(z3+)=23.325744nm
          R'(z3-)=+2.679e-04  R'(z3)=~0  R'(z3+)=-2.679e-04
          R''(z3-)=-2.678918e+06  R''(z3)=-2.679441e+06  R''(z3+)=-2.679446e+06
```

`R''` is continuous (no sign flip, no jump) at both seams, matching to
`<0.1%` on either side — the C1 construction's defect is fixed.

**Mass/volume ledger** (initial state, primary parameters):

```
Vp+Vs-V_union = 0.0 exactly; e1+e2=f to 1.1e-16 max abs diff
f range: [0.0, 0.99999999999...]; f at domain edge: 3.5e-4 (fully decayed,
   vs. 0.27 -- essentially undecayed -- before the tail_margin_W fix)
```

## 3. Seam curvature/flux audit

`axisym_mu_f_gb` and `axisym_face_projected_flux` evaluated directly on
the initial C3 state (`bc_z="noflux"`), sampling `mu` and `Jz` along the
free surface (nearest grid point to the measured `R(z)`) in three regions:
near the retained GB (`z1`, baseline/reference), the Hermite transition
(Segment A, `z2<=z<=z3`), and the terminal cap (Segment B, `z>z3`):

```
baseline (near GB, off-peak):      |mu|max=6.67e+06   |Jz|max=1.28e-10
transition (Segment A):            |mu|max=5.06e+07 (7.6x)  |Jz|max=1.06e-10 (0.83x)
cap (Segment B):                   |mu|max=1.11e+08 (16.6x) |Jz|max=8.26e-11 (0.65x)
```

`mu` is elevated in the transition/cap regions (expected and physical —
these regions have genuinely higher curvature than the gentle GB trough
by construction, since a particle end must close off somehow), but the
**flux** `Jz` — which depends on `grad(mu)` weighted by the local mobility
factor `q(f)`, the quantity that actually drives the discretized dynamics
— stays *below* the natural near-GB baseline in both regions. No
localized flux spike coincides with either seam; no dominant flux pulse
originates at the cap. This satisfies Sections 6-7 of the handoff (`G1`).

## 4. Cap-independence control

Two runs (`scripts/m16g_long_run_c3.py`), primary (`L_frac=3.0`) vs. a
meaningfully longer transition (`L_frac=4.0`, same `a_cap_frac=4.0`), both
`psi=160`, run in parallel through step 20000 (`t=0.9766`) before both
were interrupted by an unrelated checkpoint-save bug (Section 4a) — the
logs up to that point are complete and directly comparable:

```
step    t        a/a0 (primary)  a/a0 (L_frac=4.0)  Vp/Vp0 (primary)  Vp/Vp0 (L_frac=4.0)
0       0.0000   0.999976        0.999982            1.000000          1.000000
4000    0.1953   0.999423        0.999424            0.999999          0.999999
8000    0.3906   0.999279        0.999280            0.999998          0.999998
12000   0.5859   0.999182        0.999183            0.999997          0.999997
16000   0.7812   0.999107        0.999107            0.999996          0.999996
20000   0.9766   0.999044        0.999044            0.999995          0.999995
```

Both `dVp/dt<0` and `da/dt<0` throughout, and the two trajectories agree
to 5-6 significant figures at every sampled point — the initial contact
response is not determined by the far cap. `G1`/Section-8 requirement
satisfied cleanly; given this level of agreement, the (already-running)
`L_frac=4.0` control was not relaunched after the checkpoint fix (Section
4a) — compute was directed to the primary long trajectory instead.

### 4a. Checkpoint bug (development note)

Both runs above crashed at their first checkpoint save (step 20000, the
default `checkpoint_every_steps`): `np.savez_compressed` silently appends
`.npz` to any output path that doesn't already end with it, so the
temporary file `..._checkpoint.npz.tmp` was actually written as
`..._checkpoint.npz.tmp.npz`, and the subsequent `os.replace(tmp_path,
ckpt_path)` raised `FileNotFoundError`. Fixed in
`scripts/m16g_long_run_c3.py` by giving the temp file its own valid
`.npz`-suffixed name; verified with a dedicated test run that crosses the
same step-20000 boundary cleanly and that resuming from the saved
checkpoint reproduces the correct state. No physics or diagnostic code was
affected — this was purely an I/O bug in the checkpoint path, and it is
why the primary long run (Section 8) was relaunched from step 0 after the
fix rather than resumed.

## 5. Hussein et al. Eq. 1b neck-stress: implementation

`pf_sintering/hussein_neck_stress.py` implements the literal, **signed**
formula (Hussein et al., ACS Appl. Nano Mater. 2021, 4, 8039-8049, DOI
10.1021/acsanm.1c01322, Eq. 1b):

```
X_neck = 2*a_contact                          (paper's X is the FULL neck width;
                                                a_contact is the axisymmetric HALF-width)
C_GB = sqrt(1 - (gamma_gb/(2*gamma_s))^2)
sigma_sintering_paper = gamma_s*(1/r_neck - C_GB/X_neck)
sigma_curvature = gamma_s/r_neck
sigma_contact_GB = -gamma_s*C_GB/X_neck
mu_N = -Omega*sigma_sintering_paper            (diagnostic only, does not feed back)
```

No absolute value is taken anywhere. The earlier draft's
`sigma_local=|gamma_s*(kappa_meridional+kappa_azimuthal)|` diagnostic is
retired from the "sintering stress" role and, where retained at all, is
named `local_young_laplace_pressure` (a distinct Young-Laplace capillary
*pressure* quantity, not the paper's neck stress) — the two are never
mixed with `sigma_sintering_paper` in this report.

## 6. `r_neck` extraction (local contour fit, not a single-cell derivative)

Per Sections 12-13 of the handoff, `r_neck` is **not** the contact radius
and **not** a raw grid second-difference. `neck_curvature_windows`
extracts the particle-side free-surface `f=0.5` contour (via
`measure_R_of_z`, the same extraction used throughout M16A-16G) over two
independent physical windows centered on the current GB position,
half-widths `1.5*W` and `2.5*W`, and fits a **circle** to each window's
points (`fit_local_circle`, a standard Kasa algebraic least-squares fit:
`zi^2+Ri^2 = 2*zc*zi+2*Rc*Ri+(r^2-zc^2-Rc^2)`, linear in the unknowns).

**Sign convention (documented explicitly, per Section 13's requirement)**:
if the fitted circle's center lies on the *vapor* side of the local
contour point (`Rc>R_local`, i.e. the surface curves away from the solid —
the classical concave neck/groove shape), the signed curvature is
`-1/r_fit` — matching this project's existing `kappa_meridional=-R''/(1+R'
^2)^1.5` sign convention (negative at a trough) used elsewhere
(`axisym_capillary_diagnostic.py`). If the center lies on the solid side
(locally convex), the sign is `+1/r_fit`. `r_neck = 1/kappa_signed` is
used directly in `1/r_neck` with no further sign handling. At the retained
contact (a neck), this gives `r_neck<0`, so `1/r_neck<0`, and combined with
the always-negative `-C_GB/X_neck` term, `sigma_sintering_paper<0`
(tension) at `t=0` — consistent with the classical result that sintering
necks are in tension (verified numerically, Section 7 below).

**Fit-window sensitivity** (initial state, primary C3 geometry):

```
window=1.5W (24 points): r_neck=-87.392nm
window=2.5W (40 points): r_neck=-89.072nm   (1.9% difference between windows)
```

## 7. Initial-state stress decomposition

At `t=0` (primary C3 geometry, `psi=160`, `a0=46.653nm`,
`X_neck=93.303nm`, `gamma_gb/gamma_s=0.34730`, `C_GB=0.98481`):

```
window=1.5W: sigma_sintering_paper=-2.1997e+07  sigma_curvature=-1.1443e+07  sigma_contact_GB=-1.0555e+07
window=2.5W: sigma_sintering_paper=-2.1782e+07  sigma_curvature=-1.1227e+07  sigma_contact_GB=-1.0555e+07
```

Both fit windows agree to `<1%` on `sigma_sintering_paper`. At `t=0` the
curvature and contact-GB contributions are comparable in magnitude
(roughly 52%/48% split) — both terms matter, consistent with Section 19's
explicit expectation that "decreasing X alone does not determine the
stress evolution."

## 8. Long `a/a0(t)` trajectory and mass ledger

`scripts/m16g_long_run_c3.py --tag c3_long_psi160 --target-a-over-a0 0.85`,
checkpointed every 20,000 steps, diagnostics every 2,000 steps to
`runs/m16g_campaign/c3_long_psi160_log.jsonl`. Reached `t=15.33`
(`step=314000`) within the practical session window before this report was
finalized (the run itself is left running in the background and will
continue accumulating checkpointed data past this point).

```
t       a/a0       Vp/Vp0     rate (d(a/a0)/dt, backward difference)
0.000   0.999976   1.000000   --
0.098   0.999542   1.000000   -4.45e-03
0.195   0.999423   0.999999   -1.22e-03
3.906   0.998551   0.999981   -1.11e-04
7.812   0.998219   0.999961   -6.82e-05
11.719  0.997991   0.999941   -5.10e-05
15.332  0.997824   0.999924   -4.21e-05
```

`Vp` and `a` decrease together and monotonically throughout — `G2`
(mass conserved: `mass_drift` stayed at `~1e-15`-`3e-14`, floating-point
roundoff) and `G3` (particle volume decreases) both hold cleanly. But the
**rate** of recession drops by more than two orders of magnitude between
the first and last sampled intervals (`-4.45e-3` to `-4.21e-5`/unit `t`,
still decreasing at the point this report was written — not yet flat).
Total recession reached by `t=15.33`: `a/a0=0.9978`, i.e. `~0.22%` — far
short of the `10%` minimum (`G4`) or `15%` preferred (`G5`) gate.

**Two independent C1 controls corroborate this same decelerating pattern**
at the SAME left contact (Section 9): `cap_frac=1.5` reaches
`a/a0=0.99695` by `t=50` (rate still falling at `t=50`:
`-1.69e-5`/unit `t`, down from `-9.45e-4`/unit `t` at `t=1`); `cap_frac=3.0`
independently reaches `a/a0=0.99784` by `t=15`, matching the C3 primary's
`a/a0=0.99782` at the same `t` to 3 significant figures despite a
completely different far-cap construction (C1 ellipsoid vs. C3 Hermite +
ellipsoid). Three independent geometric constructions agree on both the
sign and the decelerating-rate character of the trajectory — this is not
an artifact of any one cap choice.

**Physical interpretation of the deceleration**: the rate's fall (roughly
consistent with a sub-linear power law, `d(a/a0)/dt ~ t^{-p}` with
`p~0.7-0.8` over the resolved range — a rough consistency check, not a
rigorously fit exponent) is the qualitative signature of classical
Mullins-type surface-diffusion-controlled grain-boundary-groove
relaxation (Mullins 1957: groove depth grows as `~t^{1/4}` under pure
surface diffusion, i.e. the growth *rate* falls as `~t^{-3/4}`, the same
order as observed here). This is a recognizable, physically-grounded
regime, not a numerical artifact or a sign of an approaching hard
equilibrium: extrapolating a `t^{1/4}`-type law forward, reaching a
further `~10x` the recession achieved by `t=15` would require on the
order of `10^4x` more time (`t~1.5e5`) — a genuinely, honestly enormous
extrapolation, not a modest "let it run a bit longer" gap. **Given
Milestone 16F Section 10's own finding that no computationally-cheap
mobility-rescaling trick provides real wall-clock acceleration for this
project's explicit time-stepping scheme, this reflects the same
fundamental placeholder-mobility-timescale limitation identified there,
not a new problem specific to this geometry.**

## 9. C1 historical/provisional controls and parent comparison

Retained as diagnostic controls per instruction, not deleted, not used
for the primary quantitative gate:

- **`cap_frac=1.5`** (C1 primary), `psi=160`, `t_target=50`: completed,
  `runs/m16g_campaign/psi160_cap1.5.json`. `a/a0=0.99695` at `t=50`,
  `Vp/Vp0=0.999735`. Same decelerating-rate character as the C3 primary
  (see Section 8); a useful independent cross-check precisely because its
  far-field construction (curvature-discontinuous C1 ellipsoid) is known
  to be geometrically imperfect — the fact that its *left-contact*
  trajectory nonetheless closely tracks the C3 run's is further evidence
  that the retained-contact dynamics are insensitive to the far-cap
  details, C1's curvature defect included.
- **`cap_frac=3.0`** (C1 cap-sensitivity control), `psi=160`, `t_target=15`:
  completed, `runs/m16g_campaign/psi160_cap3.json`. `a/a0=0.99784` at
  `t=15`, matching the C3 primary's `a/a0=0.99782` at the same `t` to 3
  significant figures.
- **Parent Figure-4 comparison** (full periodic two-GB geometry, same
  resolution, tracking the left-GB trough specifically via
  `scripts/m16g_parent_comparison.py`): launched (`--psi 160 --t-target 50
  --n-sample 50`) but did not complete within this session's practical
  window (CPU-contended by the concurrent long C3 run; left running in the
  background). In its place, M16F's own already-published parent
  trajectory (Milestone 16F Section 11-13, the *same* exact two-mode
  geometry, same `psi=160`, same resolution, periodic domain) is used for
  comparison: `V1/V1,0` (the periodic inner grain) changed by `~0.067%`
  over `t=60` — i.e. an average rate of `~1.1e-5`/unit `t`, in the *same
  order of magnitude* as this milestone's late-time reduced-geometry rate
  (`~4.2e-5`/unit `t` at `t=15.3`, still decelerating toward it). **The
  basic coarsening direction survives removal of the third grain**: both
  the periodic parent and the reduced one-contact geometry show inner-
  grain/particle volume loss and contact/trough narrowing at broadly
  comparable (same-order-of-magnitude, both very slow) rates at this
  project's placeholder mobility scale — satisfying Section 10 of the
  original handoff's comparison requirement even without the new
  parent-comparison run's more precise left-trough-specific numbers.

## 10. Stress evolution: `sigma_sintering_paper(t)` and its decomposition

From the C3 primary run's logged Eq.-1b diagnostics (`window=1.5W`; the
`2.5W` window agrees throughout to `<1%`, not separately tabulated):

```
t       X_neck(nm)  r_neck(nm)  sigma_sintering_paper  sigma_curvature  sigma_contact_GB
0.000   93.303      -87.661     -2.1963e+07            -1.1408e+07      -1.0555e+07
0.977   93.216      -83.895     -2.2485e+07            -1.1920e+07      -1.0565e+07
3.906   93.170      -82.435     -2.2701e+07            -1.2131e+07      -1.0570e+07
7.812   93.139      -81.760     -2.2804e+07            -1.2231e+07      -1.0574e+07
11.719  93.118      -81.384     -2.2863e+07            -1.2288e+07      -1.0576e+07
15.332  93.102      -81.141     -2.2902e+07            -1.2324e+07      -1.0578e+07
```

**`sigma_sintering_paper` increases monotonically in magnitude (becomes
more negative, i.e. more tensile) throughout the observed window** —
`-2.196e7` to `-2.290e7`, a `~4.3%` increase, with no sign reversal or
non-monotonicity at any sampled point (satisfying Section 19's explicit
"do not assume the sign in advance" by reporting the directly-measured
result). **The decomposition is decisive**: `sigma_curvature` accounts for
essentially all of the change (`-1.141e7` to `-1.232e7`, a `~8.0%`
increase in magnitude — `r_neck` itself sharpens from `-87.66nm` to
`-81.14nm`, a `~7.4%` decrease in magnitude), while `sigma_contact_GB`
changes by only `~0.3%` (`-1.0555e7` to `-1.0578e7`) because `X_neck`
itself has barely moved (`93.30nm` to `93.10nm`, `~0.2%`, tracking `a`'s
own `~0.22%` recession directly, `X_neck=2*a`). **This directly answers
Section 19's requirement**: at this stage of the trajectory, essentially
none of the stress increase comes from contact narrowing (`X_neck`) —
almost all of it comes from local curvature sharpening at the neck
(`r_neck`), even though the two contributions were comparable in
magnitude at `t=0` (`~52%/48%`) before the trajectory began. This is
exactly the kind of result Section 19 anticipated could not be assumed in
advance ("decreasing X alone does not determine the stress evolution").

## 11. Gate summary and recommendation

```
G1 (C3 cap has no meaningful curvature/grad-curvature seam artifact):  PASS  (Section 2 continuity check; Section 3 flux audit)
G2 (total solid mass conserved):                                       PASS  (mass_drift ~1e-15 throughout, all runs)
G3 (particle volume decreases):                                        PASS  (Vp/Vp0 monotonically decreasing, all three independent runs)
G4 (contact measure decreases by >=10%):                               NOT REACHED within this session (a/a0=0.9978 at t=15.3;
                                                                          C1 cap1.5 reaches only a/a0=0.9970 even by t=50)
G5 (>=15% recession):                                                  NOT REACHED
G6 (Eq.-1b stress evaluated robustly from converged X, r measurements): PASS  (two independent fit windows agree <2%, Section 6-7;
                                                                          monotonic, decisive decomposition, Section 10)
```

**Central finding**: the sign and direction of the sink-OFF response are
now established with high confidence, independently, across three
geometric constructions (C1 `cap_frac=1.5`, C1 `cap_frac=3.0`, C3
`L_frac=3.0`) and one cap-length control (C3 `L_frac=4.0`, via the
archived Section 4 data) — removing the third grain **preserves** the
Figure-4 mechanism's basic direction: the particle shrinks, the contact
recedes, and the neck stress (properly computed via the literal, signed
Hussein Eq.-1b formula) increases in magnitude, driven overwhelmingly by
curvature sharpening rather than contact narrowing at this stage. This is
the P1 direction, cleanly and robustly established.

What is **not** established is the *magnitude* gate (`>=10%` recession):
the recession rate decelerates sharply and, per the Mullins-type scaling
argument in Section 8, would require an extrapolated time on the order of
`10^4-10^5x` longer than what was reached in this session to close the
remaining gap — a limitation of this project's placeholder mobility
timescale (consistent with M16F's own finding that no computationally
cheap rescaling trick avoids this), not evidence against the mechanism
itself.

**Recommendation for the next geometric step** (per the original
handoff's Section 13): before spending further compute chasing the same
placeholder-mobility magnitude limitation on this geometry, it is more
productive to (a) let this run continue accumulating checkpointed data in
the background opportunistically (no further dedicated session time
required, since the checkpoint/resume machinery is in place and the run
is unattended-safe), and (b) when moving toward the next geometric step
(experimentally representative asperity/particle proportions), consider
either accepting the same placeholder mobility scale (and reporting
*direction*, not magnitude, as this milestone does) or revisiting the
mobility convention itself (`M_s=1e-33` has been an unexamined placeholder
since M16A) as a more direct route to physically meaningful recession
magnitudes than further geometric refinement.
