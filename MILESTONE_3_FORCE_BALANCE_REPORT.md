# Direct interfacial capillary-force / zero-force-neck report

Scope per the updated handoff: Sections 4-10 (direct TJ vector construction, synthetic
validation, signed-curvature audit, static neck-force map, restoring-direction test,
x0(V2,L)). **Stopping here for review, as instructed** (Section 14/15) — no stochastic
nucleation, forced-sink events, long coarsening trajectories, or rate sweeps were
attempted.

## A. Branch and commits

- Branch: `codex/coarsening-stress-buildup`
- HEAD at session start / end: `b4a19de10c5b7c165232013c9ac9cce8e4022fdd` (pulled
  `PHYSICS_BACKGROUND.md` mid-session; no conflicts with the pre-existing uncommitted
  Milestone-1/2 work, which was preserved intact).
- Nothing has been committed yet this session (see Section N below for the working
  tree and a suggested commit split matching handoff Section 13).

## B. Tests and pass/fail status

- Before this session's changes: 14/14 passing (the Milestone-1/2 baseline).
- After all changes in this session: **26/26 passing** (`pytest ./PF-Sintering/tests -q`).
  12 new tests added: `tests/test_tj_force.py` (10, synthetic TJ + signed-curvature
  validation) and `tests/test_static_neck_geometry.py` (2, static-family construction).
- No existing test was modified or weakened.

## C/D. Exact capillary-vector definitions and sign convention

Implemented in `pf_sintering/tj_force.py`, kept entirely separate from
`model.compute_stress`/`measure_dihedral`/`curvature` — nothing in the legacy pipeline
was changed, and `sigma` (with its Milestone-1 decomposition) remains available as a
parallel diagnostic exactly as instructed (Section 5).

- **Per-branch capillary vector**: for a free-surface branch with outward unit tangent
  `v` (pointing from the TJ into the bulk of that branch),
  `xi = gamma(theta)*v + (dgamma/dtheta)*n`, where `n` is the outward vapor-side normal
  (found via the existing `model._vapor_normal`) and `gamma`/`dgamma/dtheta` come from
  the existing anisotropic LUT (`model._aniso_gamma`) when anisotropy is enabled, or
  reduce to `(gamma_s, 0)` when it is not. This is the *same* Cahn-Hoffman
  construction `compute_stress`'s anisotropic branch already used per-branch — the only
  change is that the **vector** `xi` is kept (and signed) instead of collapsing
  immediately to `abs(F[0])`.
- **GB capillary vector**: `xi_gb = effective_gamma(s, p) * v_gb` (isotropic, per
  Section 4's allowance; `v_gb` is the GB's outward tangent at the TJ, pointing into
  the bulk of the GB).
- **Resultant**: `F_TJ = xi_s1 + xi_s2 + xi_gb`, computed **separately at the upper and
  lower TJ** whenever both resolve (`TJForceReport.top` / `.bottom` in `tj_force.py`).
  Equilibrium is `F_TJ -> 0`; `psi_eq` is **never** inserted anywhere in this
  computation — it is not used, not even as a fallback.
- **Sign convention**: every branch vector points *away from the TJ, into the bulk of
  its own interface* — the standard Young/Herring convention where each interface's
  tension "pulls" the junction toward itself. Verified analytically and numerically on
  the synthetic wedge (Section E below): for a symmetric groove this convention gives
  `F_TJ = (0, gamma_s*2*cos(psi/2) - gamma_gb)` in the isotropic case, which is
  `F_TJ -> 0` exactly at `gamma_gb = 2*gamma_s*cos(psi/2)` (Young's equation) — i.e.
  the sign convention reproduces the correct equilibrium condition from first
  principles, not by construction.
- **Projections reported** (`TJForce.F_TJ_x`, `.F_gb_normal`, `.F_gb_parallel`):
  - `F_TJ_x`: component along the fixed x-axis (the particle/substrate densification
    direction — the axis RBM actually translates along). `TJForceReport.F_drive` sums
    `F_TJ_x` over both resolved TJs; this is the quantity used as `F_drive(x_neck)` in
    Sections 8-10.
  - `F_gb_normal` / `F_gb_parallel`: components normal/parallel to *that TJ's own*
    local GB tangent (`n_gb` is `v_gb` rotated 90 degrees, sign chosen so
    `dot(n_gb, +x) >= 0`), reported per-TJ rather than in a single global frame since
    the two TJs' own local GB tangents point in opposite senses (top TJ's tangent
    points "down" into the bulk, bottom TJ's points "up").
- `psi` (angle between `v_s1` and `v_s2`) is reported purely as a diagnostic, computed
  independently of `measure_dihedral`, and is never fed back into the force
  calculation (see F below).

## D. How branch tangents/normals are extracted

`tj_force._branch_directions` / `_circle_crossings`: sample the relevant scalar field
(`f` for the two free-surface branches; `e1 - e2`, masked to the solid interior via
`f > 0.5`, for the GB branch) on a small circle centered at the TJ, find the field's
zero/level crossings by sign change around that circle, then refine each crossing to a
true local tangent via a centered-finite-difference gradient at the crossing point
(`tangent = rotate90(grad(field))`, oriented away from the TJ). No `find_contours`
call, no windowed-clustering, no angular-gap heuristic — this was a deliberate
departure from `measure_dihedral`'s approach (see F). The circle radius for the free
surfaces / GB is capped at `min(2.5, 0.4*neck_height/W)` / `min(0.7*r_surf,
0.35*neck_height)` interface-widths respectively, so it self-scales down for narrow
necks and never reaches the *other* TJ.

**Robustness bug found and fixed along the way**: the TJ column itself is located by
searching a small window around `overlap_col`'s e1*e2-weighted centroid for the
column with the *narrowest* finite (non-domain-spanning) solid extent, rather than
trusting the centroid column directly (`tj_force.locate_neck_tjs`). On the actual
Milestone-1/2 candidate geometry, the centroid column sat 1-2 grid cells away from the
column where the neck first becomes visible (finite y-extent) — a small offset, but
enough that `f[:, nc]` spanned the *entire* domain (touching both y-edges) at the
centroid column, which is exactly `measure_dihedral`'s own silent-failure condition.
This is very likely a **major contributor** to the Milestone-1/2 finding that
`measure_dihedral()` always fell back to `psi_eq` (`MILESTONE_1_2_REPORT.md`, J.1) —
not (only) a fundamental resolution problem, but partly a fragile column estimate.

## E. Synthetic TJ validation (`tests/test_tj_force.py`)

A synthetic diffuse-interface groove/wedge (straight GB descending from the TJ, two
straight free-surface flanks at a prescribed dihedral angle psi, all masked in the
exact same style as `model.initialize_fields`) was built for psi in {80, 100, 120,
140, 160} degrees. All 10 tests pass:

- **A. Branch directions recovered correctly**: measured `v_s1`, `v_s2`, `v_gb` all
  matched the analytically prescribed tangents to within ~2.5 degrees, for every angle.
- **B. Independent psi measurement**: `acos(dot(v_s1, v_s2))` matched the prescribed
  synthetic angle to within ~2.5 degrees, for every angle — this is the *new* psi
  measurement, and it succeeds where `measure_dihedral` did not on the real geometry.
- **C. Equilibrium -> near-zero resultant**: with `gamma_gb` set to the Young's-equation
  value for psi=120, `|F_TJ| / gamma_s < 0.06` (discretization-level residual, not
  exactly zero as expected for a diffuse, finite-resolution interface).
- **D. Perturbation sign, monotonic through equilibrium**: with `gamma_gb` fixed at the
  120-degree equilibrium value, `F_gb_parallel` (component of `F_TJ` along the GB) is
  strictly monotonically increasing across {80, 100, 120, 140, 160} degrees, negative
  below 120, ~0 at 120, positive above — matching the closed-form prediction
  `F_gb_parallel = gamma_gb - 2*gamma_s*cos(psi/2)` exactly in sign structure.
- **E. Anisotropic Cahn-Hoffman actually exercised**: with anisotropy enabled
  (`aniso_delta=0.15`) and an asymmetric psi=100 orientation, `xi` deviates from the
  isotropic prediction `gamma_s*v` by >2% of `gamma_s` for at least one branch, while
  the isotropic-mode run deviates by <1e-9 (i.e. reduces to the isotropic case exactly
  when anisotropy is off).

One real numerical bug was found and fixed during this validation, not after: the
symmetric synthetic wedge is exactly mirror-symmetric about the GB, so `e1 - e2 = 0`
identically along the entire GB line, and the very first crossing-detection
implementation double-counted the exact-zero sample that this symmetry produces at
`theta = 270` degrees (`n_theta=1440` divides 360 evenly, landing a sample exactly
there). Fixed in `_circle_crossings` by not re-triggering the sign-change branch when
the *next* sample is itself an exact zero (that zero is picked up when the loop
reaches it).

## F. `measure_dihedral()` status

**Not modified.** It remains exactly as it was, still used by the legacy `sigma`
pipeline, and still a required input there (`compute_stress` still falls back to
`psi_eq` when it fails, unchanged). The new `tj_force.py` psi measurement is entirely
independent — it never calls `measure_dihedral`, never uses `psi_eq`, and is purely
diagnostic (its value does not feed into `F_TJ` or anything else). On the real
Milestone-1/2 t=0 candidate geometry, once the column-location bug (Section D) was
fixed, the new method **did** resolve both TJs, with a striking result: measured
`psi ~= 51.9` degrees, versus `psi_eq ~= 120.0` degrees implied by `theta_mis=30`
degrees. That is a very large deviation from equilibrium — consistent with the
raw analytic initial condition (half-space intersected with an ellipse) being a sharp,
unrelaxed geometric intersection rather than any kind of relaxed meniscus, and it
independently explains (beyond the column-location bug) why `measure_dihedral`'s own
angular-gap clustering had trouble on that state.

## G. Signed-curvature audit (`pf_sintering/signed_curvature.py`)

Confirmed: `model.curvature()` (used for the legacy `sigma_curv = gamma_s * kappa`) is
**unsigned** — it fits a circle via Kasa least squares and returns `1/R` with no
information about which side of the interface the center lies on. A new, separate
`signed_curvature_at()` was added (not wired into `compute_stress` or `tj_force.py`'s
`F_TJ`, per Section 7's explicit instruction to keep the two ideas distinct): same
Kasa fit, but the fitted circle's center is sampled against `f` — positive when the
center lies inside the solid (convex, particle-cap-like), negative when in the vapor
(concave, neck/groove-fillet-like). Validated on synthetic convex/concave circular
arcs of known radius `R`: recovered `+1/R` and `-1/R` respectively, matching the known
radius to within 15% (test `test_signed_curvature_convex_positive_concave_negative`).

## H/I. Static neck-force map and x0

**Reference geometry** (`scripts/static_neck_force_benchmark.py`): `dev` preset,
`dx=5nm`, `R2=80nm`, `aspect_ratio=2.0`, `contact_orientation=short_plane`,
`V2 = pi*R2^2 = 2.011e-14` (a round reference volume — a full circle of the nominal
particle radius, not tied to any specific `initial_overlap` state), `L = 100nm`
(chosen after the original Milestone-1/2 candidate's own `L` (~117nm) was found, in an
initial scan, to admit *no* zero-force neck width at all in the physically accessible
range — see Section M).

**Static family construction** (`pf_sintering/static_neck_geometry.py`, new): two
earlier constructions were tried and rejected before arriving at the one used —
documented in the module's own docstring:
1. A perturbed-ellipse ("Gaussian bump" added to the level set near the wall) family:
   rejected because `x_neck(bump amplitude)` was **non-monotonic**, making it
   unbisectable.
2. A directly-parametrized boundary profile blending a flat neck collar into a
   circular far cap over a *fixed transition length*: rejected because whenever `L`
   was large enough that the far cap doesn't reach the wall, blending toward
   `y_far(wall) = 0` pulled the profile **below** the target neck width before the cap
   "caught up" — a spurious dip.
3. **Used**: `y(x) = max(x_neck, y_far(x))` for `x <= cx` (wall side) and `y(x) =
   y_far(x)` for `x > cx` (far side) — a flat collar of exactly `x_neck` half-width
   wherever the cap is narrower, continuous at `x = cx` (both branches equal `Rfar`
   there), closing off to zero on the far side as a plain circular cap. `Rfar`
   (far-cap radius) and `cx` (far-cap center, `L ~= cx - wall`) are found by a nested,
   feasibility-scanning bisection (not a fixed bracket — a fixed bracket was tried
   first and found to fail because not every candidate `cx` admits *any* feasible
   `Rfar`) so V2 and L match their targets. Achieved V2/L match targets to ~1e-8 to
   1e-12 relative error across the tested range (`x_neck` in roughly 15-45 nm for
   this reference state); `x_neck` itself is not hit exactly (`contact_width`'s
   area/peak-ratio measure differs slightly from the raw profile value) but tracks the
   target monotonically to within ~5-10%.

**Force map** (V2 = V20, L = 100nm; `x_neck_target` scanned 15-34nm, `x_neck_achieved`
in parentheses):

| x_neck target (nm) | achieved (nm) | F_drive | psi_top (deg) |
|---:|---:|---:|---:|
| 15 | 22.44 | +1.132 | 31.7 |
| 18 | 24.00 | +1.138 | 22.3 |
| 20 | 25.13 | +1.199 | 21.4 |
| 22 | 26.35 | +1.024 | 29.0 |
| 24 | 27.73 | +0.995 | 31.9 |
| **26** | **29.17** | **-0.087** | 28.5 |
| 28 | 30.66 | -0.011 | 26.9 |
| 30 | 32.24 | -0.102 | 32.2 |
| 32 | 33.89 | -0.123 | 33.2 |
| 34 | 35.58 | -0.050 | 31.2 |

**x0 = 24.98 nm** (target; achieved `x_neck ~= 28-29nm`), located by bisection between
the `x_neck_target = 24` and `26` nm samples, where `F_drive` changes sign cleanly
(`+0.995 -> -0.087`) and `psi` varies smoothly (no jump). `F_drive` stays small and
negative from 26-34nm rather than growing — see Section M for a second, abrupt (and
almost certainly artifactual) sign flip found further out (~38-40nm) that was
deliberately excluded from the x0 search.

## J. Restoring-direction test around x0

Five states at `frac * x0` for `frac in {0.90, 0.95, 1.00, 1.05, 1.10}` were built
(same V2, L), then evolved for 100 steps of **ordinary capillary PF relaxation only**
(CH + mass-preserving eta projection, structural relaxation + mass-preserving eta
projection — no Ostwald, no hazard, no RBM, matching Section 9 exactly). Mass/L held
essentially fixed during the relaxation (`V2` drift ~1e-15 relative; `L` drift
~-0.2nm out of 100nm, i.e. ~0.2%, consistent with the same small centroid-drift effect
documented for the Milestone-1/2 sink-off benchmark, not RBM).

| frac of x0 | x_neck start (nm, target) | dx_neck/dt (nm/s) | dG_interface/dt (J/s) |
|---:|---:|---:|---:|
| 0.90 | 22.49 | -150.4 | -4.25e-6 |
| 0.95 | 23.74 | -171.8 | -4.13e-6 |
| 1.00 | 24.98 | -191.2 | -4.00e-6 |
| 1.05 | 26.23 | -205.1 | -3.68e-6 |
| 1.10 | 27.48 | -220.1 | -3.57e-6 |

**Result: the acceptance criterion in Section 9 fails.** `dx_neck/dt` is negative at
every tested point, both below and above x0 — there is no sign change, so x0 (the
direct TJ force-balance zero) does **not** behave as an attractor for `x_neck` under
this short-time capillary-relaxation test. `dG_interface/dt` is negative throughout
too (expected — spontaneous relaxation should lower total interfacial energy
regardless), but that alone doesn't imply `x_neck` should be attracted to x0.

**Leading hypothesis, not fully confirmed**: the static family's `y(x) = max(x_neck,
y_far(x))` construction has a genuine curvature *discontinuity* (a kink) at `x = cx`,
where the flat wall-side collar meets the circular far cap — necessary to
independently dial `x_neck` at fixed V2/L, but not present in any physically relaxed
meniscus. A direct check with the new signed-curvature diagnostic at x0 found
comparable-magnitude curvature at both the real TJ (`R ~= 39nm`) and at this
construction kink (`R ~= 74nm`, versus the plain cap's own apex curvature of `R ~=
71nm` for comparison — i.e. the "kink" barely stands out from the cap's own curvature
at the resolution of a several-interface-width circle fit, so this check is
suggestive rather than conclusive). The kink's own curvature-driven relaxation is a
plausible confound that could dominate the short-time `dx_neck/dt` signal and mask
whatever TJ-force-driven restoring tendency might exist underneath it. This was **not**
investigated further in this session, per the explicit Section 9 instruction: *"If
they do not [produce opposite restoring responses], stop and diagnose the PF
capillary thermodynamics before reintroducing coarsening."*

## K/L. x0(V2, L) at fixed L

| V2/V20 | x0 (nm) |
|---:|---:|
| 1.000 | 24.984 |
| 0.999 | 24.984 |
| 0.998 | 24.984 |
| 0.995 | 24.984 |

**x0 did not move at all** (identical to the bisection tolerance, 0.05nm) across the
full requested `V2/V20` range. This is very likely a **construction limitation, not a
physical result**: the branch-detection circles used to compute `F_TJ` have radius
capped at a few interface widths (Section D), so they only ever sample the
*wall-side flat collar* region of the static family — which, by construction, is
identical (flat, half-width `x_neck`, straight walls parallel to the wall) for *any*
`Rfar`/`V2`, as long as the collar is long enough to keep the far-cap kink outside the
probe radius (true for all four V2 values tested here — the collar length only
changes by <1nm across a 0.5% V2 change). In other words, the local TJ force
calculation, as currently probed, is **blind to V2** in this specific geometry family
whenever the collar is comfortably long. This means **Section 10's actual physical
question — does the zero-force neck width shift toward smaller contact as the particle
shrinks at fixed separation — was not actually answerable with this construction**,
and the "no shift" result should not be read as evidence against the coarsening
mechanism. Answering it properly would need a static family where the *near-neck*
curvature itself (not just the far cap) responds to V2 — e.g. a smooth, kink-free
neck-fillet-to-cap profile — which was not attempted here, consistent with stopping
for review rather than continuing to iterate on the geometry generator.

## M. Numerical pathology / ambiguity notes

1. **Second, abrupt F_drive sign flip near x_neck ~= 38-40nm** (V2=V20, L=100nm):
   `F_drive` jumps from -0.11 (x_neck_target=38nm) to +0.98 (x_neck_target=40nm), with
   `psi_top` jumping from 36 to 63 degrees over the same step — a discontinuity, not a
   smooth crossing. This coincides with `cx` (the far-cap center) approaching the
   coordinate origin as the collar shortens for a wider neck, which is suspicious but
   was not root-caused in this session. It was excluded from the x0 search deliberately
   (the 24-26nm crossing is smooth, resolved, and analytically sensible; this one is
   not) and should be treated as **unverified / likely artifactual** rather than a
   second physical equilibrium.
2. **The originally-planned reference L (~117nm, inherited from the Milestone-1/2
   `overlap=8nm` candidate) admits no zero-force neck width** in the physically
   accessible range at all (`F_drive` stayed positive and increasing from `x_neck
   ~= 27` to `51nm`, the full range that construction supports for that `V2`). This is
   consistent with that geometry being far from any equilibrium contact configuration
   (also see Finding F: `psi ~= 52` degrees versus `psi_eq ~= 120` degrees measured on
   that same state) — a further, independent line of evidence that the raw analytic
   initial condition used throughout Milestone 1/2 was not close to a physically
   relaxed neck, which is itself worth keeping in mind when interpreting *that*
   report's conclusions.
3. **Very wide necks are geometrically infeasible** for a given (V2, L) in this
   construction (roughly `x_neck_target > 1.15-1.2 * x0`-ish, geometry-dependent):
   the flat collar alone can already exceed `v2_target` before any `Rfar` is even
   searched. `build_neck_state` falls back to the closest feasible grid sample in that
   regime rather than raising, which is now covered by an explicit test
   (`test_infeasible_family_member_raises_or_falls_back_without_crashing`) so this
   behavior is at least caught, not silent.
4. Two coordinate conventions coexist in this codebase and were carefully *not*
   conflated, but are worth flagging for anyone extending this work: `model.py`'s
   field-construction code (`initialize_fields`, and the new
   `static_neck_geometry.py`, modeled after it) uses a domain-*centered* coordinate
   system (`X = (index - Nx/2)*dx`), while its contour/TJ measurement code
   (`measure_dihedral`, `curvature`, `_vapor_normal`, and the new `tj_force.py`,
   deliberately matching that existing convention) uses an *uncentered* one (`X =
   index*dx`). Each is internally self-consistent (build vs. sample always use the
   matching convention within their own module), so this is not a bug, but a
   `cx`/`Rfar` value from `static_neck_geometry` and a `tj_xy` from `tj_force` are in
   different frames and must not be compared or combined directly without an explicit
   `Nx/2*dx` (or `Ny/2*dx`) shift.

## N. Working tree / suggested commits

Nothing has been committed. Current uncommitted files (Milestone-1/2 work preserved
in full, per instructions):

- `pf_sintering/model.py`, `pf_sintering/parity_kernels.py` (Milestone 1/2: sigma
  decomposition, H1/H2 controls)
- `pf_sintering/diagnostics.py`, `scripts/scan_near_critical_geometry.py`,
  `scripts/run_sinkoff_stress_matrix.py`, `tests/test_diagnostics.py`,
  `tests/test_mechanism_controls.py`, `MILESTONE_1_2_REPORT.md` (Milestone 1/2)
- `pf_sintering/tj_force.py`, `tests/test_tj_force.py` (this session: TJ vector
  construction + synthetic validation)
- `pf_sintering/signed_curvature.py` (this session: signed-curvature audit)
- `pf_sintering/static_neck_geometry.py`, `tests/test_static_neck_geometry.py` (this
  session: static neck-width family)
- `scripts/static_neck_force_benchmark.py` (this session: Sections 8-10 driver)
- `MILESTONE_3_FORCE_BALANCE_REPORT.md` (this report)

Suggested split, matching handoff Section 13 (not yet executed — awaiting review):
1. Milestone 1/2 diagnostics + controls + report (already reviewed).
2. TJ branch/vector diagnostic + synthetic geometry tests (`tj_force.py`,
   `test_tj_force.py`, including the `locate_neck_tjs` column-search fix).
3. Signed-curvature audit/tests (`signed_curvature.py`).
4. Static neck-geometry family + its tests (`static_neck_geometry.py`,
   `test_static_neck_geometry.py`).
5. Force-map / restoring-direction / x0(V2,L) benchmark driver
   (`static_neck_force_benchmark.py`) + this report.

---

**Stopping here per Section 14/15.** The headline result is mixed and should be read
carefully: the direct capillary-vector construction itself is validated (Section E,
10/10 synthetic tests) and *did* resolve on real geometry where the legacy
`measure_dihedral` could not — but the static benchmark built to test whether its
zero (`x0`) is a genuine dynamical equilibrium **failed that test** (Section J), most
plausibly because the specific geometry family used to vary `x_neck` at fixed V2/L
introduces its own curvature artifact. The V2-dependence question (Section K/L) could
not be meaningfully answered with the same construction, for a related but distinct
reason (the local force probe not "seeing" the far-field volume at all). Both point to
the same next step: a smoother, kink-free static-family construction, before any
further conclusion about whether coarsening shifts the zero-force neck width can be
drawn. No hazard tuning, forced-event benchmark, long runs, or PR/de-sintering model
were attempted.
