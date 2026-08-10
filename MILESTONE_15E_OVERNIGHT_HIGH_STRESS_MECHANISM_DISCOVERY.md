# Milestone 15E — Overnight High-Stress Mechanism Discovery

Starting checkpoint: `971b11a` (235/235 tests passing, clean worktree). Preserved: unified
Milestone-14G free energy, calibrated physical `M_GB`, `face_projected` surface diffusion,
constrained tangent-cone GB migration, endpoint capillary-force construction. Kept OFF: sink,
hazard, RBM, anisotropy, independent `M_TJ`. No hazard/sink/RBM/anisotropy was implemented or
activated at any point in this milestone.

## 1. Bottom line

- **GLOBAL-S75 / GLOBAL-S100: FAIL.** The best trajectory found across an 18+36+6+7-case
  adaptive screen (Stages A–D) reaches only `A_sigma=1.038` (`sigma_reset=31.1MPa` →
  `sigma_peak=32.3MPa`), classified **H1** (weak amplification, `1<A_sigma<1.5`). No candidate
  approached `A_sigma>=1.5`, let alone the `2.5`/`3.33` targets for 75/100 MPa.
- **Stage D (narrow-overlap, dx=1.25nm) closes off the geometric escape route.** Overlaps
  `<=7.5nm` fail TJ/subgrid-contact resolution even at doubled resolution — not a
  discretization artifact but a structural mismatch (`overlap < interface_width=20nm` is an
  intrinsically ill-posed initial condition for this diffuse-interface formulation). The one
  resolvable narrow case (`overlap=10nm`) gives `L_contact=41.6nm`, still ~1.5–2x too wide for
  the `<=26.7nm`/`<=20nm` needed for 75/100 MPa, and shows no post-reset amplification at all.
- **`gamma_s` sensitivity (Section 10): amplification does NOT survive, and weakens with
  higher `gamma_s`** (`A_sigma`: 1.038 at `gamma_s=1.0` → 1.025 at `1.5` → 1.006 at `2.0`).
  Scaling surface energy is not a path to higher driving force here, confirming Section 10's
  warning that `gamma_s` should not be used to manufacture a result.
- **LOCAL configurational-force track (Sections 13–16): LOCAL-HOLD.** A diffuse
  Eshelby/configurational-force diagnostic was derived directly from this project's free-energy
  functional, one real bug was found and fixed (see Section 3 below) bringing the isolated-
  planar-interface benchmark to a clean pass, but the diagnostic does **not** reproduce the
  correctly-*signed* Young-Herring residual through equilibrium (order of magnitude is right,
  sign does not flip between a narrower- and wider-than-equilibrium wedge the way the
  already-validated `tj_force.compute_tj_force` ground truth does). Per Section 16 this is
  reported as unresolved and **not** used as a hazard-track candidate.
- **Recommended next physics step:** neither further `M_GB`/`M_s` rate-competition tuning nor
  `gamma_s` scaling is likely to reach 75–100 MPa with this endpoint-capillary/contact-width
  mechanism (diminishing returns, consistent with and now re-confirmed beyond Milestone 15D).
  The two directions with remaining unexplored value are (a) a proper grand-potential-corrected
  configurational-force construction for the conserved `f` field (see Section 4), which could
  still reveal a large *local* driving force even though the *global* endpoint-stress metric is
  saturated near 30 MPa, and (b) revisiting whether `interface_width` itself (fixed at 20nm
  throughout this campaign) is the binding geometric constraint — Stage D suggests any path to
  `L_contact<=27nm` will require either a substantially narrower diffuse interface or an
  entirely different contact-narrowing mechanism than initial-overlap reduction.

## 2. Campaign structure and results (Stages A–D)

All cases are restartable (`runs/m15e_campaign/{case_id}.json` idempotency markers,
append-only `runs/m15e_campaign/manifest.csv`), gitignored. Scripts: `scripts/m15e_campaign_lib.py`
(shared infrastructure), `scripts/m15e_stage_{a,b,c,d}.py`, `scripts/m15e_gamma_s_sensitivity.py`.

### Stage A — cheap morphology screen (dx=2.5nm, t=0.06s, 18 cases)

`(A_nm,lambda_nm) x overlap_nm` = 6 morphologies × 3 overlaps, `gamma_GB/gamma_s=1.4`,
`M_GB=100*M_GB_ref`, `M_s=baseline/10`. **11 valid, 7 rejected** (mostly
`subgrid_contact_not_resolved` for the `A=140nm` morphologies and the `A=80,overlap=30nm` case;
one `sinusoid_self_intersects`).

**Key finding:** every valid Stage-A case was still in its initial monotonic-relaxation phase at
`t=0.06s` (`sigma_endpoint` strictly falling, `L_contact` strictly broadening) — `A_sigma`
was degenerate at exactly `1.000` for 8/11 cases (the reset minimum coincides with the final
sample because no turnover had occurred yet). This means the raw Section-3 ranking instruction
("rank by `A_sigma`, not `t=0` stress") could not discriminate morphologies at this short
horizon. Endpoint-slope flattening (least-negative `d(sigma)/dt` over the last 4 samples) was
used as the Stage-B seed tiebreak instead — `A_A100_L360_ov30` showed the clearest flattening
(slope an order of magnitude smaller than the rest, with the first hint of an uptick), and was
promoted alongside `A_A120_L280_ov20`, `A_A120_L320_ov15`, and other near-flattest morphologies.

### Stage B — adaptive physics refinement (dx=2.5nm, t=0.10s, 36 cases)

Aspect ratio {1.5,2.0,3.0}, `gamma_GB/gamma_s` {1.0,1.4,1.7}, and rate competition
`(M_GB_scale,M_s_scale)` in {(30,1),(100,0.1),(100,1/30)} around the 4 best Stage-A morphologies.
**33 valid, 3 rejected** (`aspect=3.0` broke particle-arc resolution for 3 of the 4 morphologies).

**Key finding:** the `(M_GB_scale,M_s_scale)=(30,1.0)` rate-competition point (slower GB
migration, full-baseline surface mobility) was the only physics variant to show genuine
post-reset upturn within the 0.10s window, on both surviving morphologies it was tried on
(`A120_L280_ov20`: `A_sigma=1.017`; `A120_L320_ov15`: `A_sigma=1.015`) — consistent with Milestone
15D's established direction (slower `M_GB` relative to `M_s` favors stress buildup). All
`gamma_GB/gamma_s` and `aspect_ratio` variants alone stayed at the same degenerate `A_sigma=1.000`
as Stage A.

### Stage C — longer horizon (dx=2.5nm, t=0.5s, top 6 candidates)

All 6 candidates ran cleanly to completion (`F_monotonic=True`, `mass_drift_max<1e-14` in all
cases). Full results:

| case_id | gamma_GB/gamma_s | M_GB_scale | M_s_scale | sigma_reset | sigma_peak | A_sigma | L_contact_reset=min | class |
|---|---|---|---|---|---|---|---|---|
| `B_A120_L320_ov15_MGB30_Ms1.0` | 1.4 | 30 | 1.0 | 31.07 MPa | 32.25 MPa | **1.038** | 47.5 nm | H1 |
| `B_A120_L280_ov20_MGB30_Ms1.0` | 1.4 | 30 | 1.0 | 30.63 MPa | 31.55 MPa | 1.030 | 50.8 nm | H1 |
| `B_A100_L360_ov30_gbratio1.7`  | 1.7 | 100 | 0.1 | 26.81 MPa | 27.21 MPa | 1.015 | 65.9 nm | H1 |
| `B_A100_L360_ov30_aspect2`     | 1.4 | 100 | 0.1 | 26.77 MPa | 27.16 MPa | 1.014 | 66.0 nm | H1 |
| `B_A100_L360_ov30_gbratio1.4`  | 1.4 | 100 | 0.1 | 26.77 MPa | 27.16 MPa | 1.014 | 66.0 nm | H1 |
| `B_A100_L360_ov30_MGB100_Ms0.1`| 1.4 | 100 | 0.1 | 26.77 MPa | 27.16 MPa | 1.014 | 66.0 nm | H1 |

Best: `A_sigma=1.038`, `sigma_peak=32.25MPa` — **H1** by Section 6's classification, far short of
H2 (`1.5<=A_sigma<2.5`). `L_contact` never narrows below its reset value in any of the 6 cases
(`L_contact_reset == L_contact_min` in every row) — the mild sigma rebound comes from continued
endpoint-force realignment, not contact narrowing, matching Milestone 15D's finding that
alignment is already near its ceiling and narrowing is the missing ingredient.

**This meets Section 12's explicit pivot trigger** (`A_sigma<1.5` and `sigma_peak<60MPa` for
all 6 defensible conditions tried).

### Stage D — narrow-contact high resolution (dx=1.25nm, t=0.03–0.15s)

Before fully pivoting, Section 7's narrow-overlap regime (flagged as unresolved-but-possibly-
promising in Milestone 15D) was checked once at doubled resolution, on the best Stage-C
morphology (`A=120nm, lambda=320nm`):

| overlap_nm | status | reason |
|---|---|---|
| 2 | rejected | `tj_force_not_resolved` |
| 3 | rejected | `tj_force_not_resolved` |
| 4 | rejected | `subgrid_contact_not_resolved` |
| 5 | rejected | `subgrid_contact_not_resolved` |
| 7.5 | rejected | `subgrid_contact_not_resolved` |
| 10 | **ok** | `A_sigma=1.000`, `L_contact_reset=41.6nm` (no amplification within 0.03s) |

Doubling resolution (dx=2.5nm→1.25nm) did **not** rescue the narrow-overlap cases — the failure
mode is structural, not a discretization artifact: `interface_width=20nm` was held fixed across
the whole campaign (per the Stage A–D scripts' `W_nm=20.0` default), so an initial overlap of
2–7.5nm is smaller than the diffuse interface's own thickness, leaving no room for a distinct,
resolvable solid contact/TJ structure to form regardless of grid `dx`. The only overlap that
*does* resolve (10nm) still gives a contact width (41.6nm) roughly double what 75 MPa requires.
Per Section 9, since neither `L_contact<25nm` nor `A_sigma>1.5` was reached, no ultra-fine
(dx=0.625nm) qualification run was attempted.

### `gamma_s` material-family sensitivity (Section 10)

Run on the single best trajectory (`C_B_A120_L320_ov15_MGB30_Ms1.0000`), same physical
mobilities, `gamma_GB/gamma_s` held fixed at 1.4:

| gamma_s (J/m^2) | sigma_reset | sigma_peak | A_sigma | L_contact_reset | F monotonic |
|---|---|---|---|---|---|
| 1.0 | 31.07 MPa | 32.25 MPa | 1.038 | 47.5 nm | yes |
| 1.5 | 31.26 MPa | 32.05 MPa | 1.025 | 48.8 nm | no (max +0.9% relative uptick, t=0.125→0.15s) |
| 2.0 | 30.92 MPa | 31.12 MPa | 1.006 | 50.1 nm | yes |

Amplification **weakens monotonically** with increasing `gamma_s`, and the endpoint stress stays
essentially flat (~31 MPa) regardless of `gamma_s` — the mechanism is not a simple function of
surface energy scale, and higher `gamma_s` does not help. The `gamma_s=1.5` case shows a small
(0.9% relative) non-monotonic total-energy uptick between two samples, most likely from
`reproject`'s non-variational clipping correction (tracked elsewhere in this project via
`cum_safety_correction`/`cum_safety_ratio`); given its tiny magnitude relative to the F-scale, it
does not change the qualitative conclusion and was not investigated further.

## 3. Local configurational-force diagnostic (Sections 13–16)

Implemented in `pf_sintering/configurational_force.py`. Derivation, both correct and initially
incorrect, is documented in the module's own docstring; summary:

- Raw local energy density (`E_surface+E_coupling+E_GB` from this project's own `energy_ledger`)
  does **not** vanish in bulk solid even far from any real grain boundary (`E_coupling` alone is
  `-0.5*Wc` at `f=1`, single grain) — using it directly made the "isolated planar interface"
  benchmark force grow ~linearly with contour radius instead of vanishing. Fixed by using the
  same background-subtracted excess-energy construction this project already validated for scalar
  GB-excess-energy bookkeeping (`gb_excess_energy`, Milestone 15B Section 11): `excess_coupling =
  Wc*e1*e2*f*(2-f)`, `excess_grad = (k_eta/2)(|grad e1|^2+|grad e2|^2) - (k_eta/2)|grad f|^2`.
- A hand-guessed tanh profile is not a genuine equilibrium state of the real coupled dynamics
  (`evolve_f`'s chemical potential has a nonzero coupling term even for an isolated single grain);
  every benchmark profile was relaxed through the actual `evolve_f`/`evolve_eta`/`reproject` steps
  before evaluating the contour force (`scripts/m15e_conf_force_benchmarks.py`).
- Benchmark **A** (relaxed isolated planar interface): **PASS** — `|F_conf|/gamma_s` falls from
  1.2% (r=2W) to 0.5% (r=5W), consistent with no enclosed defect.
- Benchmark **B** (relaxed equilibrium wedge, psi=120deg): **FAIL** — a clean radius-independent
  plateau exists (`|F_conf|/gamma_s ~1.14` for r>=4W) but it does not approach zero; relaxation
  itself let the measured dihedral angle drift from 120deg to ~116deg, so a nonzero residual at
  that point may be partly genuine, but 1.14*gamma_s is far larger than expected for a ~4deg
  deviation.
- Benchmark **C** (off-equilibrium wedges, psi=100/140deg vs ground-truth `F_TJ` from
  `tj_force.compute_tj_force`): magnitude plateaus cleanly (`|F_conf|/|F_TJ|~3.0` and `~4.5`
  respectively, radius-independent for r>=3W) and direction is perfectly aligned/anti-aligned
  (`|cos_sim|=1.0` in both cases) — but the **sign relative to F_TJ does not flip** between the
  narrower- and wider-than-equilibrium cases the way the already-validated ground truth does
  (`cos_sim=+1.0` at psi=100, `cos_sim=-1.0` at psi=140, i.e. anti-aligned in the second case).
  `|F_conf|` also grows asymmetrically with `|psi-120|` (0.85 vs 1.45 for the same 20deg
  deviation), inconsistent with the clean antisymmetric restoring-force behavior `F_TJ` exhibits.

**Conclusion: LOCAL-HOLD** (Section 16) — a numerically well-defined, radius-independent plateau
exists, but it does not reproduce the known signed physics, so it is not qualified for use as a
hazard-track coordinate. One ruled-out hypothesis is documented in the module: treating `f`,
`e1`, `e2` as three independent fields versus eliminating `e2=f-e1` via the exact `e1+e2=f`
constraint gives *algebraically identical* tensors (verified by hand), so the bug is not there.
The leading remaining hypothesis is that `f`'s dynamics are conserved (Cahn-Hilliard-type, not
simple Allen-Cahn relaxation), which generally requires a grand-potential correction
(`psi - mu*f` relative to a reference chemical potential) to the energy density before taking the
Eshelby tensor — deriving and validating that correction is out of scope for tonight and is the
right next step for a future milestone. No hazard/first-passage/sink code was written or wired to
this diagnostic in either its passing or failing form.

## 4. Files added

- `scripts/m15e_campaign_lib.py` — restartable campaign infrastructure (idempotent per-case JSON,
  append-only manifest CSV, Section-3 reject-before-dynamics validation, `A_sigma` extraction).
- `scripts/m15e_stage_a.py`, `_stage_b.py`, `_stage_c.py`, `_stage_d.py` — the four screening
  stages described above.
- `scripts/m15e_gamma_s_sensitivity.py` — Section 10's material-family sweep.
- `pf_sintering/configurational_force.py` — the diffuse Eshelby/configurational-force
  diagnostic (derivation, benchmarks documented in its own docstring).
- `scripts/m15e_conf_force_benchmarks.py` — Section 14's three benchmarks (planar interface,
  equilibrium wedge, off-equilibrium wedge vs `tj_force` ground truth).
- `runs/m15e_campaign/` (gitignored) — all 70 case results (manifest + per-case JSON).

## 5. Status

- 235/235 tests still pass with all new code in place (confirmed after adding
  `configurational_force.py`).
- No hazard, first-passage integration, sink action, RBM, or anisotropy was implemented or
  activated, per Section 17.
- Per Section 20: **STOP here.**
