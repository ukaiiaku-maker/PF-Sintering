# Milestone 15G — Orientation-Selected Anisotropy and Contact-Broadening Mechanism Audit

Starting checkpoint: `0800c0a` (235/235 tests, clean worktree). Preserved: unified calibrated free
energy, physical `gamma_s`/`gamma_GB`/`M_GB`, integrated physical `M_s`, `face_projected` conserved
surface transport, constrained tangent-cone GB migration, the anisotropic production `mu` path and
Cahn-Hoffman endpoint force added in M15F. Kept OFF: sink, hazard, RBM, independent `M_TJ`. No new
broad `W`/geometry/mobility campaign was run (per Section 17).

## Bottom line

- **PASS-A (orientation matters, quantitatively): yes, modestly.** Rotating `theta_mis_deg` to
  40-45deg (aligning the anisotropy's easy axis with the TJ-flank's actual ~40-45deg normal
  direction, which M15F found was sitting near the *hard* orientation at `theta_mis_deg=0`)
  reproducibly increases `A_sigma` by ~2-2.6x the isotropic "excess" (`A_sigma-1`), at both a short
  screen (22 cases) and long (0.5s) trajectories (4 cases): isotropic control `A_sigma=1.023` vs.
  `theta=45deg, AN3` `A_sigma=1.059` (the best result found anywhere in the M15 series so far). Still
  solidly **H1** — nowhere near H2.
- **PASS-B (mechanism identified): yes, cleanly.** The operator audit (Section 8) found that surface
  diffusion (SURF) and GB migration (GB) act with **opposite sign** on `L_contact`: SURF consistently
  *broadens* the contact, GB consistently *narrows* it, at every checkpoint tested (t=0.01, 0.02,
  0.1s) and every anisotropy configuration tested (isotropic and `theta=45deg` AN2/AN3). The combined
  (FULL) trajectory is a **near-cancellation** of these two comparable-magnitude, opposite-sign
  effects — not GB migration being negligible, as a naive reading of the smooth, slowly-broadening
  `L_contact(t)` curves might suggest. This explains both the persistent small non-monotonic wobble
  in `sigma(t)` seen throughout Milestones 15-15F, and why the net drift is broadening: surface
  diffusion's broadening effect is quantitatively slightly larger than GB migration's narrowing
  effect at this rate-competition point (`M_GB_scale=30, M_s_scale=1`), by a THIN margin.
- **Energetic explanation (Section 11): SURF lowers `E_surface` only; GB lowers `E_coupling+E_GB`
  only.** The two operators are essentially independent in their *energetic* bookkeeping (`Delta
  F_total` is close to additive across SURF+GB vs FULL), but their *geometric* consequences on
  `L_contact` are opposed. Broadening is favorable because it is how the model relaxes pure
  surface/gradient energy (`E_surface`) toward the flatter, lower-curvature profile the free-surface
  double-well+gradient term prefers; narrowing is favorable for the GB/coupling energy for the
  opposite geometric reason. Which one wins on net is a close competition, not a foregone conclusion.
- **Flux budget (Section 9): mass flows TOWARD the TJ at 1W and AWAY from the TJ at 2W-4W**, at
  every checkpoint and configuration tested — a source-like redistribution centered around 1-2W from
  the TJ, consistent with classical Kuczynski-style neck-thickening (material drawn from the near
  particle surface and deposited at the immediate neck region), which is exactly the geometric
  process that widens `L_contact`.
- **TJ kinematics (Section 10) cross-validate the sign**: the direct level-set TJ velocity (no
  heuristic attribution) shows the top TJ moving *away* from the bottom TJ under SURF (consistent
  with broadening) and *toward* it under GB (consistent with narrowing), at every checkpoint —
  independent confirmation of the operator-level `L_contact` finding from a completely different
  calculation.
- **O2 verdict** (Section 13): favorable orientation changes the capillary force / stress magnitude
  measurably and reproducibly, but does **not** change the sign of contact evolution — `L_contact`
  still broadens throughout every long trajectory tested, at every orientation and anisotropy level.
- **S75/S100: still FAIL**, as in M15F.
- **Sections 14-16 (strong/faceted anisotropy): not attempted**, with documented rationale — see
  below.

## 1. Orientation screen (Sections 2, 5-7)

W=20nm/dx=2.5nm (Section 5: the width question is closed), G2 (A=120nm, lambda=320nm), overlap in
{5,10}nm, `theta_mis_deg` in {30,40,45,50,60}deg, AN0/AN2/AN3 (AN1 omitted per Section 3),
`gamma_GB/gamma_s=1.4, M_GB_scale=30, M_s_scale=1`, t_target=0.10s. **22/22 cases resolved and ran
cleanly** (`scripts/m15g_stage_orient.py`).

Top results (all `overlap=5nm` — `overlap=10nm` stayed uniformly weaker/flat, `A_sigma<=1.023`,
matching M15F):

| case | theta | AN | A_sigma | sigma_reset | sigma_peak | L_contact_reset |
|---|---|---|---|---|---|---|
| ov5, th45, AN3 | 45 | 0.0647 | **1.0536** | 36.21 MPa | 38.14 MPa | 35.3 nm |
| ov5, th40, AN3 | 40 | 0.0647 | 1.0531 | 36.07 MPa | 37.98 MPa | 35.4 nm |
| ov5, th50, AN3 | 50 | 0.0647 | 1.0432 | 35.90 MPa | 37.46 MPa | 34.9 nm |
| ov5, th45, AN2 | 45 | 0.045 | 1.0379 | 36.27 MPa | 37.64 MPa | 35.1 nm |
| ov5, AN0 (control) | — | 0 | 1.0000 | 37.40 MPa | 37.40 MPa | 36.1 nm |

**theta=40-45deg is the clear peak** of the orientation ladder (Section 2's `~45deg` hypothesis
confirmed), with a smooth falloff toward 30deg and 60deg — a genuine, well-resolved orientation
dependence, not noise. Every AN0 control stayed exactly `A_sigma=1.000` at this short horizon
(still in its initial monotonic descent, matching the same degeneracy pattern noted in M15E/M15F's
short screens) — the anisotropic cases' amplification is real, not a screening-window artifact of
the control.

## 2. Positive-stiffness anisotropy result

Confirmed (`aniso_delta<1/15` throughout): AN2 (`d=0.045`) and AN3 (`d=0.0647`, at the edge of the
positive-stiffness boundary) both show the SAME qualitative orientation dependence, with AN3
consistently giving slightly stronger amplification than AN2 at the same orientation (e.g. th45:
AN3=1.0536 vs AN2=1.0379) — a smooth, monotone trend in anisotropy amplitude, with no discontinuity
or instability, as expected in the smooth (non-faceted) regime.

## 3-4. Promotion and long favorable-orientation trajectories (Sections 7, 12)

Promoted 4 candidates (`scripts/m15g_stage_long.py`) to t=0.5s: the AN0 isotropic control, and the
top 3 anisotropic screen results (th45/AN3, th40/AN3, th45/AN2), all at `overlap=5nm`.

| case | theta | AN | A_sigma | sigma_reset | sigma_peak | t_reset | t_peak |
|---|---|---|---|---|---|---|---|
| AN0 control | — | 0 | 1.023 | 37.27 MPa | 38.12 MPa | 0.40 | 0.50 |
| th45, AN3 | 45 | 0.0647 | **1.059** | 36.00 MPa | 38.13 MPa | 0.125 | 0.45 |
| th40, AN3 | 40 | 0.0647 | 1.057 | 35.71 MPa | 37.74 MPa | 0.125 | 0.45 |
| th45, AN2 | 45 | 0.045 | 1.047 | 36.27 MPa | 37.95 MPa | 0.03 | 0.45 |

All 4: `F_monotonic=True`, mass drift `~1e-14`, clean full trajectories. The orientation effect
**persists and even strengthens slightly** at the long horizon relative to the short screen (best
`A_sigma` moved from 1.0536 at t=0.10s to 1.059 at t=0.5s) — this is a genuine, reproducible
physical effect, not a short-window artifact. All four are **H1** (`1<A_sigma<1.5`). `L_contact` is
monotonically broadening in every one of the four trajectories from its true minimum (`t_reset`)
onward — `Lc_reset == Lc_min` in all four rows, i.e. no sustained narrowing was ever achieved, even
at the most favorable orientation/anisotropy combination tested.

## 5. Contact-width operator decomposition (Section 8)

Implemented `scripts/m15g_mechanism_lib.py::operator_audit`: given a saved state, runs three SHORT
(100 production-`dt` steps, `Delta_t~1-1.3ms` depending on config), NON-COMMITTED trial evolutions
(SURF: surface diffusion only; GB: GB migration only; FULL: both, matching the canonical dynamics)
from the SAME state, and reports `Delta(L_contact)` for each. Run at three checkpoint times per
config (`scripts/m15g_stage_mechanism.py`: t=0.01s "initial post-transient," t=0.02s "stress
minimum" region, t=0.10s "later loading state"), for isotropic AN0 and `theta=45deg` AN2/AN3
(Section 8's specified comparison set).

**Result, consistent across all 3 checkpoints x all 3 configs (9 audits, 27 operator trials):**

| operator | Delta L_contact (typical, t=0.01) | sign |
|---|---|---|
| SURF | +1.7 to +1.8e-10 m | **broadens** |
| GB | -1.6 to -1.7e-10 m | **narrows** |
| FULL | +1.3 to +1.8e-11 m | small residual (near-cancellation) |

At t=0.02s FULL is even slightly *negative* for all three configs (a real, if fleeting, narrowing
instant); by t=0.1s FULL is again small and positive. **SURF and GB are comparable in magnitude and
opposite in sign at every checkpoint** — the smooth, slowly-broadening `L_contact(t)` curves seen
throughout Milestones 15-15F are the residual of a close, non-trivial competition, not evidence
that GB migration is unimportant for contact width. The interaction remainder `R_coupled =
Delta_FULL - Delta_SURF - Delta_GB` is itself comparable in size to the individual terms (e.g.
`-5.0e-10` in a standalone t=0 check), confirming the two operators do not simply commute/add —
reported as a diagnostic, not assumed away.

## 6. Surface-flux budget (Section 9)

`m15g_mechanism_lib.flux_budget`, reusing `curvature_extraction.branch_mu_J_profile` with the
production-consistent (anisotropy-aware) `mu`/flux fields, sampled at control sections
1W/2W/3W/4W from each TJ along the particle-side branch. Consistent across every checkpoint and
config tested:

- **At 1W (closest to the TJ): `J_tangent<0`** — mass flows TOWARD the TJ (into the immediate neck
  region). Magnitude grows with time (`-2.6e-10` at t=0.01 to `-2.4e-9` at t=0.1, for the AN0
  config).
- **At 2W-4W: `J_tangent>0`** — mass flows AWAY from the TJ (out along the broader particle
  surface), roughly constant in magnitude (`~1-3e-10`) across checkpoints.

This is a genuine source-like redistribution centered around 1-2W from the TJ: material is drawn
from the broader particle surface (2W-4W) and deposited immediately at the neck (near 1W and
closer) — precisely the classical Kuczynski-style neck-thickening picture, and directly consistent
with (not merely correlated with — causally the same process as) SURF's observed `L_contact`
broadening in Section 5. **Mass is flowing toward the contact overall, not away from it** — the
"broadening" is neck growth (thickening), not the neck receding.

## 7. Direct level-set TJ velocity (Section 10)

`tj_velocity_kinematic`: solves `grad(A).v=-dA/dt, grad(B).v=-dB/dt` locally at each TJ (`A=f-0.5`,
`B=eta1-eta2`), using the before/after states of each operator trial (midpoint-field gradients,
Eulerian finite-difference time derivatives, bilinearly sampled at the TJ location) — no heuristic
attribution to either process, a direct consequence of the two moving level sets' intersection
kinematics. **Cross-validates the operator-level `L_contact` sign independently**: at every
checkpoint, the top TJ's velocity has **positive y-component under SURF** (moving away from the
bottom TJ — consistent with broadening) and **negative y-component under GB** (moving toward the
bottom TJ — consistent with narrowing) — e.g. at t=0.01s (AN0): `v_TJ_top,y = +1.38e-7` (SURF) vs
`-5.27e-8` (GB). This is an independent calculation (local gradients + Eulerian time derivatives of
the raw fields) arriving at the same conclusion as the geometric `L_contact` measurement from
Section 5, a meaningful cross-check.

## 8. Energy decomposition (Section 11)

Reusing `m15f_campaign_lib.energy_ledger_f`'s components (`E_surface`, `E_coupling`, `E_GB`) before
and after each trial:

- **SURF**: `Delta E_surface<0` (always, e.g. `-7.02e-10` at t=0.01), `Delta E_coupling` small
  (responds passively to `f`'s change), **`Delta E_GB` EXACTLY 0** (eta doesn't move under SURF).
- **GB**: **`Delta E_surface` EXACTLY 0** (f doesn't move), `Delta E_coupling<0` and `Delta E_GB<0`
  (both, always — e.g. `-3.27e-10` and `-1.06e-10` at t=0.01).
- **FULL**: close to the sum of the two (energy descent is close to additive, e.g.
  `-1.22e-9 ~ -6.93e-10 + -4.33e-10` at t=0.01) even though the GEOMETRIC (`L_contact`) consequence
  is a near-cancellation, not additive.

**Physical explanation for why broadening is energetically favorable**: it is exactly how the model
relaxes the pure surface (double-well + gradient) energy `E_surface` — surface diffusion moves mass
to reduce local curvature/flatten the free-surface profile, and in this specific neck geometry that
manifests as a widening (thickening) contact. Narrowing is favorable for a DIFFERENT part of the
energy (`E_coupling+E_GB`, the GB/obstacle-potential terms) for the analogous reason on the eta
side. The two mechanisms are essentially decoupled in energy bookkeeping but geometrically opposed,
and the net observed behavior (slow broadening, with a persistent wobble) is the signature of this
being a close, non-trivial competition rather than either mechanism dominating.

## 9. O1-O4 classification (Section 13)

**O2**: favorable orientation changes `F_cap`/`sigma` measurably and reproducibly (theta=45deg,
AN3 gives `A_sigma` 2.6x the isotropic control's excess amplification, the strongest result found
anywhere in the M15 series), but does **not** change the sign or qualitative character of contact
evolution — `L_contact` broadens throughout every trajectory tested regardless of orientation or
anisotropy amplitude, and the operator audit shows the SAME opposing-sign SURF/GB competition (with
SURF winning) at every anisotropy configuration tested, not a reversal at the favorable orientation.

## 10. S75/S100 status

**FAIL**, same conclusion as M15F: best `A_sigma=1.059`, `sigma_peak=38.13MPa`, far short of
`A_sigma>=2.5`/`sigma>=75MPa` (S75) or `A_sigma>=3.33`/`sigma>=100MPa` (S100).

## 11. Strong/faceted anisotropy gate (Sections 14-16): not attempted

Section 14 gates entry into the `delta>1/15` regularized-faceting regime on the favorable
orientation having been tested in the positive-stiffness regime first — done (Sections 1-4 above).
Section 16's optional strong-anisotropy screen is explicitly conditioned on Section 15's
qualification passing, and Section 13 itself frames **O3** ("changes the sign/rate of contact
evolution") as the threshold that makes further investment in faceting physics worthwhile ("O3
alone is scientifically important even if S75 is not yet reached"). This milestone found **O2**,
not O3: a real, reproducible, but purely quantitative reinforcement of the SAME already-understood
mechanism (SURF-vs-GB competition, SURF still winning), not a qualitative change that a faceted/
missing-orientation surface might plausibly reverse. Given the substantial additional
implementation and qualification cost Section 15 itself specifies (isolated-particle Wulff-shape
construction, facet-length grid convergence, a full 2-DOF asymmetric-flank anisotropic
Young-Herring solve — none of which exist in the codebase yet), and that Section 17 explicitly
discourages further broad exploration without a stronger trigger, **the regularized-faceting branch
was not pursued this milestone**. This is a considered decision given the evidence, not a resource
shortfall — a future milestone revisiting this should first look for an O3-producing lever (e.g. a
different rate-competition point where GB narrowing already nearly wins, so a modest anisotropy-
driven push might flip the sign) before spending the qualification budget Section 15 requires.

## 12. Physical explanation for why `L_contact` broadens (not narrows)

Synthesizing Sections 5-8 above: in this model, at the physical point explored throughout
Milestones 15-15G (`M_GB_scale=30, M_s_scale=1`, i.e. GB migration scaled down and surface
mobility at its full physical baseline), **surface diffusion is the numerically dominant of two
competing, opposite-sign, energy-lowering processes**. Surface diffusion draws mass from the
broader particle free surface (2W-4W from the TJ) and deposits it immediately at the neck (within
~1W), lowering the pure surface energy `E_surface` — a real, physical Kuczynski-style neck-
thickening process, not a numerical artifact. GB migration acts in the opposite geometric direction
(narrowing) while lowering a DIFFERENT part of the free energy (`E_coupling+E_GB`). Because the two
are comparable in magnitude, the net `L_contact(t)` trend is a thin-margin competition — explaining
both the observed net broadening AND the small, persistent, non-monotonic wobble in `sigma(t)`
throughout every trajectory of every M15-series milestone. Favorable-orientation anisotropy
(Section 1-4) shifts this competition's absolute stress level upward measurably, but not by enough
to change which side wins.

## Files added

- `scripts/m15g_stage_orient.py`: Sections 2/5-7 orientation screen.
- `scripts/m15g_stage_long.py`: Section 12 long favorable-orientation trajectories.
- `scripts/m15g_mechanism_lib.py`: Sections 8-11 operator-trial/kinematic/flux/energy audit
  machinery (`trial_evolve`, `tj_velocity_kinematic`, `flux_budget`, `operator_audit`).
- `scripts/m15g_stage_mechanism.py`: runs the Section 8-11 audit at 3 checkpoints x 3 configs.
- `runs/m15f_campaign/`, `runs/m15g_campaign/` (gitignored): all case results.

## Status

- 235/235 tests pass with all new code in place (no changes to `pf_sintering/` itself this
  milestone — all new code is diagnostic/campaign scripts reusing M15F's production modules
  unchanged).
- No hazard, first-passage integration, sink action, or RBM was implemented or activated.
- Per Section 20: **STOP here.**
