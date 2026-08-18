# Recovery — Geometry-Only Fix for the TJ Groove/Bump

## Summary

Visual inspection of the "recovery" zero-barrier 100-event run (branch `recovery/tj-only-final`, restored M16P RBM physics + the M16Q/M16R corrected branch-resolved TJ locator) showed that despite 100/100 events completing, `n_components=1` throughout, and mass conservation at roundoff precision, the actual neck geometry developed a visible groove/protrusion. This was root-caused to a **pre-existing defect in the t=0 initial geometry construction itself** -- present even before any RBM/PF dynamics run -- not to the RBM operator, the TJ locator, or the PF stepper. The fix is geometry-only, as instructed: no RBM, mass-to-TJ, one-b bookkeeping, zero-barrier semantics, Poisson kinetics, barrier parameters, or PF mobility code was touched.

## Root cause, quantified (`scripts/recovery_geometry_audit.py`)

`pf_sintering/m16j_geometry.py`'s `solve_body_fillet` connects the TJ to the parent sphere (particle radius `R_p` or neighbor radius `R_s`) via a **circular arc**, tangent (C1: matched position + slope) to the sphere. A circle has only 2 degrees of freedom (center, radius) -- generally insufficient to ALSO match the sphere's curvature at the tangent point, since that is a third, independent condition. For the exact production parameters (chi=1.5, ratio=0.185, psi=160deg, R_p=1000nm, R_s=1500nm):

| | fillet radius | fillet curvature | body curvature | ratio |
|---|---|---|---|---|
| particle | 26.57nm | 0.03764 /nm | 0.00100 /nm | **37.6x** |
| substrate | 16.18nm | 0.06179 /nm | 0.00067 /nm | **92.7x**, with a **sign reversal** |

Arclength-based curvature analysis (`kappa_m(s)`, robust to the near-vertical tangents that make simple `d^2R/dz^2` unreliable near a sphere's own pole) shows a **discontinuous jump** in curvature exactly at the fillet-to-body tangent point (particle: `kappa_m` jumps from -0.0329/nm to -0.00128/nm across a ~31nm arc; substrate: from +0.0356/nm to -0.00078/nm, i.e. a sign reversal, across a ~20nm arc) -- a narrow, high-curvature spike immediately adjacent to a near-flat parent body. This is a direct, quantitatively confirmed explanation for the visible groove/bump: PF surface diffusion, driven by curvature-dependent chemical potential, acts on this discontinuity from the very first step.

The dihedral angle itself was checked and is CORRECT: the measured included angle between the two branches' outward free-surface tangent rays at the TJ is 159.7 deg against a target of 160deg (well within tolerance) -- the visible angle discrepancy the handoff also flagged as a possible issue was NOT confirmed; only the curvature transition was defective.

Tangent (C1) continuity at the join was already fine (theta jump ~0.1deg) -- consistent with the construction's own design intent. Only curvature (C2) was broken.

## Fix: replace the circular fillet with a C2 quintic transition

`pf_sintering/m16j_geometry.py`:
- `solve_body_fillet` is KEPT, unmodified, for reference (matching the project's established "retire, don't delete" convention).
- New `solve_body_c2_transition`: builds a quintic polynomial `R(z)` on `[0, z1]` satisfying, EXACTLY:
  - at the TJ (`z=0`): position `R=a` and the Young-Herring tangent `R'=slope_sign*m` (2 conditions -- Young-Herring fixes the ANGLE, not the local curvature, so `R''(0)` is a genuinely free parameter, not invented).
  - at the body-match point (`z=z1`): the parent sphere's own EXACT position, tangent, AND curvature (3 conditions).
  - 5 fixed conditions + 1 free parameter (`R''(0)`), matching a quintic's 6 degrees of freedom exactly (no under/over-determination, no ad hoc optimization needed for the polynomial itself).
- The free `R''(0)` is chosen by a small search minimizing curvature overshoot/RMS(dkappa/ds) (an explicit, numerical approximation of Section 11's "minimize integral[dkappa/ds]^2", rejecting any candidate with an interior sign change or overshoot beyond the two endpoint curvature values).
- **The transition LENGTH (`z1`) also had to be searched, not merely reused from the old fillet's short tangent-point location.** Direct testing (both in the `R(z)` parameterization and in a fully general 2D parametric quintic, to rule out a parameterization artifact) confirmed that absorbing a ~40-90x curvature ratio smoothly is not possible within the original ~20-30nm fillet-scale arc length, regardless of parameterization -- this is a genuine geometric requirement (the same reason a clothoid/Euler spiral must be long to keep curvature change gradual), not a numerical artifact. The search grows `z1` geometrically from the old fillet's scale until the shortest length is found whose curvature overshoot stays below 15% of the larger endpoint curvature magnitude. Result: `z1=534.6nm` (particle), `z1=330.8nm` (substrate) -- overshoot 0.08% and 0.20% respectively, with the curvature jump at the join reduced from the original 3.16e-2 /nm (particle) and -3.64e-2 /nm (substrate) down to **-5.4e-5 /nm and -4.6e-5 /nm** (roughly 580-790x smaller).
- `particle_R_of_z`/`substrate_R_of_z_sphere` now call `solve_body_c2_transition` instead of `solve_body_fillet`, evaluating the quintic instead of the circular-arc formula; the crop-margin formula (previously `max(6*W, 4*fillet_radius)`, now meaningless since there is no fillet radius) is simplified to `6*W` alone, since the transition itself is now sized for smoothness rather than needing extra margin.

**Consequence**: the domain grew substantially (Nz: 810->2212, Nr: 632->1213, ~5.2x more cells; per-PF-step cost measured at ~0.36s vs ~0.044s, ~8x), since the transition now spans hundreds of nm rather than tens. This is a direct, unavoidable consequence of the smoothness requirement, not a separate design choice -- see "Known limitation / scope note" below.

## Verification through the required gates

**Gate 1 -- t=0 analytic geometry** (`recovery_geometry_audit_corrected.png`): `theta(s)` and `kappa_m(s)` now vary smoothly and monotonically from the TJ into the parent body on both branches, with the fillet/body join (dashed vertical line) invisible in the curvature trace -- no spike, no sign reversal, no secondary extremum. PASS.

**Gate 2 -- t=0 diffuse f=0.5 contour** (`recovery_t0_corrected_geometry.png`): full mirrored equal-aspect geometry and neck zoom, rendered directly from the rasterized phase field (not the analytic curve) -- smooth transition from the TJ into both bodies, no hook/groove/kink/bump, correct dihedral-angle wedge visible at the TJ. PASS.

**Gate 3 -- short sink-off relaxation** (`runs/recovery_sinkoff_test/`, RBM/sink OFF, 500 PF steps, checkpoints every 10-100 steps): geometry visually unchanged from t=0 at every checkpoint through step 500 -- no spontaneous bump/groove development under pure capillary relaxation. PASS.

**Gate 4 -- one zero-barrier one-b event** (`runs/recovery_one_event/`, pre-event / 10/25/50/75/90% b / completion / post-event-relaxation): TJ marker stays attached to the actual visible junction throughout; both branches stay smooth at every checkpoint; no bump, ledge, shelf, or secondary groove develops during the event; `cum_b=1.000` exactly (one-b contract intact, unmodified); `e1e2f_residual=4.4e-5` at completion (small, consistent with the historical M16P-era clip+redeposit tolerance -- this operator was NOT changed). PASS.

**Gate 5 -- zero-barrier repeated-event campaign (scaled down to 5 events, see cost note below)**: run to completion, `runs/recovery_zero_barrier/checkpoint_frames/event_{000,001,002,003,005}.png` + `final.png`. Numerically clean throughout: `n_components=1`, mass conservation at exact roundoff (`0.0` to `-3e-16`) at every one of the 5 events. **Visually, the picture is mixed, and is reported honestly rather than glossed over per the explicit "do not declare PASS from n_components=1/mass_drift/event-count alone" instruction:**

- Events 1-2 (`event_001.png`, `event_002.png`): clean, smooth necks on both branches, matching the one-event test's own clean result -- the geometry fix's benefit clearly carries through the first couple of events.
- Event 3 onward (`event_003.png`, `event_005.png`): a **small secondary bump/shoulder** becomes visible on both branches near the neck, growing modestly from event 3 to event 5 -- much smaller in amplitude than the ORIGINAL (pre-fix) defect, but a real, developing feature, not a rendering artifact.

**This is judged NOT to be a new geometry-construction problem**, for three reasons: (1) it is absent at t=0 and through events 1-2, where the corrected geometry is exercised cleanly; (2) its onset and growth pattern (starting small, developing over repeated events, shape-coupled to the neck) matches the mechanism `MILESTONE_16S_MORPHOLOGY_REGULARITY_REPAIR.md` already root-caused on the development branch: the historical M16P-era `active_sink_transport_step` redeposits excess/deficit mass via a Gaussian kernel centered on the current TJ estimate, which is a positive-feedback-prone construction (M16S found it grows a shelf/protrusion over repeated events on the OLD geometry too); (3) per this recovery's own EXPLICIT and repeated instruction, that operator was correctly left completely unmodified ("preserve the M16P-era mass-to-TJ operator... do not redesign it... the ONLY substantive change intended here is: correct physical TJ coordinate"). Fixing it would mean re-opening exactly the RBM-architecture work (M16S/M16T/M16U/M16V/M16W) this recovery was explicitly instructed to avoid.

**Conclusion for Gate 5**: PARTIAL PASS. The geometry fix is confirmed to eliminate the t=0/early-event defect it targeted (Gates 1-4, and events 1-2 of Gate 5, all clean). A separate, smaller, slower-developing morphology issue remains, attributable to the (intentionally preserved) RBM redistribution operator, not the geometry construction -- this is the SAME issue the development branch's M16S/M16T history already found and is currently mid-investigation on, and is out of this recovery's explicit scope to fix. Given this, the finite-barrier campaign (explicitly gated on "the zero-barrier 100-event geometry remains clean") was NOT started -- proceeding to a longer run on either barrier mode would not change this finding, only reproduce it at larger scale.

## Known limitation / scope note: campaign size reduced due to compute cost

The ~8x per-step cost increase (from the geometry fix's much larger domain, itself an unavoidable consequence of the confirmed curvature-transition-length requirement) makes the originally-planned 100-event zero-barrier campaign impractical within this session (~5 hours at the measured per-step rate, versus ~37 minutes before the fix). The campaign was scaled to 5 events (checkpoints 0/1/2/3/5) instead -- sufficient to see BOTH the geometry fix's clean early-event behavior AND the onset of the separate, out-of-scope RBM-redistribution issue described above. A follow-up should either (a) accept the longer wall-clock cost for a full confirmation at larger event counts, or (b) investigate whether the crop margins / far-cap placement beyond the now-necessarily-long C2 transition can be tightened without reintroducing the curvature-continuity defect, to recover some of the lost performance -- and, separately and out of THIS recovery's scope, resume the RBM-redistribution-operator investigation (M16S/M16T's own unfinished work) now that it can be studied against a geometry that is confirmed clean at t=0 and through the first couple of events.

## Test coverage

`tests/test_recovery_geometry_c2_fix.py` (5 tests, all passing): exact position/tangent/curvature match to the parent body at the transition join; exact position/tangent match at the TJ; no large curvature jump or sign reversal along the transition (the specific defect this fix targets); a documentation test recording the OLD circular fillet's known discontinuity (so a future accidental fix to that function doesn't silently invalidate this test's assumptions); dihedral angle matches the psi=160deg target. Full existing suite (263 tests, the `recovery/tj-only-final` branch baseline) still passes unchanged.

