# Milestone 15I — Matched Rate-Ratio and Driving-Reservoir Test of the GB/Surface Crossover

Starting checkpoint: `1c19fc7` (235/235 tests, clean worktree). Preserved: unified calibrated free
energy, physical `gamma_s`/`gamma_GB`/`M_GB` mapping, integrated physical `M_s`, `face_projected`
surface transport, constrained tangent-cone GB migration, anisotropic production `mu`, Cahn-Hoffman
endpoint force. Kept OFF: sink, hazard, RBM, independent `M_TJ`. No new `pf_sintering/` code was
needed — all work is diagnostic/campaign scripts reusing M15F-H's production modules unchanged
(235/235 tests still pass). `M_GB` was NOT increased further as the primary strategy — the primary
lever tested here is proportionally SLOWING both mobilities at fixed rate ratio `R`.

## Bottom line

- **Central hypothesis CONFIRMED, decisively.** M15H's "GB-reservoir-exhaustion" hypothesis — that
  fast GB migration burns through its own finite driving energy faster in physical time than slow
  GB migration does, even at the SAME instantaneous rate ratio — is now directly demonstrated, not
  just inferred. At matched t=0.05s, the fast-GB family member (`M_GB_scale=90, M_s_scale=1`,
  R=3) has its mobility-independent GB driving strength `D_eta` collapsed to 0.022, while the
  slow-both family member (`M_GB_scale=15, M_s_scale=1/6`, same R=3) still has `D_eta=1.39` —
  **a ~63x difference** at the identical nominal rate ratio.
- **First-ever C2 (sustained, >=0.1s) contact narrowing in the M15 series.** Slowing BOTH
  mobilities proportionally (family "C") at fixed R reaches C2 at every R tested (R=2: 0.108s
  duration; R=3: 0.135s; an even-slower probe: 0.170s) — while dip DEPTH stays essentially
  constant (~0.52-0.70nm) across the whole A->B->C progression at fixed R. This is the
  absolute-timescale effect the milestone was designed to isolate, cleanly separated from the
  rate-ratio effect M15H already established.
- **But: the fractional-narrowing criterion is NOT met.** The sustained dip is only ~1.5-2.1% of
  `L_contact` (0.5-0.7nm out of ~32-33nm), well short of Section 11's 5% alternative threshold.
  C2 is achieved by DURATION, not by fractional depth.
- **And: geometric narrowing does NOT translate into stress amplification here.** In both long
  (t=0.5s) C2 trajectories, `sigma_endpoint` DECLINES smoothly through the entire narrowing window
  (t~0.03-0.15s) rather than rising — the capillary force `F_cap_n` is relaxing faster (in
  magnitude) than the narrowing concentrates it. `A_sigma` stays at 1.00-1.04 in every promoted
  case. **C2 (geometric) is achieved; C3 (geometric + stress-coupled) is not.**
- **Anisotropy is still not required for the kinetic effect** (confirmed again): the isotropic
  AN0 control at the best slow condition shows an even DEEPER dip (0.700nm) than its anisotropic
  counterpart (0.565nm), both C2 with identical duration.
- **`R_L` (M15H's SURF/GB interaction remainder) is CONFIRMED as a genuine, finite, nonzero
  continuum-limit effect**, not finite-difference error: under TRUE dt refinement (dt, dt/2, dt/4
  at fixed physical window), `R_L/Delta_t` is essentially flat (5.567e-10 -> 5.618e-10 ->
  5.643e-10, <1.4% drift across a 4x refinement) — it does not vanish as dt->0.
- **S45/S60/S75/S100: all FAIL.** No genuine post-reset amplification reaches even the 45MPa
  intermediate milestone in any trajectory this campaign.
- **Section 15 decision: revisit the geometry/reservoir question before entering
  `delta>1/15`.** Slow-M_s DOES produce sustained narrowing, but it remains geometrically shallow
  (1.5-2.1%, not the >=5% that would represent a materially different neck shape) and does not
  drive stress. Per Section 15's own explicit decision tree, this is the "still cannot generate
  MEANINGFUL narrowing" branch, not the "sufficient narrowing but insufficient stress" branch —
  so strong/faceted anisotropy is **not** entered this milestone; the more urgent open question is
  whether the present finite geometry contains enough GB/coupling free-energy excess to support a
  materially deeper narrowing cycle at all.

## 1. Matched-R families (Sections 2-3)

`scripts/m15i_stage_matched_r.py`, `A=120nm,lambda=320nm,overlap=5nm,W=20nm/dx=2.5nm,gamma_GB/
gamma_s=1.4`, theta=45deg/AN3, `t_target=0.20s`, dense sampling 0-0.05s:

| R | family | M_GB_scale | M_s_scale | classification | max dip | duration |
|---|---|---|---|---|---|---|
| 2 | A (speed GB) | 60 | 1 | C1 | 0.566nm | 0.0305s |
| 2 | B (slow surface) | 30 | 0.5 | C1 | 0.565nm | 0.0510s |
| 2 | C (slow both) | 15 | 0.25 | **C2** | 0.566nm | **0.1075s** |
| 3 | A (speed GB) | 90 | 1 | C1 | 0.574nm | 0.0225s |
| 3 | B (slow surface) | 30 | 1/3 | C1 | 0.576nm | 0.0670s |
| 3 | C (slow both) | 15 | 1/6 | **C2** | 0.573nm | **0.1350s** |

All 6 ran cleanly (`F_monotonic=True`, mass conserved). The dip DEPTH is remarkably constant
across every family member at a given R (~0.565-0.576nm regardless of A/B/C) — confirming R
(chi's peak value) governs depth, while the ABSOLUTE mobility scale governs how long that depth
is sustained.

## 2. Central prediction: (90,1) vs. (30,1/3) (Section 5)

The single most important comparison in the milestone, both at R=3: `(M_GB_scale=90, M_s_scale=1)`
gives C1, 0.0225s; `(M_GB_scale=30, M_s_scale=1/3)` gives C1, 0.0670s — **nearly 3x longer duration
at the SAME depth**, from lowering both mobilities by a factor of 3 while holding R fixed. This
already shows the predicted direction clearly; family C (a further 2x slowdown from B) crosses into
C2 territory. **The self-limitation M15H observed was an absolute-timescale effect, not a
fundamental inability of the rate-ratio mechanism to narrow the contact** — confirmed.

## 3. Thermodynamic driving reservoirs D_eta(t), D_surf(t) (Section 6)

`scripts/m15i_driving_lib.py`: `D_eta` reuses `constrained_eta.structural_thermodynamic_force`
(the EXACT variational derivative `constrained_tangent_cone_eta_update` itself uses) and measures
`eta_dot_i` from an ACTUAL short GB-only trial (so it captures the real tangent-cone-projected +
`reproject`-corrected velocity, not an idealized unconstrained guess), giving
`Fdot_eta = dx^2*sum_i sum_grid(g_i*eta_dot_i)`, `D_eta=-Fdot_eta/M_eta` (verified >=0 always).
`D_surf` reuses `surface_transport.exact_dissipation_face_projected` (the ALREADY
machine-precision-verified `Fdot_chain=-D_h` identity, Milestone 13E) divided by `M_s` to strip
the mobility dependence — no new energy decomposition was invented for either.

| t | R3_A (M_GB=90) D_eta | R3_C (M_GB=15) D_eta | ratio C/A |
|---|---|---|---|
| 0.001 | 8.14 | 19.32 | 2.4x |
| 0.01 | 1.04 | 5.93 | 5.7x |
| 0.02 | 0.240 | 3.49 | 14.5x |
| 0.05 | 0.0221 | 1.39 | **63x** |

## 4. Direct test of the exhaustion hypothesis (Section 7)

**Confirmed unambiguously**: increasing `M_GB` (family A) causes `D_eta(t)` to collapse MUCH
earlier in PHYSICAL time — by t=0.05s it has fallen by ~370x from its t=0.001s value (8.14->0.022)
for the fast-GB case, vs. only ~14x for the slow-both case (19.3->1.39) over the identical physical
window. `chi(t)` tracks this directly: R3_A's `chi` crashes to 0.017 by t=0.05 (SURF completely
dominant, matching its D_eta near-total exhaustion), while R3_C's `chi` stays near 1 (1.003) at the
same physical time (matching its still-substantial D_eta). Lowering `M_s` at fixed baseline
`M_GB=30` (family B) gives an intermediate result in both `D_eta` decay rate and narrowing
duration, exactly as the linear-superposition picture would predict.

## 5. Operator-remainder dt scaling (Section 8)

`scripts/m15i_stage_dt_scaling.py`: at the t=0.02s canonical state, (a) the M15H-style `n_steps`
ladder at FIXED per-step `dt` (10,5,2,1) shows `R_L/Delta_t` decreasing from 5.57e-10 to 4.57e-10
as the window shrinks; (b) TRUE dt refinement (dt, dt/2, dt/4) at a FIXED total `Delta_t=10*dt0`
shows `R_L/Delta_t` essentially FLAT (5.567e-10 -> 5.618e-10 -> 5.643e-10, monotonically
*increasing* very slightly, not decreasing toward 0) under a genuine 4x step-size refinement.
**Conclusion: `R_L` is a real, finite, nonzero continuum-limit SURF/GB interaction term, not
finite-difference truncation error.** This does not change the SURF/GB sign result (already robust
across all of M15G-I); it confirms the small nonlinear coupling M15H flagged is physical.

## 6. Corrected mass-balance units (Section 9)

M15H's report mislabeled the control-volume mass-balance quantity as "kg" / "kg-equivalent." The
underlying computation (`sum(f)*dx^2`) was always correct; the label was wrong. In this 2-D model,
`integral f dA` has units of **m^2 per unit out-of-plane depth** — corrected throughout this
milestone's own reporting (`scripts/m15i_stage_shellmap.py`).

## 7. Shell balance / -div(J) reconciliation on the C2 trajectories (Section 10)

Repeated the shell mass balance (0-1W, 1-2W, 2-3W, 3-4W around the top TJ) on the R2_C and R3_C
states at t=0.01/0.05/0.10/0.15s (spanning before/during/after the narrowing window), using
`bc_ops.flux_divergence`'s exact primitive as the "authoritative cell-level counterpart" per
Section 10's instruction. **Same pattern as M15H holds throughout the entire narrowing episode**:
[0,1W) consistently GAINS area (~7-13e-22 m^2 per 5-step trial), [1W,2W)/[2W,3W)/[3W,4W)
consistently LOSE (with [2W,3W) the largest single contributor every time), and the direct
`Delta(sum f)` matches the flux-divergence prediction to <0.25% relative error in every shell,
every checkpoint — confirming the mechanism identified in M15G/H (surface diffusion feeding the
immediate neck from the whole surveyed outer band) persists unchanged even while genuine sustained
narrowing is occurring.

## 8. Long trajectories and the C2/C3 distinction (Sections 11-12)

Promoted `R2_C`, `R3_C`, an even-slower `R3_D` (`M_GB_scale=7.5, M_s_scale=1/12`, same R=3), and
the isotropic AN0 control at `R3_C`'s mobilities, all to t=0.5s (`R3_D` screened to t=0.2s first).

| case | classification | max dip | duration | A_sigma | sigma_reset | sigma_peak |
|---|---|---|---|---|---|---|
| R2_C (t=0.5s) | C2 | 0.516nm | 0.108s | 1.000 | 35.86MPa | 35.86MPa |
| R3_C (t=0.5s) | C2 | 0.565nm | 0.135s | 1.036 | 37.04MPa | 38.36MPa |
| R3_D (t=0.2s) | C2 | 0.571nm | **0.170s** | 1.000 | 39.59MPa | 39.59MPa |
| R3_C AN0 control (t=0.5s) | C2 | **0.700nm** | 0.135s | 1.000 | 37.95MPa | 37.95MPa |

Going even slower (R3_D) extends the narrowing duration further (0.170s), consistent with the
D_eta-exhaustion picture continuing to apply. All 4: `F_monotonic=True`, mass conserved.

**Fractional-reduction check (Section 11's alternative criterion)**: max dip / baseline
`L_contact` (~32-33nm at the pre-dip local max) is **1.5-2.1%** in every case — well short of the
5% threshold. **C2 by duration, not by depth.**

**C3 check (Section 12)**: examining the raw `sigma_endpoint(t)` trace for R3_C and R3_D directly
(not just the endpoint `A_sigma` summary) shows `sigma` declining smoothly and essentially
monotonically THROUGH the entire narrowing window (t~0.03-0.15s: ~46.2MPa -> ~42.1MPa for R3_C) —
the endpoint capillary force `F_cap_n` is relaxing in magnitude FASTER than the narrowing
concentrates it, so `sigma=-F_cap_n/L_contact` still falls even as `L_contact` itself is
genuinely shrinking. **No C3 trajectory was found.**

## 9. Isotropic vs. anisotropic (Section 14)

The isotropic AN0 control at R3_C's exact mobility settings gives a DEEPER dip (0.700nm) than the
anisotropic case (0.565nm) at identical duration (0.135s) — reconfirming (a third time, after
M15G's O2 and M15H's own isotropic-control finding) that the kinetic SURF/GB narrowing mechanism
is orientation-independent; favorable anisotropy's role remains a separate, additive stress boost,
not a driver of the underlying geometric competition.

## 10. High-stress gate (Section 13)

**S45/S60/S75/S100: all FAIL.** No trajectory this campaign shows genuine post-reset amplification
reaching even the 45MPa intermediate milestone — `sigma_endpoint` declines through the narrowing
window in every long run, and the small late-trajectory wobbles (`A_sigma` up to 1.036) never
approach it.

## 11. Strong/faceted anisotropy decision (Section 15)

**Not entered this milestone.** Section 15's decision tree offers two branches: (a) sustained
narrowing with insufficient stress -> faceting is well-motivated as a targeted perturbation on an
established mechanism; (b) narrowing itself remains insufficient -> revisit whether the geometry
contains enough GB/coupling free-energy excess before investing in faceting. This milestone's
result sits closer to (b): C2 was achieved by the DURATION criterion, but the geometric depth
(1.5-2.1%) is shallow and does not translate into any stress signal at all (not even a weak,
transient one) — the mechanism is real and now well-characterized (Sections 3-7 above), but not
yet "established" in the sense Section 15 intends (a mechanism whose main limitation is
capillary-force magnitude, which faceting could plausibly amplify). The more direct open question
for a future milestone is whether a DIFFERENT starting geometry (larger initial GB-excess reservoir
— e.g. a different misorientation/`gamma_GB` ratio, or a geometry with more initial dihedral-angle
imbalance) could deepen the narrowing episode past the 5% threshold using the SAME (now
well-quantified) absolute-timescale lever, before considering the added complexity of the
regularized faceting branch.

## Files added

- `scripts/m15i_stage_matched_r.py`: Sections 2-4 matched-R families.
- `scripts/m15i_driving_lib.py`: `D_eta`, `D_surf` (Section 6).
- `scripts/m15i_stage_driving.py`: Sections 6-7 D_eta(t)/D_surf(t)/chi(t) per family.
- `scripts/m15i_stage_dt_scaling.py`: Section 8 true-dt-refinement test.
- `scripts/m15i_stage_long.py`: Sections 11-12, 14 long trajectories + isotropic control.
- `scripts/m15i_stage_shellmap.py`: Sections 9-10 corrected-units shell balance.
- `runs/m15i_campaign/`, additions to `runs/m15f_campaign/` (both gitignored): all case results.

## Status

- 235/235 tests pass (no `pf_sintering/` changes this milestone).
- No hazard, first-passage integration, sink action, or RBM was implemented or activated.
- Per Section 16: **STOP here.**
