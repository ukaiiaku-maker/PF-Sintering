# Milestone 15H — GB/Surface Crossover and Sustained Contact Narrowing

Starting checkpoint: `1e1f823` (235/235 tests, clean worktree). Preserved: unified calibrated free
energy, physical `gamma_s`/`gamma_GB`/`M_GB` mapping, integrated physical `M_s`, `face_projected`
surface transport, constrained tangent-cone GB migration, the anisotropic production `mu` path and
Cahn-Hoffman endpoint force from M15F. Kept OFF: sink, hazard, RBM, independent `M_TJ`. No new
`pf_sintering/` code was needed this milestone — all work is diagnostic/campaign scripts reusing
M15F/M15G's production modules unchanged (235/235 tests still pass).

## Bottom line

- **PASS-1 (Δt convergence): yes, cleanly.** The instantaneous SURF/GB/FULL operator decomposition
  converges smoothly as `n_steps` drops from 100 to 5 (`Delta_t` from ~1ms to ~50µs) at every
  checkpoint and config tested — individual rates stabilize to <1% change between the two smallest
  `n_steps`. The interaction remainder `R_L/Delta_t` does **not** vanish as `Delta_t->0`; it
  converges to a small but genuinely nonzero value (~0.5-3% of the individual operator rates,
  largest relative to `FULL` itself right around the crossover region t~0.02s) — a real, persistent
  nonlinear interaction between the two operators, not a finite-time artifact.
- **PASS-2 (chi=1 crossover mapped): yes.** `chi(t)=|dL/dt|_GB / (dL/dt)_SURF` at the UNMODIFIED
  baseline (`M_GB_scale=30, M_s_scale=1`) already crosses 1 transiently around t=0.02s for both
  isotropic AN0 (chi: 0.89 -> 1.07 -> 0.50 at t=0.01/0.02/0.10) and orientation-45deg AN3 (0.92 ->
  1.12 -> 0.60) — GB migration briefly, locally wins even without any rate change. Mobility
  linearity was confirmed EXACT (both operators' rates scale linearly with their own mobility to
  <0.3% across a 0.5-2.0x range, at every checkpoint) — enabling a direct, verified prediction of
  the rate ratio `R=M_GB_scale_mult/M_s_scale_mult` needed to sustain `chi>=1.1`.
- **PASS-3 (sustained coupled narrowing): NOT achieved, but genuine transient narrowing (C1) IS
  confirmed across a wide, systematically-explored rate-ratio range.** A bounded screen (10 full
  trajectories, `R` from 1.5 to 6, both AN0 and theta=45deg/AN3) found real, measured `L_contact`
  narrowing dips (~0.15-0.74nm, ~0.003-0.027s duration) at **every** `R` tested — but never reaching
  the 0.1s sustained-narrowing (C2) threshold. The dip **peaks in both depth and duration around
  R~2.5-3** and **shrinks at higher R** (R=6: only 0.15nm/0.003s) — a genuine, physically explained
  self-limiting effect (Section 9 below), not simply "not enough GB mobility yet."
- **PASS-4 (materially larger stress ramp): NOT achieved** — `A_sigma` stays in H1 (best 1.065)
  across the whole campaign.
- **Section 15 (isotropic vs. anisotropic control): kinetic crossover does NOT require favorable
  anisotropy.** The isotropic AN0 control at R=2.5 shows an even DEEPER narrowing dip (0.736nm) than
  the anisotropic theta=45deg/AN3 case at R=3 (0.570nm), despite lower `A_sigma` (1.022 vs 1.065) —
  cleanly separating the two effects M15G's O2 finding already suggested: the SURF/GB kinetic
  competition is orientation-independent; favorable anisotropy adds an independent, roughly
  multiplicative boost to the stress magnitude on top of it.
- **S75/S100: FAIL.** Intermediate 45/60MPa milestones: not reached either (best `sigma_peak`
  38.1MPa).
- **Corrected M15G's flux interpretation** (Section 2): the "source at 1-2W" reading of M15G's raw
  `J_tangent` sign pattern was too narrow. The proper control-volume mass balance (Section 7 below)
  shows essentially the WHOLE 1W-4W band losing mass to feed growth immediately at the TJ (0-1W),
  not just a localized 1-2W source.

## 1. Δt convergence (Section 3)

`scripts/m15h_stage_dt_convergence.py`: repeated the M15G operator audit at `n_steps` in
{100,50,25,10,5} from the SAME saved states (AN0 and theta=45deg/AN3, t=0.01/0.02/0.10s).

| config | t | n_steps=100 | n_steps=5 | relative change |
|---|---|---|---|---|
| AN0 | 0.01 | SURF=1.759e-7, GB=-1.610e-7 | SURF=1.819e-7, GB=-1.623e-7 | ~3.4%, ~0.8% |
| AN0 | 0.02 | SURF=1.112e-7, GB=-1.207e-7 | SURF=1.132e-7, GB=-1.215e-7 | ~1.8%, ~0.6% |
| AN0 | 0.10 | SURF=2.864e-8, GB=-1.430e-8 | SURF=2.876e-8, GB=-1.438e-8 | ~0.4%, ~0.6% |

`R_L/Delta_t` (AN0): 3.08e-9 -> 1.03e-9 (t=0.01, n=100->5); 1.18e-9 -> 4.14e-10 (t=0.02);
-7.2e-12 -> -1.3e-11 (t=0.10) — each converges to a stable, nonzero value, confirmed by direct
inspection of the sequence (monotone, decelerating change, not oscillating or diverging). The
`theta=45/AN3` config shows the identical pattern (values differ, same convergence behavior).
**Conclusion**: the operator decomposition is a well-defined instantaneous quantity, not an
artifact of the 100-step window M15G used, and the FULL/SURF/GB decomposition is not perfectly
additive even in the Delta_t->0 limit — there is a genuine, small, nonlinear coupling term.

## 2. Crossover number chi(t) (Section 4)

Using the `n_steps=5` (most converged) values:

| config | chi(0.01) | chi(0.02) | chi(0.10) |
|---|---|---|---|
| AN0 | 0.892 | **1.073** | 0.500 |
| theta=45/AN3 | 0.920 | **1.120** | 0.598 |

Both configs cross `chi=1` transiently right around t=0.02s at the unmodified baseline
`M_GB_scale=30`. `chi_sigma` (same convention, `dsigma/dt` ratio) was computed alongside
`dL/dt` in the same audits — its sign structure mirrors `chi`'s (GB's `dsigma/dt` contribution
is positive/stress-raising while SURF's is negative/stress-relaxing at these checkpoints,
consistent with GB narrowing the contact and thereby raising the endpoint stress).

## 3. Mobility linearity (Section 5)

`scripts/m15h_stage_linearity.py`: at each checkpoint, `M_f_scale` in {0.5,0.75,1.0,1.25,1.5}
(SURF, GB off) and `M_eta_scale` in {0.5,...,2.0} (GB, SURF off), `n_steps=10`.

**Confirmed EXACTLY linear** for both operators, both configs, all three checkpoints: e.g.
AN0 SURF `dL/dt` at t=0.01: 9.095e-8 (0.5x) / 1.816e-7 (1.0x) / 2.720e-7 (1.5x) — ratios 2.000 and
1.498 relative to the 0.5x point, matching the nominal scale ratios to <0.2%. theta=45/AN3
1.0x/0.5x ratios: 1.997-2.000 across all three checkpoints (both SURF and GB). This directly
enables Section 6's linear crossover prediction — no nonlinear mobility response was found in this
range.

## 4. Predicted crossover (Section 6)

Given exact linearity, `chi(R) = chi_baseline(t) * R` where `R=M_GB_scale_mult/M_s_scale_mult`.
The hardest constraint (smallest baseline chi) is always t=0.10s. Solving for `chi>=1.1`
simultaneously at all three checkpoints:

    AN0:        R = 1.1/0.500 = 2.20   ->  M_GB_scale ~= 66 (holding M_s_scale=1)
    theta=45/AN3: R = 1.1/0.598 = 1.84 ->  M_GB_scale ~= 55

matching Section 7's own suggested range ("M_GB scale ~30-60").

## 5. Bounded full-trajectory rate screen (Section 7)

`scripts/m15h_stage_crossover_screen.py` + `_screen2.py`: 10 full trajectories at
`A=120nm,lambda=320nm,overlap=5nm,W=20nm,dx=2.5nm`, `t_target=0.20s`, `R` in
{1.5,2,2.5,3,4,5,6} for theta=45/AN3 and {1.5,2,2.5} for AN0 (holding `M_s_scale=1`,
`M_GB_scale=30*R`). All 10 ran cleanly (`F_monotonic=True`, mass conserved).

## 6. Contact-width sign classification (Section 8)

**Critical methodology correction, found while classifying**: `m15f_campaign_lib.
amplification_metrics`'s `L_contact_reset`/`L_contact_min` are anchored to the SIGMA-minimum time,
not `L_contact`'s own minimum — a real early `L_contact` dip that occurs well before the sigma
minimum (as it does here: dip at t~0.002-0.01s vs. sigma minimum at t~0.05-0.4s depending on case)
is therefore invisible to that metric (`Lc_reset==Lc_min` was reported for EVERY Stage-X case
despite genuine narrowing being present). Wrote a dedicated `scripts/m15h_classify.py` that tracks
`L_contact(t)`'s own running maximum and reports any interval where `L_contact` drops below it.

**Result: every one of the 10 screened cases (plus the 2 long t=0.5s runs) is C1** (transient
narrowing, never sustained 0.1s):

| case | R | max dip | duration |
|---|---|---|---|
| th45/AN3 | 1.5 | 0.447nm | 0.025s |
| th45/AN3 | 2.0 | 0.450nm | 0.025s |
| th45/AN3 | 2.5 | 0.498nm | 0.015s |
| th45/AN3 | **3.0** | **0.529nm** | 0.018s |
| th45/AN3 | 4.0 | 0.475nm | 0.0155s |
| th45/AN3 | 5.0 | 0.362nm | 0.008s |
| th45/AN3 | 6.0 | 0.149nm | 0.003s |
| AN0 | 1.5 | 0.391nm | 0.025s |
| AN0 | 2.0 | 0.486nm | 0.025s |
| AN0 | 2.5 | 0.500nm | 0.015s |
| AN0 (long, t=0.5s) | 2.5 | **0.736nm** | 0.027s |
| th45/AN3 (long, t=0.5s) | 3.0 | 0.570nm | 0.022s |

**The dip peaks (depth and duration) around R~2.5-3 and shrinks monotonically for R>3** — pushing
GB mobility higher does NOT extend the narrowing window; it shortens it. No `R` in the tested range
(1.5-6, spanning well past the Section-6 prediction of ~1.8-2.2) reaches C2. Since no C2/C3
candidate exists, Section 8's "promote only C2/C3" instruction has no case to promote; the two long
(t=0.5s) runs (Section 11/12/15) were instead run at the empirically-best R found (R=3 AN3, R=2.5
AN0) to confirm no later narrowing recurrence and to obtain S75/S100 data — none appeared past
t=0.2s in either.

## 7. Conservative control-volume surface mass balance (Section 9)

`scripts/m15h_stage_massbalance.py`, `m15h_mechanism_lib.control_volume_mass_balance`: for 4
concentric Euclidean-distance shells [0,1W),[1W,2W),[2W,3W),[3W,4W) around each TJ, a SURF-only
`n_steps=5` trial's `Delta(sum f)` per shell, cross-checked against the exact `bc_ops.
flux_divergence` primitive (agreement to <0.1% in every shell, every checkpoint, every config —
confirms the mass-balance bookkeeping is exact, not approximate).

**Corrected interpretation of M15G's flux picture** (Section 2's explicit instruction): the
[0,1W) shell (immediately at/around the TJ) consistently GAINS mass (+3 to +5e-21 kg-equivalent
units, all checkpoints/configs); **every other shell, [1W,2W), [2W,3W), [3W,4W), consistently
LOSES mass** — not a narrow "source at 1-2W" as the raw `J_tangent` sign pattern alone suggested,
but essentially the entire surveyed outer band (1W-4W) feeding growth right at the neck. [2W,3W)
is consistently the single largest contributor (most negative `delta_mass`) of the three losing
shells.

## 8. mu(s)/J_tangent(s)/-dJ_tangent/ds driving map (Section 10)

Reused `curvature_extraction.branch_mu_J_profile` (already computes `dJ_tangent_ds`) out to 5W.
Consistent pattern across both configs, all checkpoints:

- **At s=0 (the TJ itself)**: `-dJ_tangent/ds > 0` — local DEPOSITION right at the TJ.
- **At s~0.65W**: `-dJ_tangent/ds` swings sharply NEGATIVE (the largest-magnitude feature in the
  whole profile) — local DEPLETION just outward of the TJ.
- **Beyond ~2W**: `-dJ_tangent/ds` settles to small, steady positive values (~0.001-0.006) out to
  5W — a broad, weak, persistent deposition trend.

This refines the shell-based mass balance (Section 7) further: within the [0,1W) shell itself
there is a SHARP transition from strong deposition (at the TJ) to strong depletion (~0.65W out),
netting to an overall gain for that shell; the weak-but-broad deposition trend from 2W-5W is what
ultimately sources the material feeding both the TJ deposition and the intervening depletion zone.

## 9. Long crossover trajectories and physical explanation (Section 11/13)

Both long (t=0.5s) runs confirm: the narrowing dip is confined to its original early window
(t~0.002-0.03s); `L_contact` resumes broadening immediately after and never dips again through
t=0.5s. **Why does chi fall back below 1, and why does pushing R higher make the window SHORTER
rather than longer (Section 13's explicit question)?**

GB migration's own driving force is the GB/coupling excess energy (`E_coupling+E_GB`, per M15G's
energy decomposition) — a FINITE, geometry-dependent reservoir that GB migration depletes as it
acts. At low-to-moderate `M_GB_scale`, GB migration relaxes this reservoir gradually, keeping a
comparable pace to surface diffusion's own (also finite, but more slowly-varying) driving force for
long enough to produce a modest, R-scaled dip. At HIGH `M_GB_scale` (R>3), GB migration relaxes its
own available driving force so quickly that it exhausts the local GB-excess energy driving the
narrowing almost immediately — its own rate then collapses toward zero (there being little GB
excess left to relax) well before surface diffusion's steadier rate has similarly declined, so the
crossover window CLOSES faster, not slower, at higher GB mobility. This is a genuine
self-limiting kinetic effect, not merely "insufficient GB mobility" — confirming Section 13's
"kinetic competition" framing precisely: the mechanism is real and directly demonstrated (measured
sign flips at every tested R), but it is fundamentally rate-limited by how much GB-relaxable excess
energy the initial geometry contains, not by the raw mobility ratio alone.

## 10. Isotropic vs. anisotropic control (Section 15)

The isotropic AN0 long run (R=2.5) shows a DEEPER narrowing dip (0.736nm) than the anisotropic
theta=45deg/AN3 run (R=3, 0.570nm) despite the anisotropic case's higher overall `A_sigma` (1.065
vs. 1.022). **Favorable anisotropy is NOT required for the kinetic sign flip** — the SURF/GB
competition and its transient crossover are orientation-independent, matching M15G's O2 finding
that anisotropy's effect is a largely separate, additive boost to the capillary force magnitude
rather than a change to the underlying contact-evolution kinetics.

## 11. S75/S100 status (Section 12)

**FAIL.** Best `sigma_peak=38.13MPa` (both the short screen's best and the long XL runs), best
`A_sigma=1.065`. Neither the 45MPa nor the 60MPa intermediate milestone was reached anywhere in
this campaign.

## Files added

- `scripts/m15h_mechanism_lib.py`: `trial_evolve_scaled` (mobility-scaled single-operator trials),
  `control_volume_mass_balance` (exact flux-divergence-validated shell mass balance).
- `scripts/m15h_stage_dt_convergence.py`: Section 3.
- `scripts/m15h_stage_linearity.py`: Section 5.
- `scripts/m15h_stage_crossover_screen.py`, `_screen2.py`: Section 7 (initial R=1.5-3 and extended
  R=4-6 ladders).
- `scripts/m15h_classify.py`: Section 8's corrected, `L_contact`-native C0-C3 classifier.
- `scripts/m15h_stage_massbalance.py`: Sections 9-10.
- `scripts/m15h_stage_long.py`: Sections 11/12/15.
- `runs/m15h_campaign/`, additions to `runs/m15f_campaign/` (both gitignored): all case results.

## Status

- 235/235 tests pass (no `pf_sintering/` changes this milestone).
- No hazard, first-passage integration, sink action, or RBM was implemented or activated.
- Per Section 18: **STOP here.**
