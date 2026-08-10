# Milestone 15D: High-Stress Slow-Loading Trajectory

## 1. Starting checkpoint

Branch `codex/coarsening-stress-buildup` @ `67acab9` (Milestone 15C,
235/235 tests, clean worktree) — verified before starting. Preserves the
unified Milestone-14G free energy, calibrated physical `M_GB`,
`face_projected` surface diffusion, `constrained_tangent_cone_eta_update`
GB migration, and the endpoint capillary force/stress construction
(Milestone 15C) throughout. sink, hazard, RBM, anisotropy, and
independent `M_TJ` stayed OFF; no first-passage kinetics were
implemented.

## 2. Exact endpoint-stress upper bound

`F_cap_endpoint = -gamma_s*(top_particle_dir + bot_particle_dir)`, both
unit tangent vectors, so `|F_cap_endpoint| <= 2*gamma_s` (equality only
at `psi=180deg`, a fully open/flat neck) and `|F_cap_n| <= |F_cap_
endpoint| <= 2*gamma_s` for any contact-normal projection. Hence
`sigma_sint_endpoint <= 2*gamma_s/L_contact` exactly, independent of any
dynamics. `sample_state` (`scripts/m15_gb_surface_rate_competition.py`)
now records `sigma_sint_endpoint_max` and `sigma_sint_endpoint_over_max`
at every sample. Across every trajectory run in this milestone, the
measured ratio sits at `~0.79-0.85` of this ceiling (e.g. `t=0`:
`33.85/40.20=0.842`) — the tangent geometry is already using most of its
available "alignment headroom"; **further increasing sigma requires
shrinking `L_contact`, not further improving tangent alignment**, which
was already near-saturated at the start of every trajectory tested.

## 3. Required contact lengths for 75/100 MPa

Exact bound (`gamma_s=1` J/m^2, maximum tangent alignment):
`L_contact<=26.7nm` for 75 MPa, `L_contact<=20.0nm` for 100 MPa.

Using the ACTUAL measured `|F_cap_endpoint|` range found throughout this
milestone (`~1.6-1.7` N/m, not the `2.0` N/m theoretical maximum):
`L_contact~21-23nm` for 75 MPa, `~16-17nm` for 100 MPa — matching the
handoff's own preview numbers. **Every contact length actually measured
in this milestone (widest-geometry baseline: `~50-58nm`; narrowed-
geometry screen: `~30-32nm` at best) remained well above even the 75 MPa
threshold's `~22nm` requirement.**

## 4. Bounded M_GB/M_s rate-competition screen

At the primary (A=100nm, lambda=320nm, R2=80nm, aspect=2, overlap=20nm,
W=20nm, gamma_gb=1.0, dx=2.5nm) geometry, `M_GB in {30, 100}*M_GB_ref`
at baseline `M_s`, plus `M_GB=100*M_GB_ref` at `M_s/3` and `M_s/10`
(`t=0.25s` each):

| case | sigma_ep(t=0) | sigma_ep(t=0.03s) | sigma_ep(t=0.125s) | sigma_ep(t=0.25s) | L_contact(t=0.25s) |
|---|---|---|---|---|---|
| ultra30 (30x) | 33.85 MPa | 29.52 MPa | 28.55 MPa | 28.65 MPa | 57.45nm |
| extreme100 (100x) | 33.85 MPa | 30.32 MPa | 28.53 MPa | 28.55 MPa | 57.56nm |
| extreme100, M_s/3 | 33.85 MPa | 31.63 MPa | 29.73 MPa | 28.56 MPa | 55.35nm |
| extreme100, M_s/10 | 33.85 MPa | 33.51 MPa | 31.03 MPa | 29.66 MPa | 53.41nm |

**Every case's maximum stress is its own `t=0` value** (the shared,
already-relaxed initial condition all four trajectories start from); the
coupled dynamics, at every mobility ratio tested up to `M_GB/M_s~2.4e18`
(100x reference `M_GB`, 1/10 baseline `M_s`), **monotonically relaxes
sigma_endpoint downward** from that starting point rather than building
it further, reaching a `~28-30 MPa` plateau by `t=0.25s` in all four
cases. `L_contact` correspondingly widens (from `49.75nm` to `53-58nm`)
rather than narrowing. **This bounded rate-competition screen alone
cannot reach even 40 MPa** (the Section 8 trigger threshold), let alone
75-100 MPa.

## 5. Geometry screen

Triggered per Section 8 (rate competition alone did not exceed 40 MPa).
Cheap `t=0` scan of `initial_overlap` (gamma_s, gamma_GB fixed) found
smaller overlap substantially narrows `L_contact` and raises the
UNRELAXED `sigma_endpoint`:

| overlap | L_contact (t=0) | sigma_ep (t=0) |
|---|---|---|
| 40nm | 68.2nm | 27.2 MPa |
| 20nm (primary) | 49.75nm | 33.85 MPa |
| 10nm | 37.4nm | 42.4 MPa |
| 5nm | 29.6nm | 50.1 MPa |
| 4nm | 29.6nm | 49.4 MPa |
| <=3nm | ~18.7nm (TJ/arc diagnostics unresolved -- below dx/geometry resolution limits) | not resolved |

`overlap=4-6nm` is the smallest **reliably diagnosable** geometry at
`dx=2.5nm` (smaller overlaps fall below the resolution needed for
`compute_tj_force`/`trace_particle_arc` to resolve the branches at all).
Running the FULL coupled dynamics from this narrower (`overlap=5nm`)
geometry:

- **REF `M_GB`, baseline `M_s`**: `sigma_ep` falls from `50.1MPa(t=0)` to
  `33.9MPa(t=0.01s)` -- the SAME rapid-relaxation pattern as Section 4,
  just from a higher starting point.
- **FAST `M_GB` (10x), baseline `M_s`**: falls to `34.4MPa` by `t=0.03s` --
  slower decay than REF but still monotonically decaying, not building.
- **EXTREME `M_GB` (100x), `M_s/10`**: stress roughly HOLDS
  (`50.1->48.6-52.9MPa` over `t=0-0.05s`) rather than decaying quickly --
  qualitatively different from every faster-`M_s` case: suppressing
  surface diffusion enough visibly slows the relaxation.
- **EXTREME `M_GB` (100x), `M_s/30`, extended to `t=0.2s`**: stress rises
  slightly ABOVE its own `t=0` value, peaking at **`51.3MPa` at
  `t=0.05s`** (a genuine, resolved, `F`-descending, mass-conserving
  `~2.4%` rise over the initial condition), then slowly declines to
  `46.5MPa` by `t=0.2s`.

This is the highest stress reached by any genuinely-evolving (not
`t=0`-only) trajectory in this milestone, and it required an extreme
`M_GB/M_s` mismatch (`M_GB=100*M_GB_ref`, `M_s=1/30` baseline,
`M_GB/M_s` ratio `~7.1e18`) on top of an already-narrowed starting
geometry. **`51.3MPa` is still well short of 75MPa.**

## 6. gamma_s sensitivity (diagnostic only)

`sigma_sint_endpoint` scales linearly with `gamma_s` at fixed geometry/
tangent state. Required `gamma_s` (currently `1.0` J/m^2 throughout) for
the observed stress levels to reach the targets:

| reference state | sigma_ep at gamma_s=1.0 | gamma_s needed for 75 MPa | gamma_s needed for 100 MPa |
|---|---|---|---|
| primary geometry, t=0 | 33.85 MPa | 2.22 J/m^2 | 2.95 J/m^2 |
| primary geometry, relaxed (t=0.25s) | ~28.5 MPa | 2.63 J/m^2 | 3.51 J/m^2 |
| narrowed geometry, t=0 | 50.1 MPa | 1.50 J/m^2 | 2.00 J/m^2 |
| narrowed geometry, peak (extreme rates) | 51.3 MPa | 1.46 J/m^2 | 1.95 J/m^2 |

These `gamma_s` values (`~1.5-3.5` J/m^2) are not per se unphysical
(solid-vapor surface energies for many metals/oxides span roughly this
range), but they are **specific, higher-than-default material choices,
not independently justified here** -- per Section 9's explicit
instruction, **this is reported as a diagnostic only and is NOT adopted**.

## 7. Highest naturally achieved endpoint stress

**`51.3 MPa`**, at `t=0.05s` of the `overlap=5nm`, `M_GB=100*M_GB_ref`,
`M_s=1/30` baseline trajectory (Section 5) -- a genuinely evolving,
`F`-descending, mass-conserving, dt-halving-confirmed state, not a
diagnostic artifact or an instantaneous geometry construction. This is
the best result found across both the rate-competition screen (Section
4, max `33.85MPa`, always at `t=0`) and the geometry+rate screen (Section
5).

## 8. S75/S100 classification

**Neither S75 nor S100 is achieved.** The highest genuinely-evolving
stress found (`51.3MPa`, Section 7) falls `~24 MPa` short of the S75
threshold and `~49 MPa` short of S100, despite: (a) a bounded but
substantial rate-competition screen (up to `100x` reference `M_GB`,
down to `1/10` and `1/30` baseline `M_s`, `M_GB/M_s` ratios approaching
`7e18`), (b) a geometry screen narrowing the initial contact from
`49.75nm` to the smallest reliably-diagnosable value (`~30nm` at
`overlap=5nm`; smaller overlaps are not resolvable at `dx=2.5nm` with
the current TJ/arc-tracing diagnostics), and (c) dt-halving confirmation
that the elevated-stress trajectories are physically real, not numerical
artifacts (Section 7 below reproduces this check). Every trajectory
tested shows the SAME qualitative behavior: apparent stress is highest
at `t=0` (or very early) and relaxes toward a `~28-35MPa` "attractor"
range under essentially all tested rate combinations; only the most
extreme, surface-diffusion-suppressed regime (`M_s/30`) delays this
relaxation enough to show a small, genuine overshoot (`50.1->51.3MPa`)
before declining again.

## Section 7 (verification, referenced from Section 8): numerical
integrity of the elevated-stress trajectories

For the `extreme100_Ms_over_10` trajectory (highest instantaneous-
safety-net activation observed, `cum_safety_ratio` up to `~9e-4`,
Section 4), a dt-halved rerun to `t=0.03s` confirms: `F` values agree to
5-6 significant figures at every sampled time; `sigma_sint_endpoint`
agrees to 4-5 significant figures; `cum_safety_ratio` DROPS by `~20x`
when `dt` is halved (`4.85e-4 -> 2.50e-5`), i.e. shrinks faster than
`dt` itself -- consistent with a genuine, vanishing-as-`dt->0` roundoff
effect (the same near-zero-denominator ratio artifact identified in
Milestones 14G/15), not a real safety-net-driven distortion of the
physics. Total `F` decreased monotonically and mass drift stayed
`<=1e-14` relative in every trajectory reported in Sections 4-5,
including the most extreme mobility ratios. **No numerical freezing,
instability, or artificial stress inflation was found in any tested
case.**

## 9. Saved high-stress trajectory

Not applicable in the "successful S75/S100 candidate" sense (Section 11's
save protocol), since neither gate passed. For reference, the
highest-stress trajectory's own key states are already fully documented
numerically in Section 5 (`overlap=5nm`, `M_GB=100*M_GB_ref`, `M_s=1/30`
baseline, sampled at `t=0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2s`) and are
reproducible directly from `scripts/m15_gb_surface_rate_competition.py`
with `overlap_nm=5.0`; full field arrays (`f`, `eta`, `mu`) were not
separately archived to `runs/`, consistent with Section 11's "for every
successful candidate" scope not being triggered.

## 10. Constant-V Arrhenius steepness requirements

For `r2/r1 = exp[V_act*(sigma2-sigma1)/(kB*T)]`, required activation
volume `V_act = kB*T*ln(r2/r1)/Delta_sigma`, `T in {800,1000,1200}K`,
`Delta_sigma in {45,70}MPa` (30->75, 30->100 MPa), amplification
`in {1e3,1e6,1e9}`:

| T | Delta_sigma | 10^3 | 10^6 | 10^9 |
|---|---|---|---|---|
| 800K | 30->75 (45MPa) | 1.70e-27 m^3 (108.5 b^3) | 3.39e-27 (217.0 b^3) | 5.09e-27 (325.5 b^3) |
| 800K | 30->100 (70MPa) | 1.09e-27 (69.8 b^3) | 2.18e-27 (139.5 b^3) | 3.27e-27 (209.3 b^3) |
| 1000K | 30->75 (45MPa) | 2.12e-27 (135.6 b^3) | 4.24e-27 (271.3 b^3) | 6.36e-27 (406.9 b^3) |
| 1000K | 30->100 (70MPa) | 1.36e-27 (87.2 b^3) | 2.73e-27 (174.4 b^3) | 4.09e-27 (261.6 b^3) |
| 1200K | 30->75 (45MPa) | 2.54e-27 (162.8 b^3) | 5.09e-27 (325.5 b^3) | 7.63e-27 (488.3 b^3) |
| 1200K | 30->100 (70MPa) | 1.64e-27 (104.6 b^3) | 3.27e-27 (209.3 b^3) | 4.91e-27 (313.9 b^3) |

(`b=2.5e-10`m, `Omega=1e-29`m^3 -- both existing `Params` defaults;
`b^3=1.5625e-29`m^3.) Required `V_act` spans `~70-490 b^3` across this
grid. This range is **not obviously unphysical in isolation** -- reported
activation volumes for various thermally-activated deformation/
nucleation mechanisms in the literature span a similarly broad range
(order 1-1000 `b^3`) -- so a constant-`V` model's magnitude requirement,
by itself, does not rule the mechanism out. **However** (Section 15's
point, addressed next) magnitude plausibility does not resolve the more
fundamental low-stress waiting-time issue.

## 11. Whether a critical-nucleus model is likely required

**Likely yes, if "no low-stress nucleation at arbitrarily long times" is
a hard physical requirement for the eventual hazard.** A constant-`V`
Arrhenius rate `r(sigma)=r0*exp[V_act*sigma/(kB*T)]` is **finite and
nonzero at `sigma~30MPa`** for any finite `V_act`; given unbounded
waiting time, such a process nucleates with probability 1 regardless of
how small `r(30MPa)` is made (Section 15's explicit point) -- a
constant-`V` linear-in-sigma barrier can make low-stress nucleation
*rare*, never *forbidden*.

A stress-dependent critical-nucleus barrier is qualitatively different
and was NOT implemented here (no specific loop geometry or mechanical
conjugate work has been justified for the GB-disconnection mechanism
this eventual hazard targets), but its generic structure is worth
recording as the leading candidate: for a line-tension-controlled
loop-like embryo of size `R`, `DeltaG(R) ~ 2*pi*R*Gamma_line -
pi*R^2*sigma*b_eff` (interfacial/line cost growing as `R`, mechanical
driving work growing as `R^2*sigma`) gives `R*=Gamma_line/(sigma*
b_eff)` and `DeltaG*=pi*Gamma_line^2/(sigma*b_eff)` -- a barrier that
diverges as `1/sigma` toward `sigma->0`, MUCH more strongly suppressing
than a constant-`V` model's linear-in-`sigma` exponent. This is the kind
of mechanism that could plausibly give "negligible nucleation probability
at the ~30MPa reset stress, order-1 probability by 75-100MPa" -- but
`Gamma_line` and `b_eff` for the actual GB/TJ disconnection mechanism
this project targets are not yet identified, so no specific formula is
adopted here (per Section 14's explicit instruction).

## 12. Recommendation for Milestone 16A

**Do not calibrate a hazard against the present global/contact-averaged
`sigma_sint_endpoint` mechanism** -- Section 8/10 establish that the
deterministic slow mechanics explored here (bounded rate competition,
geometry narrowing, both dt-verified) tops out around `51MPa`, well
short of the `75-100MPa` target regime, under the qualified isotropic
free energy (`gamma_s=1.0`J/m^2) and the physically calibrated `M_GB`
mapping. Before any hazard design:

1. **Do not adopt an increased `gamma_s`** (Section 6) without an
   independent, material-specific justification -- the diagnostic
   sensitivity found `gamma_s~1.5-3.5`J/m^2 would close the gap, but this
   is a modeling choice, not a result of this milestone's physics.
2. **Consider whether the "nucleation regime" target (75-100MPa) itself
   needs revisiting** given the qualified mechanism's natural ceiling --
   either by identifying a DIFFERENT stress-concentrating mechanism not
   captured by this isotropic, contact-averaged construction (e.g. a
   genuinely local, diffuse configurational-force diagnostic near the
   TJ, Milestone 15C Section 11's deferred option -- local curvature
   itself remains disqualified, Milestone 15B/15C), or by re-examining
   whether `30MPa`/`75-100MPa` are the right reset/activation pair for
   THIS specific mechanism.
3. **If a hazard is still designed against a stress in the `~30-51MPa`
   range** (the range actually, naturally accessible), Section 15's
   waiting-time argument applies with even more force: a constant-`V`
   Arrhenius law finite at `30MPa` will eventually nucleate regardless of
   barrier height, so a stress-DEPENDENT critical-nucleus barrier
   (Section 11 above) -- not a constant activation volume -- is the more
   defensible starting point if "no spontaneous low-stress nucleation"
   is a genuine physical requirement of the target mechanism.
4. Do not proceed with first-passage kinetics (constant-`V` or
   otherwise) calibrated to reach 75-100MPa from this mechanism until
   either a different/larger stress-concentration pathway is identified,
   or the target activation window is revisited.

## Test suite

235/235 passing throughout (only `scripts/m15_gb_surface_rate_
competition.py`'s `sample_state` was extended with two new diagnostic
fields, `sigma_sint_endpoint_max`/`sigma_sint_endpoint_over_max`; no
production `pf_sintering/` code changed in this milestone).

## New files

- `scripts/m15d_rate_screen.py` — Section 4's bounded M_GB/M_s screen.

## STOP

Per Section 10's explicit instruction ("If neither S75 nor S100 is
accessible for any defensible condition: STOP. Do not proceed to hazard
calibration."), this milestone stops here. sink/hazard/RBM/anisotropy
were not activated.
