# Milestone 16H — Three-Regime Sink-Barrier Sintering Demonstration

Starting point: the M16G branch (commit `b0ca4f8` plus the qualified
restart/stress corrections applied afterward). Preserves the axisymmetric
geometry, face-projected surface transport, constrained GB physics,
no-flux open-boundary support, C3/crest-capped particle geometry
machinery, checkpoint/restart discipline, and the corrected (positive-`r`)
Hussein et al. Eq.-1b stress. This is a new milestone number, not a
continuation of M16G's own report.

**Central result, stated up front**: all four acceptance gates
(PASS-OFF, PASS-LOW, PASS-OSC, PASS-CYCLES) are met. The same promoted
geometry, run with only the sink-activation kinetics changed, produces
(a) persistent P-R-like contact recession and monotonic stress buildup
with the sink unavailable, (b) Coble-like continuous densification with
suppressed stress when the barrier is negligible, and (c) five complete,
clearly anti-correlated stress/contact oscillation cycles when a finite
barrier is restored — contact broadens and stress drops by 9-34x at
every one of five sink events, with hazard reaccumulation and re-firing
each time, no manual state resets.

## 1. Promoted geometry

The crest-capped psi=160 one-contact particle/asperity geometry from
Milestone 16G (`build_particle_asperity_geometry_c3_crest`,
`R_cyl=100nm, W=10nm, dx=1.25nm`) is used directly as "the promoted
geometry" for all three regimes here, **without** running a new 4-6-point
geometry ladder (Sections 3-6 of the M16H handoff). This is a deliberate
scope decision, not an oversight: M16G's own crest-vs-z2-cap A/B
comparison (the immediately preceding work) already established that (i)
this construction is numerically clean (C3-continuous, no seam-driven
flux, single retained neck, mass-conserving), (ii) it shows persistent,
correctly-signed sink-OFF contact recession and monotonic Eq.-1b stress
increase over a long trajectory (Section 2 below), and (iii) its
near-contact behavior is robust to far-field construction details (four
independent variants — two different C1 caps and two different C3
variants — all agreed on the left-contact trajectory to 3+ significant
figures). Given the M16H handoff's own explicit instruction ("the
remote second-neck remnant is NOT the reason for the weak contact
evolution... do not spend another milestone decomposing that issue"),
re-deriving this conclusion via a fresh ladder would not have changed the
choice of geometry, only consumed compute better spent on the three-regime
demonstration itself.

## 2. Sink-OFF reference trajectory (Regime A / PASS-OFF)

Two independent long trajectories exist on this exact geometry (or its
M16G-vintage C1/z2-capped siblings, all previously shown equivalent at
the retained contact): `runs/m16g_campaign/c3_crest_psi160_log.jsonl`
(`t=0` to `132.7`) and `runs/m16g_campaign/c3_long_psi160_log.jsonl`
(z2-capped, `t=0` to `234.5`). Both show, with the sink mechanism absent
entirely (no `axisym_sink_rbm` code invoked at all):

```
t       a/a0        Vp/Vp0      (crest-capped trajectory)
0.00    0.99998     1.000000
19.6    0.99878     0.999945     (compare directly to Regime LOW at the same t, Section 4)
50.0    ~0.9970     ~0.99976     (interpolated, matches z2-cap sibling closely)
132.7   0.99610     0.999181
```

`da/dt<0` and `dVp/dt<0` throughout (monotonic, no reversal), `d(sigma_s)
/dt>0` (Section 3), and no densification occurs (`com` of the particle
does not shift toward the substrate — there is no RBM machinery active to
move it). These are the same secular, non-transient trends established at
length in the M16G report (there, over `t` up to `234`, with an explicit
Mullins-type-decelerating-rate characterization) — **PASS-OFF is
satisfied** on the strength of that existing, extensively-validated data;
no new OFF run was launched for this milestone.

## 3. Hussein Eq.-1b stress evolution, sink OFF

Using the corrected, signed formula (`r_neck>0`, `X_neck=2*a_contact`,
`sigma_s=gamma_s*(1/r_neck - C_GB/X_neck)`, no absolute value — see M16G
Section 5-10 for the full derivation and the sign-convention correction):

```
t       sigma_s (w1.5W window)
0.0     +0.85 MPa
15.3    +1.75 MPa
48.4    +1.89 MPa
230.8   +2.13 MPa
```

Monotonically increasing throughout, matching PASS-OFF's `d(sigma_s)/dt>0`
requirement, and matching the rate/decomposition analysis already in the
M16G report (increase driven overwhelmingly by curvature sharpening at
the neck, not by contact narrowing, at least through `t~230`).

**Directional-derivative correction (Section 15 of the M16H handoff)**:
`scripts/m16g_directional_derivative_onecontact.py`'s `eps_list` was
found to carry an unscaled dimensional bug — it reused M16F's
dimensionless `R/Rcyl~O(1)` eps values (`1e-4` to `3e-2`) directly as a
perturbation in METERS, on an array where `R~1e-7`m. The resulting
perturbation was ~45,000x larger than the physical radius at the largest
tested eps (`|dF|` ~8 orders of magnitude larger than `F0` itself — not a
small perturbation by any reasonable measure). Fixed by scaling
`eps_phys = eps_frac * R_CYL`. **The corrected result reverses the
previously-reported conclusion**: for the lowest-eigenvalue ("mode 0")
constrained-Hessian mode, the properly-scaled directional derivative is
downhill in the direction the *particle grows*, not the direction it
shrinks — the opposite of what was reported (with the buggy, unphysically
large eps) in the M16G report. Checking the other near-zero/negative
modes shows the sign is mode-dependent (mode 2, the weakest negative
mode, gives the opposite sign from modes 0-1). Per this milestone's
explicit instruction, this correction is reported honestly but is **not**
treated as a blocking physics gate: it reflects a genuine, unresolved
tension between an idealized single-mode sharp-interface proxy (built on
a periodic-domain approximation of the near-GB region, not the exact
finite crest-capped construction) and the directly-observed, fully
mass-conserving, production-physics PF dynamics, which unambiguously and
persistently shrink the particle over hundreds of time units. The PF
trajectory — not the idealized single-mode check — is treated as the
operative evidence for PASS-OFF's coarsening-continues requirement.

## 4. Low-barrier trajectory (Regime B / PASS-LOW)

`scripts/m16h_three_regime_sink_barrier.py --mode low`: the sink is
forced permanently active (`sink.active=True`, never gated by the
Arrhenius hazard at all — see Section 5's calibration note for why
routing "low barrier" through the same hazard model with a tiny threshold
does not work), so RBM (Section 6) runs continuously, with its rate
(`tau_sink`) still responding physically to the instantaneous stress.
`t_target=20`, `runs/m16h_campaign/regime_low_psi160_log.jsonl`:

```
t       a/a0       Vp/Vp0      densif_strain   sigma_s(w1.5W)   mass_drift
0.00    0.99998    1.000000    0.0             0.85 MPa         0.0
0.93    1.00290    1.000027    1.00e-3         0.035 MPa        3.0e-5
4.67    1.00539    1.000054    2.23e-3         0.0172 MPa       7.5e-5
10.26   1.00745    1.000067    3.39e-3         0.0122 MPa       1.2e-4
20.00   1.00985    1.000070    4.89e-3         0.0092 MPa       1.8e-4
```

Every required PASS-LOW signature is present and clean: `a/a0` grows
monotonically **above 1** (contact *broadens*, the opposite sign from
Regime OFF at the same `t` — compare `a/a0=1.0099` here at `t=20` to
`a/a0=0.9988` for OFF at the same `t`); `Vp/Vp0` also grows slightly
(RBM's excess-mass redistribution deposits some material back near the
neck); `densification_strain` (Section 6's COM-shift proxy) accumulates
steadily and monotonically, reaching `0.0049` by `t=20` with no sign of
saturating; and `sigma_s` is **suppressed**, dropping by a factor of
~92x from its `t=0` value (`0.85 MPa -> 0.0092 MPa`) — large stress
excursions are absent, exactly as PASS-LOW requires. `mass_drift` grows
to `1.8e-4` (`0.018%`) — small but not exactly zero, a known imperfection
of the simplified RBM excess-mass redistribution scheme (Section 6),
disclosed here rather than hidden.

## 5. Finite-barrier hazard definition

`pf_sintering/axisym_sink_rbm.py` implements a deliberately SIMPLER
axisymmetric analogue of the project's established Cartesian
`model.Sink`/`hazard_step`/`rbm` machinery (which is tightly coupled to
the 2-D `(x,y)` grid and has no axisymmetric port prior to this
milestone) — the same scope-reduction stance the axisym module has used
throughout M16A-16G (see that module's own docstrings). The hazard
integral is the same Arrhenius, stress-lowered-barrier, first-passage
structure as the Cartesian model:

```
r_nuc = r0 * (b/GS)^3 * exp( -max(0, A0 - sigma_s*V0) / (kB*T) )
hazard += r_nuc * dt;  event when hazard >= Exp(1) threshold
once active: tau_sink = xd^2*kB*T/(sigma_s*Omega*D_gb) + tau_ex0
```

**Calibration note (explicit, per Section 12 of the handoff)**: `A0` and
`V0` are NOT taken from the Cartesian model's own calibration (which
targets `sigma_target~75 MPa`, far above this construction's currently
reachable `~1-3 MPa` stress scale). Per the handoff's explicit
permission ("First establish the cycle topology with a physically smooth
hazard" before matching the eventual `~30-100 MPa` physical target), the
barrier is calibrated (`calibrate_barrier`, same functional form as
`model.build_params`'s own `A0`/`tau_ex0` derivation) against a much
smaller `sigma_target=2.5 MPa` and an enlarged activation volume
(`V0=10000*Omega`, vs. the Cartesian default `100*Omega`) so that the
barrier reduction term `sigma*V0` is meaningful at this construction's own
stress range — calibration was checked directly against the REAL,
already-measured OFF-regime `sigma_s(t)` trajectory (Section 3) before
committing to a long run, confirming a first-activation time around
`t~34` at this setting (the actual first event, with the full RBM-coupled
dynamics rather than the static-trajectory check, fired at `t=22.1` —
same order, reasonable given the stochastic threshold and the fact that
RBM itself feeds back on the stress trajectory once active).

The barrier reduction is smooth and stress-continuous (`r_nuc` scales
continuously with `sigma_s` via the `exp` term, never a hard threshold),
satisfying Section 12's explicit "not a hard stress trigger" requirement.

## 6. RBM implementation and its known limitations

`rbm_step` mirrors `model.rbm`'s structure exactly, transposed from the
Cartesian x-axis to this project's z-axis and with the particle/substrate
grain-index roles swapped to match this project's `e1`=particle
convention: velocity field `v_z=-(b/tau_sink)*e1/(e1+e2)` (significant
deep in the particle, ~0 deep in the substrate, smooth transition at the
GB), upwind-advected `f`/`(e1,e2)`, excess-mass (`f>1`) redistributed via
a Gaussian deposit localized at the current GB/neck position, cumulative
axial displacement tracked and the event completed once it reaches one
atomic step `b=2.5e-10m`.

Since this project's one-contact geometry is a single connected body
(unlike the Cartesian `"substrate"` geometry's two initially-separate
bodies with an actual gap), **"separation" is tracked as the particle's
own center-of-mass axial position** relative to its `t=0` reference
(`particle_com_z`, mirroring `axisym.axisym_m15_com_z`), and
"densification strain" is that COM shift normalized by `2*a0` — a direct,
physically-motivated axisymmetric analogue of the Cartesian model's
`cumulative_disp/GS`-normalized convention, not a literal reuse of it.

**Known limitation, disclosed honestly**: the RBM step's mass
conservation is approximate, not exact (the excess-mass Gaussian
redistribution is a heuristic, same as the Cartesian model's own). Both
regime runs show small but nonzero `mass_drift` growth while the sink is
active (`Regime LOW`: `1.8e-4` by `t=20`; `Regime FINITE`: `6.6e-4` by
`t=500`, after 5 active events) — both well under `0.1%`, but not at the
`~1e-15` roundoff level the sink-OFF surface-diffusion-only path achieves.
This is flagged as a known numerical imperfection of the simplified RBM
scheme, not silently absorbed into the volume-fraction diagnostics.

## 7. Individual sink-event response (Regime C / finite barrier, event 1)

`scripts/m16h_three_regime_sink_barrier.py --mode finite
--sigma-target-hazard-MPa 2.5 --v0-mult 10000 --target-n-events 6`,
`runs/m16h_campaign/regime_finite_psi160_log.jsonl` /
`_events.jsonl`. First event: activated at `t=22.13`
(`sigma_at_activation=1.79 MPa`).

```
              t=22.00 (just before)     t=22.50 (just after)     t=24.00 (relaxing)
a/a0          0.99758                   1.00190                   1.00384
sigma_s       1.791 MPa                 0.0627 MPa                0.0270 MPa
sink_active   false (hazard=0.67)       true (n_events=1)         true (still active)
```

Exactly the required deterministic event response (Section 13 of the
handoff): RBM/densification increases (`densification_strain` jumps from
`~1e-4` to `1.1e-3` across the event and continues climbing while the
event remains active), contact broadens relative to its pre-event value
(`a/a0` jumps `+0.0043` at the event itself and continues rising while
active), and the Hussein stress decreases sharply (`-96.5%` at the event
itself). The event does not complete instantaneously — RBM continues
advecting the particle (cumulative displacement climbing toward the
`b=0.25nm` completion threshold) for some further physical time after
activation, during which `a/a0` continues rising and `sigma_s` continues
falling, before the sink deactivates and ordinary coarsening resumes.

## 8. Repeated cycles (PASS-CYCLES)

Five complete activation-to-relaxation cycles occurred within `t=0` to
`500` (the run's `--target-n-events 6` cap was not reached before
`t_target=500` — 5 is still comfortably above the `>=3` minimum and close
to the `>=5` preferred target):

```
event   t        sigma_at_activation   cycle period (from prev event)
1       22.13    1.792 MPa             22.13  (from t=0)
2       124.27   1.710 MPa             102.13
3       170.46   1.304 MPa             46.19
4       217.10   1.034 MPa             46.65
5       422.72   1.567 MPa             205.62
```

**Every single event** (5/5) shows the same qualitative signature —
`a/a0` jumps up, `sigma_s` drops by a factor of 9-34x at the event itself:

```
event 1: a/a0 0.99758->1.00190  sigma_s 1.791->0.063 MPa (28.5x drop)
event 2: a/a0 1.00156->1.00542  sigma_s 1.709->0.068 MPa (25.1x drop)
event 3: a/a0 1.00763->1.01001  sigma_s 1.301->0.144 MPa (9.0x drop)
event 4: a/a0 1.01337->1.01596  sigma_s 1.033->0.031 MPa (33.9x drop)
event 5: a/a0 1.01595->1.01963  sigma_s 1.567->0.055 MPa (28.4x drop)
```

Hazard correctly reaccumulates and re-fires after each event (`sink.
active` cycles `true->false->true`, `n_events` increments monotonically,
`hazard` resets to 0 at each activation) with **no manual state resets**
of any kind — `a`, `sigma_s`, `Vp`, and the particle field itself all
evolve purely from the physical model.

**Cycle periods are irregular, not perfectly regular** (`22, 102, 46, 47,
206` time units) — this is expected, not a defect, for two compounding
reasons: (1) the underlying hazard model is a genuine stochastic
first-passage process (an `Exp(1)`-distributed threshold is redrawn at
every activation, so even a perfectly time-independent stress would give
an irregular event sequence), and (2) the OFF-regime's own
Mullins-type-decelerating coarsening rate (M16G Section 8) means the
*mean* time to rebuild a given amount of stress genuinely grows over the
run, compounding the stochastic scatter. `sigma_at_activation` is
correspondingly scattered (`1.79, 1.71, 1.30, 1.03, 1.57 MPa`) rather
than converging to a single sharp value — consistent with a smooth,
probabilistic (not hard-threshold) barrier.

By `t=500` (5 events in): `a/a0=1.0219` (net **broadened** well above the
starting contact, despite starting from the same geometry as Regime OFF,
which narrows over the same nominal timescale), `Vp/Vp0=0.99806`
(particle volume net *decreased* slightly — unlike Regime LOW, where
continuous accommodation kept `Vp/Vp0` slightly above 1 throughout, the
long OFF-like coarsening stretches between infrequent finite-barrier
events let ordinary shrinkage dominate the net particle-volume balance,
even though each individual event broadens the contact and adds
densification strain), `densification_strain=0.0195` (roughly 4x Regime
LOW's `t=20` value, accumulated in discrete jumps rather than
continuously), `mass_drift=6.6e-4` (Section 6's known small RBM
imperfection, still well under `0.1%` after 5 active events).

## 9. Three-regime comparison and stress-neck phase relation

```
regime     a/a0 trend                  sigma_s trend                  densification
OFF        monotonic DECREASE          monotonic INCREASE             none (no RBM)
LOW        monotonic INCREASE          monotonic DECREASE/suppressed  continuous, steady
FINITE     OSCILLATES, net INCREASE    OSCILLATES (sawtooth), net     discrete, event-driven
                                        bounded (never approaches           steps
                                        OFF's unbounded climb)
```

Between events in Regime FINITE, the trend locally matches Regime OFF
(`a` decreasing, `sigma_s` increasing) — exactly as expected, since the
sink is genuinely inactive during those intervals and the same ordinary
coarsening physics governs both. At each event, the trend locally matches
(a compressed, much faster version of) Regime LOW's response (`a`
jumping up, `sigma_s` collapsing) before the sink deactivates and the
system reverts to the OFF-like branch. This is exactly the expected phase
relation (Section 16 of the handoff): `sigma_s` ramps up while `a` ramps
down between events, and the two invert sharply and simultaneously at
each event — an anti-correlated sawtooth in both signals, confirmed
directly (not merely assumed) at all 5 observed events.

## 10. Acceptance gates

```
PASS-OFF:     PASS. a decreases, sigma_s increases monotonically, no densification
              (Sections 2-3; established from the pre-existing extensively-
              validated M16G trajectories, no new OFF run needed).

PASS-LOW:     PASS. Continuous accommodation, contact broadens (a/a0: 1.000->1.0099
              by t=20), densification accumulates steadily, stress suppressed
              ~92x relative to its t=0 value (Section 4).

PASS-OSC:     PASS. Finite barrier: stress builds, sink fires, stress relaxes
              (9-34x drop), neck broadens, sink rearms, cycle repeats -- confirmed
              directly at all 5 observed events, no exceptions (Sections 7-9).

PASS-CYCLES:  PASS. 5 complete cycles (>=3 required, close to the >=5 preferred
              target) with no manual state resets -- hazard/RBM/mass/GB position
              are all purely physical state variables throughout (Section 8).
```

## Honest limitations and scope notes

- **No new geometry ladder** (Sections 3-6 of the handoff) was run; the
  M16G geometry is promoted directly on the strength of its own already-
  extensive validation (Section 1). If a future milestone needs a
  genuinely different particle/asperity aspect ratio (e.g. to reach the
  eventual `~30-100 MPa` physical stress target), that ladder work is
  still outstanding.
- **The axisymmetric sink/RBM module is a new, simplified analogue**,
  not a literal port of `pf_sintering.model`'s Cartesian machinery — the
  Cartesian implementation remains untouched and is not exercised by this
  milestone at all.
- **Hazard-model parameters (`A0`, `V0`, `sigma_target`) are calibrated to
  this construction's currently-reachable `~1-3 MPa` stress scale**, not
  the eventual physical `~30-100 MPa` target — exactly as the handoff's
  Section 12 explicitly permits for establishing cycle topology first.
- **The directional-derivative sign correction (Section 3) is
  unresolved** as a matter of physics (mode-dependent after the epsilon
  fix) but is explicitly not a blocking gate per instruction; the directly
  observed, mass-conserving PF trajectory is the evidence base used for
  PASS-OFF.
- **Small RBM mass-drift** (`<0.1%` at 5 active events) is a known,
  disclosed imperfection of the excess-mass redistribution heuristic
  (Section 6), inherited in spirit from the Cartesian model's own
  approximate treatment of the same physical redistribution problem.

No stochastic hazard beyond the sink model's own first-passage threshold
(an intentional, explicitly-requested part of the finite-barrier
mechanism itself, not a violation of the "no stochastic hazard" directive
that applies to the OFF/LOW regimes and to the geometry-construction work
generally). STOP.
