# Periodic sinusoidal substrate geometry

Scope: test whether giving the substrate ("grain 1") an explicit, evolving,
curved free surface — rather than the flat semi-infinite half-plane audited
in Milestone 9 — removes the geometric ambiguity that made an
external-reservoir Ostwald closure necessary. New geometry
(`geometry="sinusoidal_substrate"`) added without touching the existing
`substrate`/`threeparticle` initializers. **Stopping here before activating
hazard/sink, as instructed.**

## 0. Headline result

**Outcome P1, with an important resolution caveat.** At the primary grid
(dx=5nm), giving the substrate explicit sinusoidal curvature and testing
three trajectories from the same initial state — S0 (no coarsening), S1
(coarsening, addition diverted to an external reservoir), S2 (coarsening,
addition redeposited locally onto the curved substrate, i.e. unmodified
production `ostwald_substrate`) — gives **`delta_L_coarsening` negative
(contact shrinks relative to the no-coarsening control) for BOTH S1
(`-0.0122nm`) and S2 (`-0.0035nm`)** at `|dV2|/V20=3e-4`. This is a
qualitative reversal from every flat-substrate result in Milestones 7-9,
where the equivalent of S2 (full local redeposition) was *always* positive
(widening).

**However, the fixed-physics grid check (Section 14) shows S2's result is
not resolution-robust**: at dx=2.5nm, S2 flips sign to `+0.0011nm`
(widening again), while **S1 keeps its sign and stays close in magnitude**
(`-0.0098nm`, 80% of the dx=5nm value). The clean, monotonic,
near-perfectly-linear coarsening-rate dependence of S1's effect
(`-0.00403`, `-0.00404`, `-0.00407` nm per unit rate at 0.3x/1x/3x) is
additional evidence that S1's result is a genuine, well-resolved physical
signal, not noise. **The external-reservoir closure (Milestone 9's Model B)
remains the robust finding on this improved geometry; adding substrate
curvature alone, without also correcting the closure, is not by itself a
reliable fix** — S2's small-magnitude, sign-fragile result should not be
read as confirming that "an explicit receiving surface is sufficient."

## 1. New geometry

`ModelConfig(geometry="sinusoidal_substrate")`, three new physical (never
dx-tied) fields: `sinusoid_wavelength`, `sinusoid_amplitude` (meters),
`sinusoid_phase` (radians). Substrate free surface:

```
x_s(Y) = wall_mean + amplitude * cos(2*pi*Y/wavelength + phase)
e1 = 0.5*(1 - tanh((X - x_s(Y))/W))
```

using the same centered `Y` convention every other geometry uses (`Y=0` at
the domain's vertical center row). `f` and the ownership partition
(`t1`,`t2`) use the *same* `x_s(Y)`, not a flat wall — `f`'s `0.5` contour
therefore follows the sinusoid by construction, not just `e1`'s (Section 2
requirement; verified in `test_f_contour_follows_the_imposed_sinusoid_away_from_the_particle`).
`phase=0` puts a crest (`x_s` maximal, protruding toward +X) at `Y=0`; the
particle is centered there (`cx = x_s(0) + Rx - initial_overlap`,
`test_particle_centered_at_the_crest`). The existing `substrate` and
`threeparticle` branches of `initialize_fields`/`build_params` are
untouched (`test_existing_geometries_unaffected`).

## 2. Periodic-boundary audit

Read directly from `lap9`/`grad`/`div`/`evolve_f` (not assumed):

| direction | axis | treatment |
|---|---|---|
| **X** (columns) | `axis=1` | **Periodic** — every stencil (`lap9`'s E/W, `grad`'s `gx`, `div`, `evolve_f`'s `Jx`) uses `np.roll(...,1)` |
| **Y** (rows) | `axis=0` | **No-flux/reflecting**, not periodic — `lap9`'s N/S edge-replicate (`np.vstack([a[:1],a[:-1]])`), `grad`'s `gy` uses one-sided differences at rows 0/-1, `evolve_f`'s `Jy` is explicitly zero at the domain's top/bottom faces |

The handoff's own default parameterization (`x_s(Y)`, oscillating with `Y`)
therefore does **not** sit on the model's actual periodic axis — X is
periodic, Y is not. Rather than rotate the entire geometry (which would
require re-deriving every existing diagnostic's row/column conventions —
`tj_force.locate_neck_tjs`'s "neck column" search, `compute_stress`'s
`cr=Ny//2` row, `contact_geometry`, `tj_subgrid` — all hardwired to the
current X=across/Y=along convention), this milestone uses the escape clause
explicitly offered ("orient the sinusoid along the existing periodic
direction... if necessary") via a **reflection-symmetry equivalence**
instead: `Ny*dx` is snapped to *exactly one full wavelength*
(`build_params`, `wavelength=ny*dx` after computing `ny=round(wavelength/dx)`),
with the crest centered at `Y=0` so the domain's top and bottom edges both
land exactly on sinusoid **troughs** — extrema, i.e. symmetry planes of an
infinite periodic cosine. A true periodic solution that remains symmetric
about a trough has zero flux across that plane by symmetry; the existing
no-flux/reflecting Y-boundary imposes exactly that condition. **The
existing boundary treatment is therefore physically equivalent to true
periodicity at this specific geometry, for as long as the solution stays
symmetric about the domain edges** — not "periodic" in the literal sense of
wrapping, but equivalent in effect, verified by construction
(`test_domain_height_matches_exactly_one_wavelength`). This equivalence
would degrade if the solution ever developed genuine up-down asymmetry
between the two troughs (not observed at the short `|dV2|/V20<=3e-4`
horizons used here — Section 11's `B1` Fourier coefficient, which measures
exactly this kind of asymmetry, stays at `~1e-14` vs. the fundamental's
`~2.4e-8`, i.e. `~6` orders of magnitude smaller, over the full trajectory).

## 3. Initial geometry parameters

`wavelength = 6*R2 = 480nm` (several times `R2=80nm`, per the handoff's
suggested design), `amplitude = 0.05*wavelength = 24nm` (chosen after an
initial default of `0.15*wavelength=72nm` — comparable to `R2` itself —
was judged not "gentle"; `24nm` gives a substrate radius of curvature
`~243nm`, gentler than the particle's own `80nm` curvature scale). `W=20nm`
fixed physical (`interface_width_override`), `4` cells/interface at
dx=5nm, `8` at dx=2.5nm. Domain: `Nx=126, Ny=96` at dx=5nm
(`Ny*dx=480nm=1` wavelength exactly, `Nx*dx=630nm` giving ample margin for
the amplitude plus the usual particle/vapor/bulk-substrate room). Particle:
same `R2=80nm`, `aspect_ratio=2`, `short_plane`, `initial_overlap=20nm` as
every prior milestone's primary case, now measured relative to the crest
position rather than a flat wall.

## 4. Initialization validation

| quantity | value |
|---|---:|
| `sigma` (legacy) | `37.5456MPa` |
| `x_neck` (legacy) | `55.00nm` |
| `psi` (legacy, measured-dihedral) | `93.87deg` |
| `L_contact_TJ_sub` | `64.95nm` |
| `L_GB_geom_sub` | `64.97nm` |
| `psi_top`/`psi_bottom` (tj_force) | `78.14deg` / `78.14deg` (symmetric, as expected for a symmetric initial condition) |
| `kappa_top`/`kappa_bottom` (signed curvature) | `-3.438e7` / `-3.438e7` m⁻¹ (symmetric) |
| `F_TJ_mag_top`/`bottom` | `1.140` / `0.463` (top/bottom asymmetric — both TJs are at the same `Y`-distance from the crest by construction, but the force-balance residual is not required to be symmetric even when the geometry is, since `xi_gb`'s sign convention differs top vs. bottom) |
| `V1` | `6.725e-14 m^2` |
| `V2` | `2.005e-14 m^2` |

`compute_subgrid_contact` and `compute_neck_tj_forces` both resolve cleanly
(`sub.resolved=True`, `n_resolved=2`) — the sinusoidal free surface joins
the contact smoothly enough that the TJ locator does not rely on any grid
defect, confirmed directly rather than assumed.

## 5. C0/S0/S1/S2 comparison at `|dV2|/V20=3e-4` (n_steps=276, primary grid)

| | S0 (no coarsening) | S1 (→ external reservoir) | S2 (→ local sinusoidal receiver) |
|---|---:|---:|---:|
| `L_contact_TJ_sub` start → end | `64.95nm` → `67.27nm` | `64.95nm` → `67.26nm` | `64.95nm` → `67.27nm` |
| `dL_contact_TJ_sub` (absolute) | `+2.316nm` | `+2.304nm` | `+2.313nm` |
| **`delta_L_coarsening` (vs. S0)** | — | **`-0.0122nm`** | **`-0.0035nm`** |
| `sigma` end | `37.533MPa` | `37.532MPa` | `37.535MPa` |
| `psi_top`/`bottom` end | `77.12`/`76.94deg` | `77.13`/`76.96deg` | `77.12`/`76.95deg` |
| `kappa_top`/`bottom` end (1/m) | `-3.424e7`/`-3.429e7` | `-3.423e7`/`-3.429e7` | `-3.424e7`/`-3.429e7` |
| `F_TJ_mag_top`/`bottom` end | `1.1247`/`0.4895` | `1.1250`/`0.4894` | `1.1249`/`0.4897` |
| `V2` end | `2.00467e-14` | `2.00407e-14` | `2.00407e-14` |
| `M_external_reservoir` end | 0 | `6.026e-18 m^2` | 0 |

The absolute `L_contact_TJ_sub` trajectory widens substantially under
capillary relaxation alone in all three cases (`+2.3nm`, dominated by the
same non-equilibrated-initial-geometry background relaxation Milestone 7
subtracted out for the flat case) — the differential quantities
(`delta_L_coarsening`, two orders of magnitude smaller) are what isolate
the coarsening-specific effect, exactly as in every prior milestone.
`sigma`/`psi`/`kappa`/`F_TJ` differ only in the 4th-5th significant figure
between S0/S1/S2 at this horizon — real but small, consistent with the
small `delta_L_coarsening` magnitudes themselves.

## 6. Sinusoid Fourier/amplitude evolution (S1 trajectory)

Fit `x_s(Y) = x_mean + sum_k[A_k*cos(k*2*pi*Y/lambda) + B_k*sin(...)]` to
the `f=0.5` crossing of every row with `|Y| > R_y + 3W = 116.6nm` (excluding
rows the particle itself occupies):

| | start | end (S1, 276 steps) | change |
|---|---:|---:|---:|
| `x_mean` | `-162.34nm` | `-162.29nm` | `+0.043nm` |
| fundamental amplitude (`amp1`) | `24.111nm` | `24.219nm` | `+0.108nm` (`+0.45%`) |
| 2nd harmonic (`amp2`) | `0.0615nm` | `0.1231nm` | small, doubled but still `<0.6%` of fundamental |
| 3rd harmonic (`amp3`) | `0.0221nm` | `0.0422nm` | small, similarly still `<0.2%` of fundamental |
| `B1` (asymmetry) | `~0` (`-2.6e-24`) | `-4.98e-14` | `~6` orders of magnitude below `A1` — negligible |

The fundamental amplitude grows very slightly (`<0.5%`) over the trajectory
rather than decaying — a mild, small-magnitude finding not strongly
interpreted here (surface-diffusion smoothing of an isolated periodic
profile would ordinarily decay a Fourier mode; the small growth observed
may reflect the free surface's coupling to the particle/GB system rather
than isolated relaxation, or may be within the fit's own noise floor given
the modest number of eligible rows). The near-total absence of `B1`/odd-
harmonic growth confirms the reflection-symmetry equivalence (Section 2)
remains intact throughout this trajectory.

## 7. RBM/strain distinction

`Sink(threshold=math.inf)` used throughout; `hazard_step`/`rbm` never
called in S0, S1, or S2 (structurally, not merely non-firing) — 
`cumulative_disp`/`cumulative_strain` are `0.0` by construction in every
trajectory. The observed particle-centroid/mean-substrate-position drift
(Section 6, `x_mean` change `+0.043nm`) is shape/mass redistribution under
capillary relaxation and Ostwald exchange, not rigid-body translation —
distinguished here explicitly rather than assumed, per the handoff's
caution.

## 8. Coarsening-rate series (S1 closure)

Same `N_ref=276` steps (matched to the primary `3e-4` target), same shared
S0 reference:

| `coarsening_rate_scale` | `dV2/V20` | `delta_L_coarsening` (vs. S0, nm) | ratio to rate |
|---:|---:|---:|---:|
| 0.3 | -3.007e-05 | **-0.001209** | -0.00403/unit |
| 1.0 | -1.002e-04 | **-0.004044** | -0.00404/unit |
| 3.0 | -3.006e-04 | **-0.012218** | -0.00407/unit |

**Monotonically more negative with rate, and strikingly close to linear**
(ratio constant to within 1% across a 10x range) — the contact-recession
tendency strengthens as coarsening outruns capillary reshaping, exactly the
directional signature the handoff asked to test for, with the same clean
linear-in-rate character Milestone 7 found for the flat substrate's
(wrong-sign) widening effect.

## 9. Fixed-physics grid check

`interface_width_override=20nm` fixed, `eta_diffusivity_fixed_physical=True`
(Milestone 9's corrected reference), `wavelength`/`amplitude`/`R2`/`overlap`
held fixed in physical units, independently dt-checked at each grid (dx=5nm
natural, dx=2.5nm at dt/4, matching every prior milestone's converged
choice):

| dx | `delta_L_coarsening` S1 (vs. S0) | `delta_L_coarsening` S2 (vs. S0) |
|---:|---:|---:|
| 5nm | **-0.012218nm** | **-0.003470nm** |
| 2.5nm | **-0.009800nm** (80% of dx=5nm, same sign) | **+0.001123nm** (**sign flip**) |

**S1 is sign-stable and reasonably magnitude-stable under mesh refinement.
S2 is not** — its dx=5nm result, while nominally matching S1's sign, does
not survive refinement. Given S2's magnitude was already ~3.5x smaller than
S1's at the coarse grid, and the fixed-physics grid check across every
prior milestone (M8, M9) has shown small-magnitude differential signals are
generally the ones most vulnerable to sign changes under refinement, this
is a real, not incidental, distinction between the two closures.

## 10. Outcome classification

**Outcome P1 for S1, not confirmed for S2.** Re-reading the handoff's own
decision tree against the *resolution-checked* evidence: P1 requires "S1
shrinks contact, S2 also shrinks contact" to conclude "the sinusoidal
receiver itself resolves the flat-substrate problem and an explicit
receiving surface may be sufficient." S1 satisfies this robustly (same sign
at two resolutions, clean linear rate dependence). S2 satisfies it only at
the single (coarse) resolution tested without a grid check — once checked,
it does not. The correct, precision-matched conclusion is a qualified
P1/P2 hybrid: **local redeposition onto an explicitly curved substrate is
not, by itself, a resolution-robust fix** (contradicting a literal P1
reading), **but the external-reservoir closure Milestone 9 identified
continues to work correctly on this improved, curved geometry** (consistent
with — and now generalizing — Milestone 9's core finding beyond the flat
substrate). This is closer to Milestone 9's own framing than to a novel
P1 resolution: curvature alone does not eliminate the need for the
external-reservoir semantics; it does not contradict them either.

## 11. Recommendation for substrate receiver semantics

1. **Do not treat "add curvature to the substrate" as a substitute for the
   external-reservoir closure.** S2's coarse-grid result would have been a
   tempting but incorrect conclusion without Section 9's grid check —
   report this prominently so it is not mistaken for a resolved question in
   any follow-up work.
2. **The Milestone 9 external-reservoir closure (Model B, `phi_local=0`)
   remains the recommended diagnostic default for the sink-off directional
   campaign**, now validated on two different substrate geometries (flat,
   Milestone 9; sinusoidal, this milestone), with consistent sign and a
   clean linear coarsening-rate dependence in both.
3. If a future explicit finite-receiver geometry is still desired (per
   Milestone 9 Q6), this milestone shows curvature alone is not sufficient
   — the receiver would also need enough resolution-robustness in its own
   right to be trusted quantitatively, which was not demonstrated here for
   local redeposition even with curvature present.
4. Before any further geometry work, extend Section 9's grid check to a
   third resolution (dx=1.25nm) for S1 specifically, to confirm its 80%
   magnitude-retention is converging rather than continuing to drift down
   further with resolution.

**STOP before activating hazard or sink**, per the handoff's explicit
instruction — this milestone is a geometry/closure qualification step, not
a stress-buildup or event-response qualification.
