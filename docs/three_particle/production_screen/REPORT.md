# Shared bicrystal-physics viability result

**The stationary null is no longer a gate. No event-producing geometry is yet
qualified by this screen; stochastic three-particle production remains disabled.**
The original unequal 0.70 case, physics, field guard and live production are
unchanged. No clipping, fitted correction, barrier tuning or null subtraction.

## What is implemented and verified

Both contacts now call the actual bicrystal stress formulas, retaining all four
one-sided local values and all four continuous-integral values. Each contact
passes the existing within-contact local stress to the unchanged live root
barrier/rate. The actual PF chemical-potential affinity and one-b transport work
are independent outputs. See [exact call-path audit](PHYSICS_AUDIT.md) and the
frozen [live launch manifest](bicrystal_launch_manifest.json).

The absent-third-grain field regression matches both bicrystal stress families
and their one-sided values exactly. Root-rate parity is tested over negative,
zero and positive stresses with the live temperature, barrier and prefactor.
The focused suite passes **37 tests**. Production source files were not edited.

## Resolved geometry screen

15 candidates were evaluated at W=4 nm and dr=0.5 nm, using aligned dz,
the same mapped CMC construction, 200 blocked-GB cleanup steps and 1000 released
native steps. **7 pass the existing cleanup/topology checks.**
The remainder are retained as rejected evidence; their rates cannot justify
selecting them. The largest two geometries deliberately test the cost of
resolving a very small center radius ratio.

| Rc/Ro | Ro (nm) | Lc/W | Cleanup | LEFT local (MPa) | LEFT root rate (s⁻¹) | Mean wait/contact (s) |
|---:|---:|---:|:---|---:|---:|---:|
| 0.40 | 509.500 | 8.200 | reject | 0.4203 | 0.06825 | 18.32 |
| 0.40 | 646.195 | 10.400 | reject | 0.6110 | 0.09185 | 13.61 |
| 0.50 | 261.671 | 8.200 | reject | 4.4126 | 0.06424 | 19.46 |
| 0.50 | 331.876 | 10.400 | reject | 3.7545 | 0.07639 | 16.36 |
| 0.60 | 152.124 | 8.200 | reject | 10.3292 | 0.05709 | 21.90 |
| 0.60 | 192.937 | 10.400 | reject | 8.4283 | 0.06448 | 19.39 |
| 0.65 | 119.999 | 8.200 | pass | 14.1187 | 0.05491 | 22.77 |
| 0.65 | 152.194 | 10.400 | pass | 11.4085 | 0.06066 | 20.61 |
| 0.70 | 96.403 | 8.200 | reject | 18.5205 | 0.05357 | 23.33 |
| 0.70 | 100.000 | 8.506 | pass | 17.7837 | 0.05390 | 23.19 |
| 0.70 | 122.268 | 10.400 | pass | 14.8707 | 0.05797 | 21.56 |
| 0.74 | 81.850 | 8.200 | pass | 22.5086 | 0.05302 | 23.58 |
| 0.74 | 103.810 | 10.400 | reject | 18.0162 | 0.05650 | 22.13 |
| 0.80 | 65.118 | 8.200 | pass | 29.2811 | 0.05292 | 23.62 |
| 0.80 | 100.000 | 12.593 | pass | 19.5194 | 0.05784 | 21.61 |

LEFT/RIGHT full quantities, derivatives of every scalar stress/work/rate and
resolution margins are in [screen.json](screen.json). Symmetric local stresses
and root rates agree between contacts; the original signed full-branch integral
conventions are preserved separately. No new stress scalar ranks these cases.
Stronger curvature contrast does not imply stronger local activation stress;
larger sizes also change the existing circular-TJ site count.

The micro-interval derivatives include substantial mapped-field relaxation.
They are not extrapolated linearly to infer an event time. The ratio 0.80 is a
reversed-curvature control, not an intended smaller-center ripening candidate.

## Preserved 100/70/100 nm actual-field trajectory

The coherent overlap -> extend -> refined -> refined-target checkpoint chain
was remeasured without evolving or changing it. Across 43
actual saved states to 0.941910 s, the LEFT production local stress
changes 17.78369 -> 20.84846 MPa,
and its root rate changes 0.053899 ->
0.060954 s⁻¹. The center volume changes
-0.084096%.

From the first saved state at/after 0.398847 s (0.462455 s),
the further local-stress rise is
0.053337 MPa.
Thus initial relaxation and later loading are kept distinct.
Sparse trapezoidal quadrature gives P(any root) approximately
8.762% through the valid trajectory.
This is a probability estimate, not a sampled or exactly localized crossing.

## Compact shortlist continuation

Rc/Ro=0.74 and Ro=81.85 nm pass cleanup with Lc/W about 8.2.
Its initial mean waiting time is 23.6 s per contact. The guarded current-field
continuation ends at **0.919749 s**, status **FIELD_OR_SOLVER_GUARD**,
after 289.2 wall seconds. Center volume changes
**+0.046748%**; final LEFT local stress is
**25.674690 MPa** and rate
**0.059296 s⁻¹**.
Integrated hazards are 0.054486 and
0.054486; P(any root) is approximately
**8.349%**.
A positive center-volume change means this shortlist also fails the required
continued smaller-center ripening direction, despite its initial negative
micro-interval volume rate. That is a viability rejection, not a null test.
The event-free continuation does not reach the preregistered combined-hazard
screen target of 1.25 unless its status explicitly says so.
See [every accepted state](continuation_0.74_81.850.json).

This study does not establish that the proposed shared-physics mechanism is
impossible. It establishes that these tested trajectories have not yet supplied
a validated event-producing geometry under the unchanged numerical limits.
A stochastic implementation and event/avalanche strain histories remain
conditional on that viability evidence. No stationary-null optimization is
needed or planned. A future numerical-bound remedy must be independently
validated rather than replacing the field guard, clipping, tuning kinetics or
choosing a favorable random threshold.

## Stronger-drive shortlist

Rc/Ro=0.65, Ro=119.999 nm also passes cleanup at about 8.2W thickness.
Its continuation ends at 0.864443 s with status
**FIELD_OR_SOLVER_GUARD**, after 400.1 wall seconds.
Center volume changes **-0.179747%**;
LEFT local stress is 17.248903 MPa
and the root rate is 0.063267 s⁻¹.
The accumulated hazards are 0.054508 and
0.054508, giving approximate any-root probability
**8.352%**.
See [accepted states](continuation_0.65_119.999.json).

Both shortlists use particle-positive bicrystal coordinates at each GB.
The preliminary global-coordinate integral histories are explicitly archived
as rejected diagnostic provenance; they are not used in these results.

## Outputs and reproducibility

[Study plots](screen_plots.pdf), [preserved unequal history](preserved_unequal_history.csv)
and the JSON records retain LEFT/RIGHT stresses, hazards and center volume.
The history also labels a center two-contact stress diagnostic and an
area-weighted cluster stress diagnostic; neither enters activation.
No event markers, avalanche sizes or activated strain are fabricated for an
events-off study. Re-run the entry points listed below;
all actual field arrays stay in the isolated runs subtree with hashes.
Entry points are `three_particle_production_screen.py`,
`three_particle_production_history.py`,
`three_particle_production_viability_continue.py`,
`three_particle_production_screen_report.py` and
`three_particle_production_screen_figures.py` under `scripts/`.
