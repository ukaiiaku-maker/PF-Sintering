# Forensic audit of stochastic campaign seed 20260910

## Scope and provenance

This audit reads the completed campaign at
`runs/three_particle_production_065/stochastic_seed20260910` without advancing,
repairing, interpolating, symmetrizing, or otherwise changing an accepted field.
It covers all 5,511 atomic history records, 616 retained field frames, 13 final
event checkpoints, source fields, ownership fields, and serialized controller
state. Input hashes are recorded in `summary.json`; their sizes and modification
times were checked again after post-processing. No production seed was launched.

The exact bicrystal activation quantity remains the serialized contact-local
stress. Cannon–Carter, center-mean, cluster-weighted, and continuous-integral
stresses are reported only as non-activation diagnostics. Event 9 uses the
committed far-endpoint fitted tangent. Its former last-chord measure changed by
1.096239 MPa, whereas the corrected diagnostic changed by 0.047448 MPa under
the unchanged 0.10 MPa guard. This was a metrology continuity correction; no
physical law or local activation formula changed.

## Main result

The original GIF combined two visual effects. It displayed only the upper
axisymmetric half-plane and colored total solid fraction `f`, which hides both
grain boundaries. More fundamentally, the production-source morphology was
already a compact two-lobe envelope containing a thin, radially tall center
grain. At the source, the center equivalent radius was 77.494 nm, its axial
span was 32.800 nm, both TJ radii were 136.861 nm, and its envelope aspect ratio
was 8.398. The mean TJ radius was 1.766 times the center equivalent radius.
That is not the geometry of three nearly round particles in a chain.

The 13 event transits then redistributed mass without producing axial
densification. Each event added center-grain volume. Summed over all events,

| Grain | Event-wise volume change |
|---|---:|
| LEFT | -4.9359e-24 m³ |
| CENTER | +1.83441e-23 m³ |
| RIGHT | -1.34082e-23 m³ |

The sum closes to numerical roundoff. The final quota strain was +2.02027%,
while outer-centroid strain was -0.083855% and near-axis extent strain was
-0.011440%. Negative measured strain means expansion. The center ended 0.241868%
larger. The quota is therefore an internal transported-distance counter in this
model; it is not a measurement of chain shortening.

## Exact stress audit

For every history record and every one-sided branch, the audit evaluated

`sigma_kappa = -gamma_s k_m`

and

`sigma_TJ = 3 gamma_s sin(theta/2) / r_TJ`.

Their sum reproduces each serialized one-sided local stress, and averaging the
two sides reproduces the serialized production contact stress. The maximum
absolute residual across 22,044 side records and the corresponding contact
means is 7.45e-9 Pa.

The TJ term remains close to 21–22 MPa. Most long-time local-stress evolution
comes from the fitted meridional-curvature term. At LEFT, repeated selected
events drive the curvature contribution more negative until the final local
stress is -0.127 MPa. RIGHT finishes at 25.061 MPa. Meanwhile the continuous
integral diagnostics remain near 21 MPa because they average turning over a
whole branch. The coexistence of a relaxed LEFT local stress and high RIGHT or
cluster stress is mechanically coherent: activation is contact-local, and the
two contacts are not constrained to carry the same local curvature state.

The center-mean stress falls from 18.575 to 14.261 MPa and the area-weighted
cluster local diagnostic from 18.507 to 12.137 MPa. The Cannon–Carter values
diverge strongly from the production local measure as the local curvature
changes, reinforcing why they must not replace the activation stress.

## Geometry, displacement, and strain

From source to final state:

| Quantity | Source | Final | Change |
|---|---:|---:|---:|
| Center volume | 1.949337e-21 m³ | 1.954052e-21 m³ | +0.241868% |
| LEFT volume | 7.270539e-21 m³ | 7.267957e-21 m³ | -0.035521% |
| RIGHT volume | 7.270539e-21 m³ | 7.268407e-21 m³ | -0.029328% |
| Center equivalent radius | 77.4936 nm | 77.5561 nm | +0.0624 nm |
| Center axial span | 32.8000 nm | 32.4489 nm | -0.3511 nm |
| Outer-centroid spacing | 160.8696 nm | 161.0045 nm | +0.1349 nm |
| LEFT TJ radius | 136.8607 nm | 139.4287 nm | +2.5680 nm |
| RIGHT TJ radius | 136.8607 nm | 135.8198 nm | -1.0409 nm |
| LEFT GB/TJ axial position | -16.4000 nm | -16.4000 nm | 0 |
| RIGHT GB/TJ axial position | +16.4000 nm | +16.0489 nm | -0.3511 nm |
| Free energy | 3.59059e-13 J | 3.64990e-13 J | +1.6518% |

The center span contracts because the RIGHT interface moves left, but both
outer centroids move apart enough to produce net expansion. A contracted
internal span is therefore not chain densification. Event-wise centroid-strain
increments range from +0.00940% to -0.01059%; their sum is -0.01201%, while
each event contributes exactly +0.155405% quota strain. Events 1, 2, 6, 7,
and 13 shorten the centroid spacing slightly; the remaining events expand it.

The six robust gross-envelope descriptors never cross the predeclared 5%
departure level. Their largest combined relative departure is 2.539% at the
final frame. Thus the unusual gross silhouette is already present at the
production source. Repeated events mainly add left/right asymmetry, change
local grooves and curvature, and raise mirror error from 4.53e-5 to 0.3346.

## Phase-resolved mass and morphology

Passive reload is the only sustained center-shrinking phase. Across the two
reload intervals it removes 1.43052e-23 m³ from CENTER, while LEFT and RIGHT
gain 2.31956e-24 and 1.19856e-23 m³. Reload lowers free energy by 3.38931e-15 J,
but it expands the outer-centroid spacing by 0.06338%. Center ripening and axial
densification are therefore distinct processes in this geometry.

Active event intervals add 1.83441e-23 m³ to CENTER in total. Facilitated
windows add another 6.78644e-25 m³. The following 20 s reload after Avalanche 1
removes 1.43012e-23 m³, but this does not offset the event-driven gain. That
balance explains the final +4.71481e-24 m³ center change.

Fixed-quota recovery deserves separate treatment. Across 1,437 serialized
recovery records it advances only 0.55971 s over both avalanches, yet it changes
CENTER by +6.01341e-24 m³, RIGHT by -6.55172e-24 m³, centroid strain by
-0.08985%, and energy by -6.90082e-15 J. Native surface relaxation during these
brief fixed-quota episodes is therefore an important part of the mass transfer
and geometric expansion, even though it occupies almost none of the physical
clock.

The retained event checkpoint contains the field before each event and the
accepted field after each source-plus-fast-relaxation step. It does not retain
the intermediate field between conservative source application and fast PF
relaxation. Those two contributions cannot be separated exactly after the fact.
The audit does not invent such a split. Representative current-state masks show
the actual operator: it fills a `0–3W` region near the LEFT TJ and removes equal
total volume from two `3W–10W` free-surface regions. It preserves local ownership
fractions and adds no rigid axial displacement. The combined source geometry,
all-LEFT selection, three-grain ownership, and global fast relaxation therefore
explain the center gain. It is not the output of a prescribed densifying
kinematic map.

## Avalanche-2 clock audit

Avalanche 2 lasts 6,400.904894 s. Its clock partitions as follows:

| Category | Time | Fraction |
|---|---:|---:|
| Accepted transfer-clock intervals | 6,400.396183 s | 99.99205% |
| Explicit fixed-q recovery | 0.481810 s | 0.007527% |
| Facilitated source windows | 0.026901 s | 0.000420% |

There are 1,237 PAUSED and 1,237 RECOVERY records in Avalanche 2, grouped into
retained fixed-q plateaus in `pause_recovery_episodes.csv`. The high record count
does not mean 6,400 s of recovery evolution. The event integrator repeatedly
approaches zero positive transport affinity; because its physical transport
clock slows strongly there, accepted quota increments account for nearly the
entire duration. Short fixed-q recovery steps restore a positive affinity and
can still change shape appreciably because they run the native surface solver.

The long avalanche thus amplifies local curvature and ownership asymmetry, but
it does not create the original two-lobe/center-slab silhouette. That silhouette
exists before Avalanche 1, and no gross descriptor changes by 5% later.

## Why all 13 events are LEFT

Only the two root selections are LEFT-versus-RIGHT stochastic competitions.
At root 1 the rates are virtually identical: 0.06607334 and 0.06607330 s⁻¹.
LEFT crosses because its pre-drawn threshold is 0.0143694, compared with
RIGHT's 1.725365. At root 2, RIGHT is slightly faster (0.06635753 versus
0.06635488 s⁻¹), but LEFT reaches its renewed threshold 1.288902 while RIGHT
has accumulated only 1.336769 toward 1.725365. Both contacts are genuinely
eligible at both roots; the two LEFT roots are explained by threshold history,
not a root-rate or site-count implementation bias.

The remaining 11 selections are descendants. The controller serializes every
descendant onto the contact selected by the root and freezes both root clocks
while that source is alive. RIGHT is mechanically evaluated but is not an
eligible descendant site. Indeed, its diagnostic hypothetical descendant rate
is often larger than LEFT's. Consequently, “13 LEFT events” is not a 13-draw
statistical outcome: it is two stochastic LEFT roots followed by 11 structurally
LEFT descendants. This semantics, combined with persistent local stress
redistribution, creates the observed asymmetry.

## Scientific decisions

### A. Visualization

Rendering total `f` without ownership accounts for much of the ambiguity: it
hides the thin CENTER grain and both GBs. Mirroring the axisymmetric half-plane
and coloring `eta_i/f` makes the three grains visible. Visualization alone does
not explain their extreme shape.

### B. Initial geometry

The source is already unlike an intuitive three-particle chain. CENTER is a
32.8 nm axial slab spanning almost the full radial envelope, with TJ radius
1.77 times its equivalent radius. The gross oddness predates stochastic events.

### C. Event mechanics

The one-b operator primarily redistributes material while quota accumulates.
It does not reliably shorten the outer-centroid or near-axis measures. Across
13 events the measured centroid strain is -0.01201%, versus +2.02027% quota
strain, and free energy rises overall.

### D. Center-particle evolution

Passive ripening shrinks CENTER, but each event grows it. Event transit plus
fixed-q relaxation adds more center material than the 20 s inter-avalanche
reload removes. Ownership-preserving source masks near the LEFT TJ and global
surface relaxation draw material from both outer grains into the center-owned
region.

### E. Long pauses

The 6,400 s duration is transport-clock time near zero affinity, not 6,400 s of
fixed-q recovery. Explicit recovery is 0.482 s in Avalanche 2. It contributes
substantial center growth and geometric expansion, while the initial odd gross
shape remains dominant.

### F. Stress response

High RIGHT and aggregate stress coexisting with relaxed LEFT stress is coherent
under a contact-local activation law. Curvature, rather than the nearly steady
TJ term, drives the separation. Aggregate diagnostics must not be used as root
activation stresses.

### G. Recommendation

**Option 4: both initial geometry and event mechanics need revision before
another stochastic seed.** A new starting state should represent three resolved,
approximately particle-like grains with contact radii smaller than the center
grain scale. A revised three-particle event should connect accepted disconnection
motion to measurable axial displacement and should pass a one-event test in
which centroid and extent strains have the intended sign and scale. Descendant
contact-selection semantics should also be made an explicit modeling choice;
the current persistent-source rule cannot test competition between contacts
inside an avalanche.

## Artifacts and limitations

`history_forensic.csv` retains exact serialized stress, geometry, displacement,
and controller values, along with reconstructed qdot and stress-decomposition
values, for every history record.
`frame_geometry.csv` contains field-derived shape, extent, moment, surface, and
curvature descriptors. `events.csv`, `avalanches.csv`,
`pause_recovery_episodes.csv`, and `decision_audit.csv` provide the requested
event and stochastic audits. Figures A–H, linear and symlog time views,
individual panels, representative masks, the synchronized ownership movie, and
the static montage are stored beside this report.

Near-axis extent and high-order shape descriptors are available at retained
frame times; history rows carry the nearest-frame value and the exact time
offset rather than an interpolated field estimate. Direct source-only versus
fast-relaxation-only volume changes are not identifiable from retained data, as
explained above. All conclusions respect those limits.
