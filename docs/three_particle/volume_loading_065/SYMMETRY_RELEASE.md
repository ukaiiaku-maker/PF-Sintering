# Full-domain symmetry release

The terminal events-off field at 82.94238957332838 s was converted once into
an exactly symmetric ordinary full-domain state. Opposite half-domain samples
were pairwise averaged, the LEFT/RIGHT outer-grain labels were swapped under
reflection, and the full `f, eta_left, eta_center, eta_right` state was written
explicitly on the original 890 x 361 grid.

This is a state handoff, not an evolution constraint. After the handoff:

- reflection is diagnostic only;
- no reflection guard rejects a trial;
- no field or ownership array is mirrored, averaged, or projected again;
- field bounds, material conservation, energy descent, nonlinear error,
  topology/resolution, and milestone checks remain active;
- LEFT and RIGHT contacts continue to be measured independently.

The one-time reconstruction changed `f` by at most 4.99999994e-9. Total
material was unchanged to reported precision, the relative energy change was
2.22044605e-16, field bounds remained [-1.37488e-108, 1], grain-field closure
was 1.11022e-16, and reconstructed scalar/ownership mirror errors are exactly
zero. The source checkpoint SHA-256 is
`131dc22400f65b793027dcbbbc6512914949a86f2af6a8c8ba8175fe765a722c`.

The released events-off lineage is
`runs/three_particle_volume_loading_065_symmetry_released`. Its launch records
`reflection_guard_enabled=false`, `reflection_is_diagnostic_only=true`, and
`no_symmetry_projection=true`. The first immutable post-handoff snapshot was
captured at 83.40925434366856 s and 1.416243405441997% center loss. Its new
post-handoff mirror error was below 2e-13; the larger maximum in its retained
history belongs to the pre-handoff prefix.

The stochastic renewal driver now accepts an explicit source checkpoint. It
loads the ordinary four-field state when present, records mirror error only as
a diagnostic, and selects LEFT or RIGHT solely by localized independent root
clock crossing. The selected pairwise event and all descendants operate on the
full arrays without reflection repair. A regression applies one selected
contact transfer to a symmetric state and verifies that the resulting
asymmetry is retained. The focused suite passes 31 tests.

No stochastic thresholds have been drawn and no production avalanche has been
launched from this state. The existing production gate remains authoritative:
Phase B is disabled and descendant morphology is not qualified. This change
prepares the symmetry-free mechanics without overriding that separate gate.

The earlier numerical-guard result and terminal native-mode audit remain
preserved as the provenance for this one-time handoff. No clipping, fitted
correction, mobility/kinetic change, or merge into production/main occurs.
