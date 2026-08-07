# Physics background: coarsening-driven stress buildup and nucleation-limited densification

## Purpose of this document

This file gives the scientific background for the current Python phase-field sintering investigation. It is intended to keep code development anchored to the experimental and theoretical mechanism rather than to a particular legacy implementation.

Read this together with `CODEX_PHYSICS_SEQUENCE.md` before changing the model.

The primary sources motivating the model are:

1. S. J. Dillon et al., *Interface nucleation rate limited densification during sintering*, Acta Materialia 242 (2023) 118448.
2. O. Hussein et al., *Plateau-Rayleigh instability with a grain boundary twist*, Applied Physics Letters, DOI 10.1063/5.0103658.
3. The accompanying theory manuscript on renewal-controlled interface-normal strain, point-defect transport/exchange, and thermodynamic generation of sintering stress.

The code does not need to reproduce every numerical choice in older MATLAB scripts. It should reproduce the physically correct mechanism and obey conservation and thermodynamic constraints.

---

## 1. Experimental response that the model is intended to capture

The experimentally important feature is intermittent densification rather than continuous Coble-like strain.

The sequence observed in the bicrystal/particle-on-substrate experiments is:

1. A particle or bridge coarsens while there is no active climb-mediating grain-boundary defect that can act as an unsaturable point-defect source/sink.
2. During this waiting period there is little or no rigid-body densification even though interfacial diffusion can be rapid.
3. The shrinking particle/contact geometry evolves and the solid-solid contact or neck becomes smaller.
4. The local sintering stress at the neck/triple-junction region increases.
5. Once the local stress becomes sufficiently large, a climb-mediating grain-boundary dislocation/disconnection nucleates.
6. Point-defect transport can then proceed through the active sink and a rapid rigid-body displacement/densification increment occurs.
7. The contact broadens and the local sintering stress relaxes.
8. The active defect has a finite displacement/strain quota and is eventually exhausted; a new event requires renewed nucleation.

The key experimental distinction is therefore:

- coarsening is continuous on the observation time scale;
- densification is discontinuous;
- stress builds during sink-off periods and relaxes during densification events.

This is the qualitative response the simulations must ultimately recover.

---

## 2. Why densification can stop even when diffusion remains fast

Grain-boundary-mediated interface-normal strain requires more than diffusivity.

At least three serial physical ingredients are required:

1. a climb-mediating defect must exist or nucleate;
2. point defects must be created/annihilated or exchanged at the active defect;
3. point defects must be transported between complementary source and sink regions.

Traditional diffusional-sintering models usually assume the first two are effectively instantaneous and that grain boundaries provide continuously available ideal sources/sinks. The experiments motivating this model show that this assumption can fail.

A boundary may possess a large diffusivity and yet not densify if no appropriate active sink exists. Once a sink nucleates, the same diffusivity can support rapid strain. Thus a long incubation period followed by fast rigid-body motion is compatible with high interfacial diffusivity.

This is why the code must never allow ordinary surface/GB diffusion to create hidden rigid-body densification while the sink is inactive.

---

## 3. Finite sink lifetime and renewal kinetics

An active grain-boundary dislocation/disconnection is not an indefinitely operating sink.

A defect traversing a finite grain-boundary facet carries only a finite normal displacement of order a Burgers-vector-scale quota, modified by geometry. After that event completes, continued densification requires another activation event.

The current conceptual model is therefore a renewal process:

- inactive state: wait for nucleation;
- active state: complete a finite transport/displacement quota;
- exhausted state returns to inactive;
- repeat.

The stochastic integrated-hazard implementation in the Python code is intended to represent the waiting period. The active-sink/RBM implementation is intended to represent the quota-completion period.

These two stages should be independently testable before being coupled.

---

## 4. Coarsening under arrested densification is the stress-building mechanism

The central scientific hypothesis is that surface/interfacial energy dissipation during coarsening can generate a *larger local stress concentration* even while total free energy decreases.

When the sink is inactive:

- rigid-body shortening is arrested;
- the particle or bridge can still shrink/coarsen;
- the relevant separation/length cannot decrease by densification;
- the contact or load-bearing GB region can decrease;
- the geometry can move away from a locally low-stress configuration;
- capillary force becomes concentrated over a smaller region;
- local neck/triple-junction stress rises.

The required sink-off directional signature is therefore:

- `dV2 < 0`;
- densification strain approximately zero;
- particle/substrate or particle/particle separation approximately fixed;
- neck/contact measure decreases;
- local sintering stress increases;
- total interfacial free energy decreases.

The last two conditions must be allowed to coexist. A rising local chemical-potential/stress concentration does not imply increasing total free energy.

---

## 5. Plateau-Rayleigh/de-sintering analogy

The Plateau-Rayleigh instability is useful as a thermodynamic analogy and as a compact sanity check, but the sintering geometry does not need to start as a PR rod.

For a constrained system with no active densification sink, coarsening can reduce the particle size while the system length is effectively fixed. The ratio of length scale to particle/contact scale therefore grows. The system can be driven toward a PR/de-sintering-like condition as coarsening proceeds.

The APL work establishes several relevant facts:

- GBs can destabilize capillary geometries and reduce the critical wavelength for pinch-off;
- GB width can decrease continuously during surface-diffusion-driven evolution;
- phase-field calculations reproduce the direction and kinetics of neck narrowing/pinch-off;
- GB energy, perturbation amplitude, and grain coarsening strongly influence the instability.

For this project, these results provide evidence that capillarity plus constrained geometry can naturally generate increasing local curvature/stress concentrations.

However, do not turn the present task into a full reproduction of the PR paper. Use a PR/de-sintering benchmark only if the ordinary sink-off substrate geometry fails to show the expected direction and the surface-transport kernel itself needs a sanity check.

---

## 6. Local sintering stress, not a uniform GB-average stress

The nucleation event is controlled by the local state at the neck/triple junction, not by a uniform stress assigned to the entire GB plane.

Do not replace the model's local stress construction with a simple `sintering potential / total GB area` expression.

The relevant stress is spatially concentrated and depends on local geometry, including some combination of:

- contact/neck scale;
- free-surface curvature near the junction;
- local dihedral/triple-junction geometry;
- surface and GB energies;
- anisotropic Cahn-Hoffman forces when anisotropy is active;
- evolving GB excess energy or related local state variables when physically justified.

Diagnostics should decompose these contributions so that a change in local stress can be traced to a change in geometry rather than inferred from a single scalar output.

The exact local stress formulation can be audited and improved if necessary, but it should remain a local neck/TJ quantity.

---

## 7. Surface diffusion, reservoir/Ostwald exchange, and GB relaxation are different processes

Do not combine these into one generic coarsening rate.

### Surface/free-surface transport

This redistributes material along the free surface in response to capillarity. It changes shape without changing total solid mass. It should decide how the free surface responds to curvature and interfacial-energy gradients.

### Reservoir/Ostwald exchange

This changes the net amount of a shrinking grain by moving mass to or from an external/remote reservoir. In the current particle-on-substrate model this is the permitted secular grain-volume-change mechanism during sink-off coarsening.

The reservoir should ideally specify the *net mass/chemical-potential exchange*, while local surface evolution determines where the surface recedes. An artificial rule that protects the neck can suppress the phenomenon of interest.

### GB/eta relaxation

This changes the structural/contact geometry but should not itself create or destroy grain volume. The mass-preserving structural projection enforces this.

GB relaxation can nevertheless be too fast dynamically. If the contact/triple-junction geometry continuously relaxes toward a low-stress state faster than coarsening drives it, local stress accumulation may be suppressed. GB mobility therefore needs to be controlled independently from free-surface mobility.

---

## 8. Two leading failure mechanisms currently under investigation

### H1: neck protection in the current reservoir/Ostwald operator

The current implementation includes a spatial exclusion/localization around the neck. This may cause the shrinking particle to lose volume primarily away from the contact while preserving too much material at the neck.

Possible symptom:

- `V2` decreases;
- contact remains large;
- local stress remains constant or falls.

The preferred alternative is not to force removal *at* the neck, but to remove the artificial protection and let capillary/surface-diffusion dynamics choose where the geometry retreats.

### H2: GB/eta relaxation is too rapid

The structural fields are now volume-neutral, but high `M_eta` can still move the contact/triple-junction geometry. If that relaxation is much faster than shrinking/coarsening, geometric frustration cannot accumulate.

The immediate benchmark should therefore compare current GB mobility against a very low/zero-mobility limit.

---

## 9. Conservation and mechanism-isolation constraints already qualified

The following numerical invariants are now part of the physics contract and should not be weakened casually:

1. `f` is the conserved solid-mass field.
2. CH/free-surface evolution must conserve total solid mass to numerical tolerance.
3. Structural/eta relaxation must not secularly change integrated grain ownership.
4. Explicit reservoir/Ostwald exchange is allowed to change grain volume.
5. RBM/advection should not create secular grain-volume loss after constrained projection.
6. A sink-off run must not contain hidden densification/RBM.

The recent operator-ledger tests verified that, with the mass-preserving structural projection, non-Ostwald contributions to integrated `V2` are at floating-point roundoff on both the small development grid and the full v64 spatial resolution.

Preserve these invariants while investigating stress buildup.

---

## 10. Required short-time sink-off response

The first scientific benchmark does **not** require nucleation.

Start from a geometry near enough to the anticipated transition that a small amount of coarsening gives a measurable directional response. Keep the sink disabled and RBM disabled.

Run only until approximately

`|Delta V2| / V20 ~ 1e-4 to 1e-3`

or an equivalently short horizon.

A successful direction is:

- particle volume decreases;
- separation/length remains fixed;
- densification strain remains approximately zero;
- neck/contact width decreases;
- GB/contact measure decreases;
- local neck/TJ stress increases;
- total interfacial free energy decreases.

The immediate 2x2 mechanism matrix is:

| Case | Reservoir neck localization | GB/eta mobility |
|---|---|---|
| A | current | current |
| B | neck-unprotected / geometry-driven | current |
| C | current | very low/zero |
| D | neck-unprotected / geometry-driven | very low/zero |

Only these two mechanism controls should differ between the four runs.

---

## 11. Event-response benchmark after stress buildup works

Once a sink-off state clearly builds stress, checkpoint it near the high-stress configuration.

Before reconnecting stochastic nucleation, force one sink active in a benchmark-only mode.

The required event direction is:

- rigid-body displacement increases;
- densification strain increases;
- separation decreases;
- neck/contact broadens;
- local sintering stress decreases;
- total solid mass remains conserved;
- grain-volume change beyond explicit background Ostwald exchange remains negligible.

If this does not happen, the RBM/transport geometry must be corrected before hazard tuning.

---

## 12. Stochastic nucleation comes only after the two deterministic directions are qualified

After both of the following are independently demonstrated:

1. sink-off coarsening -> local stress buildup;
2. forced active sink -> neck broadening and stress relaxation;

then reconnect the integrated nucleation hazard.

The first coupled stochastic qualification only needs one clean event:

sink inactive -> coarsening -> neck narrows -> local stress rises -> hazard accumulates -> nucleation -> sink active -> RBM/densification -> neck broadens -> stress drops -> quota completes -> sink inactive.

Do not tune the barrier to compensate for geometry that does not build stress.

---

## 13. Relative-rate questions to ask only after the mechanism works

Once the directional mechanism is established, study the ordering among:

- free-surface transport rate;
- reservoir/Ostwald exchange rate;
- GB/eta relaxation rate;
- active-sink completion / GB transport rate.

Short multiplicative perturbations around a qualified state are preferable to long sweeps.

Useful response measures are:

- `dx_neck/dV2`;
- `dSigma_local/dV2`;
- stress gain before activation;
- neck broadening per RBM displacement;
- stress drop per event;
- whether GB relaxation erases stress before nucleation.

Anisotropy can be tested after the isotropic/weak-anisotropy mechanism is understood; it should not be used first as a fitting knob.

---

## 14. Thermodynamic acceptance criteria

At every stage, prefer mechanism-level checks over visual similarity.

A physically acceptable trajectory should satisfy:

- total solid mass conservation except for explicitly modeled reservoir exchange when appropriate;
- decreasing total interfacial free energy for spontaneous coarsening;
- no densification without an active eligible sink;
- no grain-volume loss from numerical structural regularization;
- local stress evolution consistent with local geometry;
- finite strain quota for each active sink event;
- stress relaxation during an active densification event.

If a code change produces the expected movie but violates these constraints, it is not an acceptable solution.

---

## 15. Scientific objective

The immediate debugging problem is not the end goal.

The end goal is a physically interpretable phase-field/kinetic model that explains why a system with rapid diffusion can nevertheless densify intermittently because active point-defect sinks are nucleation limited.

The model should explain how coarsening under arrested densification generates the local stress needed for defect nucleation, and how finite sink activation then converts stored capillary driving force into a rapid, discrete densification increment.

Keep that mechanism in view when making every numerical or architectural change.
