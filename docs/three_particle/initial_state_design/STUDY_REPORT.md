# Earlier-stage initial geometry and unconstrained PF campaign

## Status

- Analytical initial-state screen: **COMPLETE**
- W = 4 nm mapping and minimal cleanup: **QUALIFIED**
- Events-off numerical qualification: **PASS**
- Unconstrained stochastic PF trajectory: **STOPPED ON A GENUINE EVENT-CONTINUATION PATHOLOGY**
- First root and first one-b event: **COMPLETE**
- Descendant event 2: **STOPPED AT 0.1025 b**
- Completed avalanche extinction/repinning cycles: **0**
- No clipping, fitted correction, symmetry enforcement, prescribed stress, or prescribed geometry trajectory

## Initial-state selection

The deterministic screen evaluated 729 sharp-interface geometries and retained 450 that were resolved through a conservative 10% topology probe and had at least one independent perturbation with decreasing interfacial energy and increasing exact local activation stress. Five geometrically diverse states were retained. The selected state was chosen before any stochastic draw:

|quantity|selected value|
|---|---:|
|outer radius|180 nm|
|center radius|99 nm|
|center axial span|138.6 nm = 34.65 W|
|TJ radius|74.25 nm = 18.56 W|
|four nominal one-sided angles|160 degrees|
|initial sharp local stress|39.7902 MPa|
|unchanged two-contact characteristic root clock|8.8703 s|
|independent energy-lowering/stress-raising directions|2|
|energy-conjugate initial stress|16.2788 MPa|

The previously simulated highly necked 0.65 CMC geometry remains a later-stage reference. It did not prescribe this campaign's evolution.

## PF mapping and qualification

The sharp contour was mapped by signed distance at 0.5 nm spacing on a 1884 by 469 grid. Five native surface-diffusion cleanup steps were used only to remove grid-scale mapping artifacts. Cleanup changed maximum grain volume by `1.98e-9`, total volume by zero at reported precision, interfacial energy by `-1.43e-6`, contact radius by `1.08e-4`, angle by 0.0996 degrees, and local stress by 0.128 MPa. No mass correction was applied.

The events-off 0.01 s qualification accepted 56 steps and rejected 9. Interfacial energy decreased by `8.38e-5`; maximum mass error was `2.22e-16`, maximum mirror error was `3.10e-14`, and all bound and topology guards passed. The qualified source SHA-256 is `55b39e90db79b675956a9465d00cfe533f585aa50ff9df1542e195f4e0d9ef6d`.

## Unconstrained pre-root evolution

Seed 20260914 and both thresholds were fixed only after the qualified source was selected. LEFT and RIGHT thresholds were 1.1077139602 and 2.2233782943. Symmetry enforcement was disabled.

The center first grew, reaching about 0.279% above the source, while local stress rose from 58.3246 MPa to a 59.6042 MPa peak at 1.01 s. The subsequent trajectory reversed without intervention. At the first LEFT root, 10.540964661 s, the center had lost 0.80137%, local LEFT/RIGHT stresses were 51.7030/51.7015 MPa, and geometric strain was `8.56668e-4`. Stress had therefore relaxed by 7.90 MPa from its peak before nucleation.

At the root, the independent interval energy-balance stress was 54.3053 MPa and exact area-weighted contact stress was 51.7023 MPa, a 5.03% difference. The energy-balance quotient is singularly sensitive while geometric strain is near zero and changes sign early in the trajectory; those early multi-GPa values are retained as diagnostics and are not interpreted as an evolution law.

## Root and avalanche response

The first one-b event completed at 10.598821918 s. LEFT stress fell to 49.6825 MPa while RIGHT remained 51.7013 MPa; area-weighted aggregate stress remained 50.6786 MPa. Mirror error grew to 0.1287 with no symmetry projection. The selected contact relaxed while the unselected contact retained the high cluster stress.

A descendant crossed at 10.598823842 s, only 1.92 microseconds after the source window opened. Descendant event 2 accepted 21 increments through 0.1025 b. At the last 0.01 b checkpoint, LEFT/RIGHT stresses were 48.5386/51.7013 MPa and aggregate stress was 50.0974 MPa.

Event 2 then rejected three trials at the qualified minimum increment of 0.00125 b because the `transport_affinity_MPa` change limit was violated. The driver stopped at 10.607426481 s with `current-state transfer failed at minimum step`. The field remained bounded, mass-conserving, and topologically valid. The minimum increment was not reduced, the limit was not relaxed, and no clipping or fitted correction was introduced.

## Interpretation

This trajectory supports high-stress stochastic nucleation during coarsening rather than loading-triggered nucleation. The root occurred after stress had peaked and substantially relaxed, while the cluster still carried about 52 MPa. The first completed event then showed direct multi-contact stress retention: relaxation localized to LEFT while RIGHT stayed near 51.7 MPa and the aggregate remained near 50 MPa.

The result does not establish a complete renewal cycle, oscillatory steady state, or preferred long-time stress because event 2 could not pass the unchanged minimum-step transport-affinity qualification. That failure is part of the scientific result and defines the current limit of the complete shared microscopic kinetics on this earlier-stage geometry.

Detailed machine-readable results are in [campaign_result/summary.json](campaign_result/summary.json), [campaign_result/events.csv](campaign_result/events.csv), and [campaign_result/trajectory_audit.png](campaign_result/trajectory_audit.png).
