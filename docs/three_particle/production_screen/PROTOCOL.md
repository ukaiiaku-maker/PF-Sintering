# Shared bicrystal-physics geometry screen

The stationary-null study is preserved as validation and is **not a Phase-B gate**.
No additional stationary-null construction is part of this study.

Use W=4 nm, dr=0.5 nm and the unchanged GB-aligned dz rule. Scan radius ratios
0.40, 0.50, 0.60, 0.65, 0.70, 0.74 and 0.80; for each, choose outer scales
max(50 nm, 1.025 times the minimum scale for Lc/W=8) and
max(100 nm, 1.30 times that minimum), rounded to 0.001 nm. Include the original
100/70/100 nm geometry explicitly. Ratio 0.80 is a reversed-curvature control.
These are geometric choices, not changes to microscopic parameters.

Every mapped CMC candidate gets the same 200 native steps with GB-face flux
blocked and the same initialization checks and frozen angle-calibration rule.
All candidates are measured even when cleanup checks fail, but rejected states
cannot be selected. Release the GB flux and measure a 1000-native-step interval.
Both GB positions stay fixed. No source, clipping, fitted correction, null
subtraction, barrier change or attempt-frequency change is permitted.

Use the live bicrystal launch manifest, not historical standalone defaults.
Preserve both one-sided local and continuous-integral stresses, the original
within-contact combination, fitted/root barriers, root rates, and GB-to-local-TJ
chemical-potential affinity. The one-b affinity times A_GB times b is recorded
as transport work; it does not enter the empirical EXP-floor root barrier.

There is no deterministic stress threshold in the root law. Report the
constant-state mean waiting time (threshold multiplier / rate) and measured
rate derivatives instead of inventing one. A positive micro-interval derivative
alone cannot justify extrapolation to a stochastic event. Selection requires
resolved geometry, accepted cleanup, the intended center-to-outer ripening,
and evidence that the unchanged hazard becomes appreciable along a valid
trajectory before a numerical/topology stop. Reuse preserved unequal actual
fields for a longer comparison; sparse hazard quadrature is explicitly an
estimate, not an exactly localized stochastic crossing.

Stochastic production is conditional on this viability screen, and remains
disabled until that evidence and the contact-event regression are available.
The old stationary-null failure is not a reason to withhold it.

## Shortlist continuation

The compact 0.74 candidate was shortlisted after its accepted cleanup, intended
initial volume-transfer sign and higher production local stress were observed.
This is not a final selection or a stress-only ranking; the unchanged law gives
a 23.6 s per-contact constant-state mean wait. Continue from the same cleaned
field with the verified current-field integrator and original acceptance checks.
Stop at a field/solver/topology/conservation guard, 30 physical seconds, 900 wall
seconds, or combined hazard 1.25. The last criterion corresponds to a 63.2%
any-root probability for independent live-law thresholds and is only a screen
diagnostic, never a substitute stochastic trigger. Record each accepted state.

A second accepted shortlist, Rc/Ro=0.65 and Ro=119.999 nm, is continued with
identical limits after the 0.74 candidate both grows its center and reaches the
field guard. This tests stronger curvature contrast at the same 8.2W center
resolution. The smaller accepted scale is chosen for faster expected PF
evolution, not by modifying the root law or selecting a random threshold.
The larger 0.65 scale and all remaining candidates stay in the screen table.
