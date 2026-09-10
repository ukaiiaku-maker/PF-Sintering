# Finite-width stationary-null qualification

Freeze the unequal case and all accepted physical/discretization parameters.
Every candidate uses a fresh compatible CMC solve, W=4 nm, dr=0.5 nm, the same
GB-aligned dz rule, signed-distance mapping, exactly 200 native steps with the
GB faces blocked, and the unchanged frozen contact estimator. No extension of
blocked cleanup, stress fitting, mobility change or profile projection.

Constrained thermodynamic chemical potentials require a locally equilibrated
state. Test admissible virtual-work bases and their KKT residuals before
interpreting fitted multipliers as dE_min/dV. If they are strongly basis-
dependent, do not select one basis to manufacture an equilibrium root.
A zero released center-volume rate is then a candidate only. Verify it with
the native solver and test the entire actual field in a released long run.

Before inspecting the released long run, use these operational gates:
- reach at least 0.9 physical seconds within the unchanged field bounds;
- maximum relative center drift <1e-6 (about 0.12% of the prior unequal signal);
- post-0.4-s projected contrast change <5000 Pa;
- post-0.4-s changes in each contact stress <5000 Pa;
- total relative mass error <1e-11 and mirror error <1e-8;
- resolved topology and unchanged curvature-watch diagnostics, inspected for
  new mesh-scale structure. Any numerical bound stop prevents qualification.

The zero of one grain integral is not F(f)=0. Passing an instantaneous flux
root or sharp-interface curvature test must not be called PF stationarity.

Audit one terminal field with native dt/refinement factors 1,2,4,8,16 at a
common finite horizon (128 baseline steps), plus one-step trials. These are
rejected diagnostic arrays, not accepted continuation or restart states.
Retain the original guard and never clip. A one-step epsilon tends to the
inherited epsilon as dt tends to zero; that limit alone says nothing about
spatially converged equilibrium. Report the finite-horizon result and the
outward semidiscrete derivative separately.
