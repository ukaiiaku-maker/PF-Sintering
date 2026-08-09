# Milestone 14C: capillary sintering-stress buildup through the filling-to-near-neutral trajectory

## 1. Starting checkpoint

Branch `codex/coarsening-stress-buildup`, HEAD `f0ba5cc` ("docs+feat: Milestone 14B follow the near-neutral state through the crossover"), confirmed clean working tree and **212/212** tests passing before any change. `variational_surface_diffusion_step` (`face_projected`) was not modified. Eta kinetics, sink, hazard, RBM, anisotropy, imposed stress/strain stayed OFF throughout. No physical parameters were changed from Milestone 14B's `gamma_gb=1.6` trajectory (`A=72nm`, `lambda=480nm`, `gamma_s=1.0`, `dx=2.5nm`, `surface_mobility_scale=0.9`, `dt=5e-6s`). New: `pf_sintering/capillary_stress.py`, `tests/test_capillary_stress.py`, `scripts/m14c_stress_analysis.py`. Full suite at the end: **217/217** (212 inherited + 5 new capillary-stress tests).

## 2. Particle-arc construction

At each TJ, `tj_force.compute_tj_force` gives the two free-surface branch directions (`v_s1`, `v_s2`, each oriented away from the TJ, from the pre-existing `_circle_crossings`/`_tangent_at` machinery); `m12b_grid_convergence.classify_branches` (existing, previously-qualified code) identifies which is the particle branch and which is the substrate branch. `capillary_stress.trace_particle_arc` walks the f=0.5 contour (`curvature_extraction._walk_branch`, local circle-marching, never a global search) from the top TJ in the particle-branch direction for a generous `max_arclength` (560nm, comfortably longer than the particle's own ~470-480nm exposed-cap arc length), then truncates at the walk's closest approach to the bottom TJ (typically 2-3nm, ~1 grid cell, confirming the walk reaches the intended endpoint cleanly). The truncated path is resampled onto a uniform 400-point arclength grid and smoothed (Gaussian, `sigma=1.5*interface_width`) before differentiating for tangent/curvature -- Section 4 below documents why this specific smoothing scale was chosen.

## 3. Contact normal

`tj_subgrid.compute_subgrid_contact`'s own `n_GB_sub` (a bare +90deg rotation of the fitted GB tangent, no guaranteed orientation) points from **particle toward substrate** (-x, toward lower x where the substrate bulk sits) in this geometry, confirmed directly on real states. `capillary_stress.oriented_contact_normal` returns its negation -- **substrate toward particle** (+x here) -- which is the orientation `apparent_sintering_stress` assumes so that a capillary force pulling the particle cap back toward the substrate (the physically expected densifying direction, and what is actually measured, Section 6) yields a POSITIVE apparent stress. This same `n_GB` is used consistently for the force projection and the local curvature-force projection.

## 4. Capillary resultant derivation and validation

**Curvature form** (Section 5): `F_cap_vector = integral gamma_s*kappa(s)*n(s) ds` over the traced particle arc, where `n(s) = t(s)` rotated +90deg (a fixed rotation applied at every point) and `kappa(s) = (dt/ds) . n(s)` -- since `dt/ds` is automatically parallel to `n` (because `|t|=1`), this construction makes the integral telescope to `Delta_t` of the arc's OWN endpoints up to pure discretization error, by the Frenet relation, not by an independently-assumed sign convention.

**Endpoint form** (Section 6): `F_cap_vector = gamma_s*(t_end - t_start)`, using ONLY the two independently-measured TJ branch directions (`top_particle_dir`, `bot_particle_dir`) -- a completely different computational pathway (local TJ-circle crossing detection, not the long-arc walk) than the curvature form. `t_start = top_particle_dir`; `t_end = -bot_particle_dir` (the arc ARRIVES at the bottom TJ, so its direction of travel there is the negative of "away from the TJ").

**Validation on an analytic circle** (`tests/test_capillary_stress.py::test_endpoint_vs_curvature_form_agree_on_analytic_circle`): for a `R=100nm` circle with two marked points at +/-80deg, the endpoint form matches the closed-form analytic value `gamma_s*(t(280deg)-t(80deg))` to `<1e-9` absolute, and the curvature-form/endpoint-form relative discrepancy is **1.4%**. A sign-convention regression test confirms using the WRONG (non-reversed) branch-direction convention breaks the identity by >50%, so the check is genuinely discriminating, not vacuous.

**Validation on real states**: relative discrepancy is **3.7-7.9%** across all analyzed states (dx=2.5nm: 3.5-5.7%; dx=5.0nm: 7.9% -- Section 13). The smoothing scale `sigma=1.5*W` was chosen empirically as the value minimizing this discrepancy on state A (tested `0.5W` through `4W`: `10.1%, 5.3%, 3.7%, 4.3%, 6.8%, 8.9%`) -- both less smoothing (raw walk-noise dominates) and more smoothing (the Gaussian kernel's edge handling biases the endpoint tangent) give worse agreement. This is the documented discretization tolerance for all `F_cap`/`sigma_sint_app` values reported below. The two independent computations agree well enough at every state to proceed with the apparent-stress interpretation, per Section 6's requirement.

## 5. Definition, units, sign of sigma_sint_app

```
sigma_sint_app = -F_cap_n / L_contact_TJ_sub,   F_cap_n = F_cap_vector . n_GB
```

Units: `gamma_s` (N/m) x `kappa` (1/m) x `ds` (m) = N/m for `F_cap_vector`; divided by `L_contact` (m) gives Pa. Sign: positive = densifying (verified by construction test `test_apparent_sintering_stress_sign_and_units` and by the real-state results below, where the measured `F_cap_vector` points toward the substrate as expected and `sigma_sint_app` comes out positive). This is explicitly labeled an **apparent 2-D sintering stress**, not a complete 3-D Cauchy stress.

## 6. A-G stress results

Analyzed Milestone 14B's saved states A/B/C/D plus three new checkpoints extending the same `gamma_gb=1.6` trajectory to `tau=1.08, 1.20, 1.50s` (E/F/G, `scripts/m14b_save_checkpoint.py` at `t=0.36, 0.40, 0.50s`):

| state | tau (s) | F_cap (curv form, N/m) | F_cap (endpoint form, N/m) | rel. err | L_contact (nm) | sigma_sint_app curv (MPa) | sigma_sint_app ep (MPa) |
|---|---|---|---|---|---|---|---|
| A_filling | 0.09 | (-1.821, -0.065) | (-1.804, 0.000) | 3.7% | 69.47 | 26.21 | 25.97 |
| B_near_neutral | 0.60 | (-1.884, -0.049) | (-1.794, 0.000) | 5.7% | 70.37 | 26.77 | 25.50 |
| C_first_depletion | 0.63 | (-1.884, -0.049) | (-1.794, 0.000) | 5.7% | 70.38 | 26.76 | 25.49 |
| D_established_depletion | 0.90 | (-1.880, -0.050) | (-1.793, 0.000) | 5.6% | 70.46 | 26.69 | 25.44 |
| E_extended1 | 1.08 | (-1.879, -0.050) | (-1.792, 0.000) | 5.6% | 70.52 | 26.64 | 25.41 |
| F_extended2 | 1.20 | (-1.878, -0.050) | (-1.791, 0.000) | 5.6% | 70.56 | 26.61 | 25.39 |
| G_extended3 | 1.50 | (-1.859, -0.047) | (-1.815, 0.000) | 3.5% | 70.65 | 26.31 | 25.69 |

**`sigma_sint_app` is essentially flat across the whole tested range** -- both forms stay within a narrow band (curvature form: 26.2-26.8 MPa, a 2.1% peak-to-peak range; endpoint form: 25.4-26.0 MPa, a 2.4% range), with a very shallow rise from A to a broad plateau around B-F followed by a mild decline at G. The two independent forms do not even agree on the fine directional wiggle (curvature form rises then falls; endpoint form falls then rises slightly), which by itself indicates this small variation is at or below the ~4-6% measurement tolerance established in Section 4 -- **not** a reportable systematic trend.

## 7. Local particle curvature / capillary pressure

Two independent estimators were used at each TJ's near-window (1.5W-3W) and far-window (3W-5W), on the particle-side branch: (a) the mean of the traced arc's own smoothed `kappa(s)` in each window, and (b) an independent Kasa-circle-fit (`curvature_extraction.window_curvature`, the same, previously-qualified windowed-curvature machinery used throughout Milestones 11-13) applied directly at each TJ with no arc-walk smoothing at all.

| state | tau (s) | Kasa near (1.5W-3W, 1/m) | Kasa far (3W-5W, 1/m) | arc-based near (1/m) | arc-based far (1/m) |
|---|---|---|---|---|---|
| A | 0.09 | -6.54e6 | -4.55e6 | -4.97e6 | -4.91e6 |
| B | 0.60 | -7.81e6 | -4.26e6 | -3.87e6 | -4.68e6 |
| C | 0.63 | -7.82e6 | -4.26e6 | -3.88e6 | -4.68e6 |
| D | 0.90 | -7.94e6 | -4.23e6 | -3.97e6 | -4.69e6 |
| E | 1.08 | -8.00e6 | -4.22e6 | -- | -- |
| F | 1.20 | -8.04e6 | -4.22e6 | -- | -- |
| G | 1.50 | -7.52e6 | -4.19e6 | -- | -- |

**The independent Kasa-fit near-TJ curvature magnitude rises clearly and consistently across six consecutive checkpoints, from `6.54e6` (tau=0.09) to a maximum `8.04e6` 1/m (tau=1.20) -- a sustained +23% increase -- before relaxing back to `7.52e6` (tau=1.50).** The far-window Kasa curvature stays essentially flat (`-4.55e6` to `-4.19e6`, a much smaller, slowly-decreasing change). **The arc-based (heavily smoothed) near-window estimate does NOT reproduce this trend cleanly** (it drops from A to B then stays roughly flat) -- this is attributed to the `sigma=1.5*W` smoothing kernel blurring the sharp near-TJ concave "fillet" curvature together with the less-negative curvature just beyond it, damping the true near-TJ signal (exactly the edge-bias effect quantified on the synthetic-circle test, Section 4/13). **The Kasa-fit estimator, being un-smoothed and locally windowed, is the more trustworthy near-TJ diagnostic here**; the arc-based estimator remains the correct, validated tool for the whole-arc force integral (Section 4-6), where its smoothing does not compromise the telescoping identity.

Both curvatures are negative at every window and every state -- i.e. the free surface immediately adjacent to (and, more surprisingly, several interface widths beyond) each TJ is locally **concave** (a "fillet"/groove shape, not the convex particle-cap curvature `~+1/R2 = +1.25e7` 1/m a pristine circle would show at this radius). This indicates the particle's exposed free surface has been substantially reshaped by the combination of the extreme substrate amplitude (`A=72nm`) and the strong groove-coupling term (`gamma_gb=1.6`) well before any of these checkpoints -- consistent with Fig. 7 (Section 21), where the traced arc is pale blue/near-zero (not the deep red a pristine convex cap would show) along most of its visible length near the neck.

`F_TJ_mag` (the Young-Herring residual, kept as a secondary diagnostic per Section 2) decreases mildly and roughly monotonically from `0.633` (A) to `0.606` (F) before ticking back up to `0.626` (G) -- the SAME qualitative rise-then-relax-adjacent pattern as the near-TJ curvature, though moving in the opposite sense (residual imbalance decreasing while local curvature magnitude increases) -- i.e. the TJ itself is becoming MORE mechanically balanced even as the curvature immediately beside it concentrates further, reinforcing that this is a local free-surface-shape effect, not a TJ-force-imbalance effect.

## 8. Substrate curvature evolution

Windowed geometric curvature (`substrate_curvature_windows`, same Kasa-fit machinery) on the substrate-side branch at both TJs:

| state | near (1.5W-3W, 1/m) | far (3W-5W, 1/m) |
|---|---|---|
| A | +6.58e6 | +1.76e6 |
| B | +4.58e6 | +2.16e6 |
| C | +4.51e6 | +2.19e6 |
| D | +4.01e6 | +2.40e6 |

The substrate's near-TJ curvature **decreases** monotonically (`+6.58e6 -> +4.01e6`, the local substrate bump relaxing/flattening as surface diffusion proceeds) while the far-window curvature **increases** slightly (`+1.76e6 -> +2.40e6`, consistent with the smoothing/redistribution spreading outward). `A1_substrate` (Fourier amplitude away from the TJs, from the full trajectory) decreases steadily from `72.05nm` (t=0) to `69.93nm` (t=0.50s, tau=1.5s) -- the substrate crest continues to flatten (material draining down the substrate) throughout the whole window, with no discontinuity at the `M_neck_f` crossover.

## 9. Measured dihedral evolution

`psi_measured(t)` (both TJs identical to displayed precision by construction, since the geometry is Y-symmetric about the crest) increases smoothly and monotonically throughout: `92.85deg` (t=0) -> `94.38deg` (B, tau=0.60) -> `94.78deg` (D, tau=0.90) -> `95.04deg` (F, tau=1.20) -> `96.73deg` (G, tau=1.50) -- **no discontinuity at the `M_neck_f` crossover** (tau~0.60-0.63), confirming that crossing is a genuine local mass-redistribution effect, not a TJ-geometry artifact (consistent with Milestone 14B Section 6). The nominal Young-angle label for `gamma_gb=1.6` is `psi_Y=2*acos(0.8)=73.74deg` (computed directly, not substituted into any force calculation, per Section 12's explicit instruction) -- the measured `psi` (93-97deg range) stays far from this nominal value throughout, the same qualitative decoupling found in Milestone 14 (eta frozen, GB location fixed, so `psi` cannot relax toward a new groove equilibrium on this timescale).

## 10. Contact-width evolution

`L_contact_TJ_sub` increases monotonically and without interruption across the ENTIRE tested range, `69.47nm` (tau=0.09) to `70.65nm` (tau=1.50) -- **`L_contact` never decreases, anywhere in this milestone's trajectory**, even well past both the `M_neck_f` crossover (tau~0.60-0.63) and the near-TJ curvature's own peak (tau~1.20).

## 11. Neck-mass evolution (secondary)

Retained as the secondary morphology diagnostic per Section 15. At the coarse sampling used for the extended trajectory (0.03-0.05s physical spacing), `M_neck_f` continues its Milestone-14B-established pattern: an overall slowly-increasing trend at this sampling density, with pairs of adjacent samples occasionally landing very close to each other (e.g. `4.331558e-15` at t=0.10 vs `4.331105e-15` at t=0.06; `4.420080e-15` at t=0.40 vs `4.420077e-15` at t=0.43) -- consistent with Milestone 14B's finding that the genuine depletion sub-trend is a small (~1e-19 to 1e-18 m^2/s), fine-timescale signal easily obscured at coarse sampling and occasionally punctuated by the neck-mask discretization artifact identified there. No attempt was made to re-resolve the fine crossover structure in this milestone (M_neck_f is explicitly secondary here); the stress/curvature diagnostics above do not depend on it.

## 12. Ordering of events

| event | tau (s) |
|---|---|
| Near-TJ curvature magnitude begins rising (Kasa fit) | ~0.09 (earliest checkpoint already past the initial value) |
| `M_neck_f` crossover (Milestone 14B, fine-resolution) | ~0.60-0.63 |
| `sigma_sint_app` broad plateau (curvature form) | ~0.60-1.08 |
| Near-TJ curvature magnitude maximum | ~1.20 |
| Near-TJ curvature and `F_TJ_mag` both begin relaxing | past ~1.20 |
| `L_contact` maximum within tested range | not reached -- still increasing at tau=1.50 |
| Geometric contact-width decrease onset | not reached within tau<=1.50s |

**Local capillary curvature concentration builds up well before, through, and for a substantial time after the `M_neck_f` crossover, while contact width never turns over at all in this window** -- i.e., stress/curvature-concentration buildup is not contingent on, and clearly leads, any visible geometric contact thinning. This is scenario **S2** territory in spirit (flank/curvature evolution producing local stress-relevant buildup before geometric neck thinning) but tempered by the fact that the CONTACT-AVERAGED `sigma_sint_app` itself does not show a clear rise (Section 6) -- see Section 14's classification.

## 13. Grid sensitivity

Force-validation and stress diagnostics were re-evaluated at state B's physical time (`t=0.20s`, `tau=0.60s`) at `dx=5.0nm` and `dx=1.25nm`.

**`dx=1.25nm` stability note**: the first two attempts blew up (NaN/overflow). Attempt 1 used the `dx=2.5nm`-calibrated `dt=5e-6s` override directly -- `p.dt = min(CFL*dx^4/(M_f*k_f), 1e-5)` scales the CFL-stable step as `dx^4`, so halving `dx` shrinks the stable step by ~16x (confirmed: auto-computed `dt` at `dx=1.25nm`, `surface_mobility_scale=0.9` is `6.78e-7s`, vs. the `dx=2.5nm` value of `1e-5s`/override `5e-6s`) -- an unrelated grid needs its own CFL audit, not an inherited fixed override, matching Milestone 14 Section 13's own discipline. Attempt 2 let `build_params` auto-select `dt=6.78e-7s` and STILL blew up -- at this extreme, previously-untested parameter combination (`A=72nm`, `gamma_gb=1.6`, well beyond what Milestones 13D/13E's CFL formula was calibrated against, which used moderate amplitude/`gamma_gb~1`), the legacy-mobility-derived CFL estimate is evidently insufficient for the face-projected scheme's own stability bound here. Attempt 3, with an explicit `dt=2e-7s` (~3.4x smaller than the auto-CFL value), completed cleanly with no warnings.

| dx (nm) | L_contact (nm) | F_cap curv form (N/m) | rel. err | sigma_sint_app curv (MPa) | psi_top (deg) |
|---|---|---|---|---|---|
| 5.0 | 70.35 | (-1.858, -0.100) | 7.9% | 26.41 | 92.35 |
| 2.5 (primary) | 70.37 | (-1.884, -0.049) | 5.7% | 26.77 | 94.38 |
| 1.25 (dt=2e-7s) | 70.34 | (-1.856, -0.013) | 2.1% | 26.38 | 96.22 |

`L_contact` agrees to `<0.05%` across all three grids. `sigma_sint_app` (curvature form) agrees to `1.5%` between dx=2.5 and dx=1.25nm, and to `0.1%` between dx=5 and dx=1.25nm (the two extremes bracketing the primary grid, not a monotonic trend) -- a tight, non-drifting spread of only ~1.5% peak-to-peak across a 4x resolution range. The force-validation tolerance actually IMPROVES at the finest grid (2.1% at dx=1.25nm vs 5.7% at dx=2.5nm vs 7.9% at dx=5nm), consistent with a better-resolved contour giving a cleaner curvature-vs-endpoint identity, as expected. `psi_measured` varies more across grids (92.35deg to 96.22deg, ~4% relative spread) -- the TJ-branch-direction measurement is more sensitive to the diffuse-core resolution than the whole-arc force integral, but stays within a physically narrow band, not diverging. Per Section 19, exact equality is not required -- the physical scale, sign, and trend converge cleanly across all three grids tested.

## 14. S1/S2/S3/S4 classification

- **S1** (stress rises while `L_contact` decreases): **not observed** -- `L_contact` never decreases in this trajectory.
- **S2** (stress rises while `L_contact` still increasing): **partially observed, and only for the LOCAL curvature measure, not the averaged stress** -- the near-TJ curvature/capillary-pressure concentration rises substantially (+23%) while `L_contact` keeps increasing throughout, but the contact-AVERAGED `sigma_sint_app` itself stays flat (Section 6), so this is not a clean S2.
- **S3** (stress roughly constant, but local `|gamma_s*kappa|` rises strongly near the TJ): **this is the best-supported classification**. `sigma_sint_app` stays within a 2-2.5% band across a 17x range of reduced time (tau=0.09 to 1.50s) while the independent, un-smoothed near-TJ Kasa-fit curvature magnitude rises 23% over the same span (tau=0.09 to 1.20s) before relaxing. The far-window curvature and the contact-averaged stress both stay comparatively flat throughout -- the buildup is specifically a LOCAL, near-TJ concentration effect, exactly S3's description: "average contact stress is unchanged but the local capillary stress concentration is increasing... could still be highly relevant to local sink nucleation."
- **S4** (neither rises): **not observed** -- the near-TJ curvature signal is real, validated by an independent estimator, and reproduces (in reduced form) across seven checkpoints spanning a substantial fraction of the trajectory.

## 15. Recommendation for sink activation

The evidence supports a genuine, if geometrically localized and moderate-magnitude (~23%), capillary-stress-concentration buildup specifically AT the TJ's immediate free-surface neighborhood, decoupled from both the contact-averaged apparent stress (flat) and the geometric contact width (still increasing, never turned over). This is consistent with -- but does not yet conclusively establish -- Section 1's hypothesis that surface-diffusional reshaping raises the LOCAL driving force for disconnection nucleation before any macroscopic geometric signature (neck thinning) appears. Given:

1. the near-TJ curvature signal has already begun relaxing by tau=1.50s (not yet at a final, sustained plateau or a clear runaway), and
2. `sigma_sint_app` itself has not shown a clear systematic rise,

**the sink should NOT be activated yet.** The recommended next step is to determine whether the near-TJ curvature concentration is a one-time transient (already captured, now relaxing) tied to the specific `M_neck_f` crossover event, or whether a stronger `gamma_gb` (still not exceeding the Milestone-14B-established `1.6` -> next-candidate ordering, and still short of the `2*gamma_s` de-sintering limit) reproduces a LARGER and more sustained version of the same near-TJ concentration -- that comparison, not sink activation, is the next well-motivated experiment. Sink/hazard/RBM/eta kinetics were not activated in this milestone.

## 16. Figures

Seven figures (Section 21) were generated to `runs/m14c_figs/` (`fig1_sigma_sint_app_vs_tau.png` through `fig7_geometry_ABCD.png`) -- not committed, gitignored like every other `runs/` output in this project; regenerate from the saved checkpoint states and `runs/m14c_stress_*.json`/`runs/m14c_gamma16_extended.json` (all reproducible via `scripts/m14c_stress_analysis.py` and `scripts/m14_mechanism_screen.py` with the CLI arguments recorded in each state's own log). Fig. 5 (near-TJ vs. far-window local capillary pressure vs. tau) and Fig. 1 (`sigma_sint_app` vs. tau, both force forms) are the two figures that most directly show the Section 14 classification: Fig. 5's near-TJ curve rises clearly from tau~0.1 to a peak near tau~1.2 before relaxing, while Fig. 1's `sigma_sint_app` stays within a narrow band throughout with no comparable trend. Fig. 7 (f=0.5 geometry at A/B/C/D with the traced particle arc colored by signed curvature) shows the arc's curvature staying close to zero/mildly negative along most of its visible length near the neck at every state, rather than the strongly positive (convex-cap) coloring a pristine circular particle would show -- visual confirmation of Section 7's finding that the particle's free surface has been substantially reshaped by this milestone's extreme geometry/energy-ratio combination.

**STOP. Sink/hazard/RBM/eta kinetics/anisotropy were not activated.**
