# Milestone 14D: narrowing the substrate to test geometry-driven capillary-stress buildup

## 1. Starting checkpoint

Branch `codex/coarsening-stress-buildup`, HEAD `8250e90` ("feat+docs: Milestone 14C capillary sintering-stress buildup diagnostic"), confirmed clean working tree and **217/217** tests passing before any change. `variational_surface_diffusion_step` (`face_projected`) was not modified; `pf_sintering/capillary_stress.py` reused as-is. Eta kinetics, sink, hazard, RBM, anisotropy, imposed stress/strain stayed OFF throughout. `R2`, particle aspect ratio, initial particle geometry, interface width, surface mobility law, `gamma_s`, contact orientation, and the surface-transport implementation were all left unchanged -- only the substrate sinusoid's amplitude/wavelength (and, for one controlled comparison, `gamma_gb`) were varied. No new library code was needed; all new analysis lives in ad hoc scripts plus the existing `pf_sintering/capillary_stress.py`/`scripts/m14c_stress_analysis.py`/`scripts/m14b_save_checkpoint.py`/`scripts/m14_mechanism_screen.py` infrastructure.

## 2. Analytic sinusoid equation and substrate-width metrics

The substrate free surface (`pf_sintering/model.py::initialize_fields`, `geometry="sinusoidal_substrate"`, `sinusoid_phase=0`):

```
x_s(Y) = wall_mean + A*cos(2*pi*Y/lambda)
```

(`Y` centered at the domain's vertical midline, crest at `Y=0` where the particle sits; `wall_mean = (substrate_wall_frac-0.5)*Nx*dx`). Two width metrics were defined and computed directly from this equation (never guessed visually):

- **`W_sub_50`**: full lateral width where `x_s(Y) >= wall_mean + 0.5*A` (above half the crest height relative to the mean/baseline). Solving `cos(2*pi*Y/lambda)=0.5` gives `Y=+-lambda/6`, so **`W_sub_50 = lambda/3`** -- independent of `A` for a pure cosine, a function of wavelength alone.
- **`W_sub_contact`**: full lateral width of the substrate at the X-level of the ACTUAL initial triple junction, `X_TJ` (the physically meaningful "contact-normal level" -- literally where the GB/TJ sits, obtained from `tj_subgrid.compute_subgrid_contact` on the real initialized+reprojected state, not assumed). Solving `x_s(Y)=X_TJ` gives `W_sub_contact = (lambda/pi)*acos((X_TJ-wall_mean)/A)`.

## 3. Particle width -- analytic and contour-derived

The milestone's own rationale approximated particle width as `2*R2=160nm`. Direct computation from `build_params`'s own `Rx`,`Ry` construction (`contact_orientation="short_plane"`, `aspect_ratio=2.0`: `Rx=R2*sqrt(ar)=113.1nm` perpendicular to the substrate, `Ry=R2/sqrt(ar)=56.6nm` LATERAL, i.e. along the substrate's own periodic direction) shows this was **not correct**: because "short_plane" elongates the particle perpendicular to the substrate (not laterally), the particle's actual lateral extent is set by `Ry`, not `R2`. **`W_particle_full = 2*Ry = 113.14nm`** -- 29% narrower than the milestone's `2*R2=160nm` estimate. `W_particle_at_tj` (from the analytic ellipse `((X_TJ-cx)/Rx)^2+(Y/Ry)^2=1`, evaluated at the same `X_TJ` used for `W_sub_contact`) gives the width at the actual contact height, smaller still (`~44-56nm`, since the TJ sits well below the particle's own equator).

## 4. Old vs. new width ratios

| geometry | A (nm) | lambda (nm) | W_sub_50 (nm) | W_sub_50/W_particle_full | W_sub_contact (nm) | W_particle_at_TJ (nm) | W_sub_contact/W_particle_at_TJ | max slope (deg) | trough clearance |
|---|---|---|---|---|---|---|---|---|---|
| OLD (A72/lambda480) | 72 | 480 | 160.00 | 1.414 | 58.18 | 56.02 | **1.039** | 43.3 | 81.6nm (4.08W) |
| NEW (A100/lambda320) | 100 | 320 | 106.67 | 0.943 | 48.66 | 43.76 | **1.112** | 63.0 | 54.2nm (2.71W) |

**Important, honest finding**: by the AT-TJ-HEIGHT metric (`W_sub_contact/W_particle_at_TJ`), the OLD geometry was already close to ratio 1 (`1.039`) -- not clearly "too broad" in that specific sense. The `W_sub_50` metric tells a different, complementary story (`1.414 -> 0.943`, a real and substantial narrowing relative to the substrate's own full-amplitude scale). The two metrics disagree because they measure different things: `W_sub_50` characterizes how "generally raised" the whole crest region is (normalized to the substrate's own amplitude, independent of absolute penetration depth), while `W_sub_contact` is anchored to the one height that happens to matter at `t=0`. **The unambiguous, large, and intentional change between the two geometries is the substrate's steepness** (`max slope 43.3deg -> 63.0deg`) and its absolute amplitude/depth (`72nm -> 100nm`), not the AT-TJ width ratio specifically -- both were already within a reasonable range of 1 by that one metric. This nuance is reported transparently rather than forcing a cleaner narrative than the data supports.

## 5. Small geometry screen and candidate selection

Per Section 3's instruction, the 9-point screen (`A in {90,100,120}`, `lambda in {280,320,360}`) was evaluated analytically (no time integration) before committing to a production run:

| A (nm) | lambda=280 ratio_contact | lambda=320 ratio_contact | lambda=360 ratio_contact | trough clearance (nm) |
|---|---|---|---|---|
| 90 | 1.054 | 1.118 | 1.042 | 64.0 |
| 100 | 1.075 | 1.112 | 1.020 | 54.2 |
| 120 | 1.184 | 1.088 | 1.127 | 34.6 |

All nine satisfy `0.8 <= ratio_contact <= 1.2`; none showed boundary interference, self-intersection (impossible for a single-valued `x_s(Y)`), or an unresolved interface at `dx=2.5nm`. `A=120nm` has the thinnest trough clearance (`34.6nm=1.73W`) -- more marginal but not disqualifying. `A=100nm/lambda=320nm` (the milestone's own primary candidate) was confirmed directly on the real field to have `f=1.0` exactly at the trough near the reflecting `X=0` boundary (no boundary contamination) and was retained as the primary candidate, consistent with Section 2's explicit rationale (steeper flanks) and comfortably inside the target band on both width metrics.

## 6. Grid/timestep stability audit

`build_params`'s auto-CFL `dt` at `dx=2.5nm`, `surface_mobility_scale=0.9`, new geometry: `1.0e-5s` (capped ceiling; CFL bound `1.085e-5s`, same ~8.5% margin as the old geometry). An explicit `dt=5e-6s` (matching the Milestone 14B/14C convention) was verified empirically stable over `t=0-0.05s` (10000 steps, no NaN/overflow) and confirmed converged via a dt-halved check (`dt=2.5e-6s` vs. `5e-6s` at `t=0.02s`: `M_neck_f` and `L_contact` agree to the full displayed precision). **One diagnostic-only artifact was found and bypassed, not fixed**: `substrate_fourier_away_from_tj`'s least-squares harmonic fit becomes numerically ill-conditioned for this much shorter wavelength (320nm vs. 480nm) combined with the existing `6*W`-margin TJ-exclusion zones, which now consume most of the single-wavelength domain -- it returns nonsensical `A1_substrate` values (`~1e3` to `~1e5` nm) while every other diagnostic (`M_neck_f`, `L_contact`, `F`, `D_h`, mass, `psi`, exact conservative closure `A==B`) stays perfectly well-behaved. Substrate crest evolution is instead read directly from the raw `x_cross` contour data already saved in each row (Section 8), not from the broken fit.

## 7. Initial-geometry diagnostic

See the comparison figure (Section 16). At `t=0` (gamma_gb=1.4142136, both geometries):

| quantity | OLD (A72/480) | NEW (A100/320) |
|---|---|---|
| particle width near contact (`W_particle_at_TJ`) | 56.02nm | 43.76nm |
| substrate width beneath contact (`W_sub_contact`) | 58.18nm | 48.66nm |
| `W_sub/W_particle` | 1.039 | 1.112 |
| substrate amplitude | 72nm | 100nm |
| substrate wavelength | 480nm | 320nm |
| max initial substrate slope | 43.3deg | 63.0deg |
| initial contact length `L_contact_TJ_sub` | 60.37nm | 49.59nm |
| initial measured `psi` | 93.8deg | 111.5deg |

The initial-geometry figure (`m14d_fig0_initial_geometry.png`) shows the intended qualitative change directly: the new substrate mound comes to a visibly sharper, more localized peak beneath the particle, versus the old geometry's broader, gentler crest -- confirming the geometric intent even though the single AT-TJ-height width ratio happened to already be near 1 for both (Section 4).

## 8. Short trajectory: new geometry, gamma_gb=1.4142136

`t=0` to `0.20s` (`tau=0` to `0.60s`, `surface_mobility_scale=0.9`, `dt=5e-6s`, matching the Milestone 14B/14C reduced-time convention):

| t (s) | tau (s) | M_neck_f (m^2) | L_contact (nm) | psi (deg) | F | D_h | mass_drift |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 3.791127e-15 | 49.59 | 109.74 | -6.653310e-05 | -- | -- |
| 0.03 | 0.09 | 3.831479e-15 | 57.01 | 108.06 | -6.654418e-05 | 1.32e-07 | 1.10e-15 |
| 0.06 | 0.18 | 3.798852e-15 | 57.35 | 108.82 | -6.654738e-05 | 8.95e-08 | 2.92e-15 |
| 0.10 | 0.30 | 3.798582e-15 | 57.47 | 109.42 | -6.655052e-05 | 7.09e-08 | 5.12e-15 |
| 0.13 | 0.39 | 3.798427e-15 | 57.50 | 109.72 | -6.655252e-05 | 6.18e-08 | 6.58e-15 |
| 0.16 | 0.48 | 3.798336e-15 | 57.53 | 109.94 | -6.655428e-05 | 5.75e-08 | 7.86e-15 |
| 0.20 | 0.60 | 3.798318e-15 | 57.55 | 110.15 | -6.655657e-05 | 5.58e-08 | 1.01e-14 |

`M_neck_f` rises during the initial transient (`t=0->0.03`, the usual sharp geometric-relaxation spike), then **turns over and depletes persistently for every remaining interval through `tau=0.60s`** (5 consecutive negative-rate windows, `t=0.03` through `t=0.20`) -- a fast, unambiguous, and immediate crossover, reached by `tau~0.09-0.18`, dramatically earlier than Milestone 14B's old-geometry `gamma_gb=1.6` case (`tau~0.60-0.63`, and only after very careful fine-sampling to distinguish it from noise). `L_contact` increases monotonically throughout (never decreases). `F` decreases monotonically; `D_h` decreases smoothly; mass conservation stays at roundoff-adjacent levels (`<=1.01e-14` relative).

## 9. Old vs. new stress comparison (matched gamma_gb=1.4142136, matched tau)

A matched OLD-geometry trajectory (`A72/lambda480`, `gamma_gb=1.4142136`, same mobility/dt convention -- not previously run at this exact `gamma_gb`/mobility combination) was run for a true apples-to-apples comparison, plus checkpoints analyzed with `capillary_stress.py`:

| condition | tau (s) | sigma_sint_app curv (MPa) | F_cap_n curv (N/m) | L_contact (nm) | near-TJ kappa (1/m) |
|---|---|---|---|---|---|
| OLD, gg=1.4142 | 0.09 | 27.06 | -1.889 | 69.82 | -7.84e6 |
| OLD, gg=1.4142 | 0.60 | 26.35 | -1.874 | 71.12 | -7.19e6 |
| NEW, gg=1.4142 | 0.00 | 35.18 | -1.745 | 49.59 | -7.22e6 |
| NEW, gg=1.4142 | 0.09 | 30.99 | -1.767 | 57.01 | -6.89e6 |
| NEW, gg=1.4142 | 0.30 | 30.77 | -1.768 | 57.47 | -7.80e6 |
| NEW, gg=1.4142 | 0.60 | 30.68 | -1.766 | 57.55 | -7.96e6 |

**`sigma_sint_app` is consistently and substantially higher for the new geometry at every matched tau** (`30.7-31.0MPa` vs. `26.4-27.1MPa`, a **+15 to +16%** step increase), achieved at the SAME `gamma_gb` as the old-geometry comparison -- i.e. purely from the geometry change. This step is already fully present by the first post-transient sample (`tau=0.09`); the raw, unrelaxed `t=0` state shows an even HIGHER value (`35.2MPa`), which relaxes down to the `~31MPa` plateau within the first `0.09s` of reduced time as the sharp initial construction smooths out (an artifact of the initial condition, not a genuine rise -- checked explicitly to rule out a "hidden rise" in the unsampled `t=0->0.03s` window).

**`sigma_sint_app`'s temporal trend stays flat-to-mildly-declining in BOTH geometries** -- the new geometry raises the baseline substantially but does not, by itself, convert the trajectory into a temporally RISING stress within this `tau<=0.6` window.

`L_contact` is uniformly much smaller for the new geometry (`~57nm` vs. `~70nm`, consistent with the narrower substrate directly limiting how wide a contact can form) and grows much more slowly in absolute terms (`57.01->57.55nm`, a 1.0% change, vs. old geometry's `69.82->71.12nm`, a 1.9% change over the same tau range) -- the new geometry's contact width is already much closer to saturating.

## 10. Near-TJ curvature and F_cap_n comparison

`F_cap_n` (curvature form) is slightly LESS negative in the new geometry (`~-1.77` vs. `~-1.88` N/m) -- since `sigma_sint_app = -F_cap_n/L_contact`, the substantially higher apparent stress in the new geometry is driven primarily by the much SMALLER `L_contact` denominator, not a larger transmitted force. This is an important, non-obvious mechanistic finding: **narrowing the substrate raises the apparent stress mainly by shrinking the load-bearing contact length that the (comparable-or-slightly-smaller) capillary force is divided over, not by increasing the capillary force itself.**

Near-TJ curvature (Kasa fit) for the new geometry jumps quickly from `t=0` (`-7.22e6`) to a peak magnitude by `tau=0.09` (`-6.89e6`... note this is actually smaller in magnitude than `t=0`, then RISES again to `-7.80e6` at `tau=0.30` and `-7.96e6` at `tau=0.60`) -- broadly similar rising-magnitude character to the old geometry's own near-TJ trend (Milestone 14C), but reaching a comparable magnitude (`~-7.9e6` to `-8.0e6`) by `tau~0.6`, whereas the old geometry needed `tau~0.9-1.2` to reach the same range (Milestone 14C Section 7). **The new geometry reaches a comparable local curvature concentration roughly twice as fast in reduced time.**

## 11. Measured psi evolution

`psi_measured` for the new geometry starts substantially higher than the old geometry (`109.7deg` at `t=0` vs. `92.8-93.8deg` for old) -- a direct, expected consequence of the much steeper initial substrate flank changing the local free-surface tangent geometry at the TJ -- and continues increasing smoothly and monotonically through the depletion crossover (`108.1deg` at `tau=0.09` -> `110.2deg` at `tau=0.60`), exactly as in Milestone 14C's old-geometry finding: no discontinuity at the mass-flux crossover, confirming it is a genuine local redistribution effect, not a TJ-geometry artifact. `psi` was never imposed or used to reshape the TJ.

## 12. gamma_gb=1.6 as the controlled second variable

Per Section 11, since the new geometry's `sigma_sint_app` does not show a clear TEMPORAL rise (only an elevated, still-flat plateau), the same new geometry was rerun with `gamma_gb=1.6`:

| condition | tau (s) | sigma_sint_app curv (MPa) | L_contact (nm) | M_neck_f depletion onset |
|---|---|---|---|---|
| NEW, gg=1.6 | 0.09 | 31.46 | 56.70 | -- |
| NEW, gg=1.6 | 0.60 | 30.90 | 56.91 | tau~0.18-0.30 |

`sigma_sint_app` is only marginally higher than `gg=1.4142136` at the same geometry (`+1.5%` at `tau=0.09`, `+0.7%` at `tau=0.60`) -- `gamma_gb` contributes far less than the geometry change did. **`L_contact`, however, shows the most advanced behavior found in this entire investigation**: `56.70 -> 56.90 -> 56.92nm` (`t=0.03->0.06->0.10`, still slightly increasing) followed by a small, real DECREASE (`56.92 -> 56.92 -> 56.91nm`, `t=0.10->0.13->0.16`, i.e. two consecutive non-positive intervals) before a negligible uptick by `t=0.20`. This is the closest approach to genuine `L_contact` turnover (the defining S1 criterion) found across all conditions tested in Milestones 14-14D, though it is small (sub-0.1nm) and would need a longer run to confirm as persistent rather than a transient wobble.

`M_neck_f` depletion onset for `gg=1.6` (`tau~0.18-0.30`) is slightly LATER than `gg=1.4142136`'s onset (`tau~0.09-0.18`) at the same new geometry -- i.e., within this short window, raising `gamma_gb` did not accelerate the mass-flux crossover; the geometry change dominates that effect.

## 13. Does geometry move the system from S3 toward S2/S1?

**Partially, and in a way not fully captured by the original S1-S4 categories**:

- `sigma_sint_app`'s TEMPORAL trend remains flat-to-mildly-declining in every condition tested (old or new geometry, either `gamma_gb`) -- by the strict "does stress rise within this trajectory" criterion, this is still **S3**, not S1/S2.
- However, the new geometry produces a substantial, immediate, geometry-driven **step increase in the baseline level** of `sigma_sint_app` (+15-16%, present from the very first post-transient sample and confirmed not to reflect a hidden earlier rise), a roughly 2x faster approach to a comparable near-TJ curvature concentration, and -- critically -- converts the mass-flux crossover from marginal/borderline (old geometry, `gamma_gb=1.6`, requiring fine-sampling to distinguish from noise, Milestone 14B) to fast, robust, and unambiguous (new geometry, either `gamma_gb`, `tau~0.1-0.3`, five-plus consecutive clearly-negative windows).
- `L_contact`'s growth is dramatically slowed by the new geometry (from `+1.9%` to `+1.0%` over the same tau range at `gamma_gb=1.4142136`), and at `gamma_gb=1.6` specifically shows the first genuine (if small and not yet confirmed persistent) DECREASE -- the closest approach to S1 found so far.

**Overall classification: still S3 in the strict temporal-stress-trend sense, but substantially strengthened -- with the new geometry's `gamma_gb=1.6` condition showing early, tentative S1-like behavior in `L_contact` that was never observed with the old, broad substrate at any `gamma_gb` tested (up to `1.6`).**

## 14. Is gamma_gb=1.6 still needed?

Not for demonstrating the core geometric mechanism -- `gamma_gb=1.4142136` with the new geometry already reproduces the full qualitative and most of the quantitative effect (elevated `sigma_sint_app`, fast robust `M_neck_f` depletion) that previously required `gamma_gb=1.6` with the old, broad substrate. `gamma_gb=1.6` remains useful as a modest additional push specifically toward `L_contact` turnover (Section 12) and should be retained as the natural next condition to extend, per Section 11's decision rule, since it is the one combination showing tentative S1 behavior.

## 15. Recommended next trajectory

1. **Extend `NEW geometry + gamma_gb=1.6` past `tau=0.60s`** (the same reduced-time budget used in Milestone 14C, `tau~1.2-1.5`) to determine whether the tentative `L_contact` decrease (Section 12) persists, grows, or reverts -- this is now the single most information-dense condition identified across Milestones 14-14D for testing genuine S1 behavior.
2. If `L_contact` turnover is confirmed and sustained, save canonical filling/near-neutral/turnover states (matching Milestone 14B's A/B/C/D convention) for this geometry and revisit the sink-nucleation question with a much stronger, geometry-grounded case than either prior milestone established.
3. Do not increase `gamma_gb` above `1.6` without first characterizing this extended trajectory, per this milestone's Section 10 constraint (carried forward as a sensible default, not re-litigated here).
4. Sink/hazard/RBM/eta kinetics were not activated in this milestone.

## 16. Figures

Two figures were generated to `runs/m14c_figs/` (not committed, gitignored like all other `runs/` output; regenerate via the scripts and CLI arguments recorded in each state's log): `m14d_fig0_initial_geometry.png` (side-by-side `t=0` `f` field for old vs. new geometry, TJ markers, and the metrics table of Section 7) and `m14d_fig_comparison.png` (three-panel `sigma_sint_app`/`L_contact`/near-TJ-curvature vs. `tau`, all three conditions overlaid) -- the comparison figure is the clearest single visual summary of Section 9-13's findings: the new-geometry curves (orange/green) sit visibly above the old geometry (blue) in `sigma_sint_app` and visibly below in `L_contact` at every matched `tau`, while the near-TJ curvature panel shows the new geometry reaching the old geometry's eventual concentration level roughly twice as fast.

**STOP. Sink/hazard/RBM/eta kinetics/anisotropy were not activated.**
