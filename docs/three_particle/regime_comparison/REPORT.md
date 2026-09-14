# Bicrystal–particle-cluster loading/nucleation regime comparison

This is deterministic postprocessing of the completed sharp-interface tables and the existing bicrystal production history. No PF state was initialized or advanced, no threshold was drawn, and no production parameter was changed.

## Result

All 10 fully thermodynamic cluster paths remain in the near-zero-loading regime at median nucleation: Δσ50 is 0.0040–0.0430 MPa and AΓ,50 is 0.999395–1.000076.
Moving the median root to 1% loss requires M_s/M_s0=2.82–29.54; moving it to 10% requires 33.9–374.6. These changes buy at most a few MPa because the thermodynamically allowed stress slopes are small.

The two completed bicrystal cycles show visible load/relax morphology cycles, but the unchanged root rate changes only weakly during loading. On the requested Δσ–AΓ map they do not occupy a large-hazard-amplification loading-triggered regime. Cycle 2 nucleated before its trajectory accumulated median hazard, so its median point is censored and its observed root is plotted explicitly.

The evidence therefore supports a robust particle-cluster regime of stochastic nucleation during coarsening before appreciable additional loading. It supports bicrystal load/relax cycles, but does not support the stronger claim that their root nucleation is stress-loading-controlled under the shared unchanged microscopic rate law.

## Definitions

Cluster Λ=(1/1.25)∫(ΓL+ΓR)/xdot dx and S=exp(-Λ). Mobility scaling uses Λ_m=Λ_0/m_M, so m_M=Λ_0(x_target)/ln2. The same multiplier is the equivalent root-rate reduction. Site-only suppression leaves 1/m of the effective sites; barrier-only suppression requires ΔG_root=kBT ln(m).

The bicrystal coordinate is x_B=1-(r_neck/r_neck,post)^2, the fractional contact-area loss. The final high-stress portion begins at 80% of the observed post-avalanche-to-pre-root stress rise.

## Cluster quantile ranges

|root probability|center loss|Lambda|Delta sigma (MPa)|A_Gamma|
|---:|---:|---:|---:|---:|
|10%|0.0052-0.0547%|0.105361|0.0006-0.0065|0.999908-1.000011|
|50%|0.0345-0.3585%|0.693147|0.0040-0.0430|0.999395-1.000076|
|90%|0.1144-1.1756%|2.302585|0.0131-0.1422|0.998010-1.000251|

## Mobility and equivalent root-process sensitivity

|target loss|M_s/M_s0|site-count reduction|Delta G_root (eV)|Delta sigma (MPa)|A_Gamma|
|---:|---:|---:|---:|---:|---:|
|1%|2.82-29.54|64.5-96.6%|0.163-0.534|0.060-0.255|0.997141-1.000453|
|2%|5.73-60.31|82.5-98.3%|0.275-0.647|0.121-0.515|0.994241-1.000921|
|5%|15.13-161.51|93.4-99.4%|0.428-0.802|0.310-1.319|0.985287-1.002423|
|10%|33.94-374.58|97.1-99.7%|0.556-0.935|0.642-2.751|0.969452-1.005283|

## Completed bicrystal cycles

|cycle|post sigma (MPa)|pre-root sigma (MPa)|Delta sigma (MPa)|post rate (/s)|pre-root rate (/s)|rate amplification|duration (s)|x_B at root|hazard in final high-stress portion|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|55.707|57.152|1.445|0.07645|0.07318|0.9573|13.826|13.672%|11.1%|
|2|53.112|55.879|2.768|0.06982|0.07178|1.0281|4.644|5.955%|33.6%|

Cycle 1 starts from the initialized relaxed production state; cycle 2 starts from the preceding avalanche extinction. Cycle 3 is still right-censored and is excluded from completed-cycle conclusions.

## Maximum thermodynamic stress loading in the saved full search

|loss|full-search maximum Delta sigma (MPa)|design|eligible paths|maximum among fully 20%-thermodynamic paths (MPa)|
|---:|---:|---|---:|---:|
|1%|0.906|`Ro120_q0.85_L8W_r1.00_th140_km-30`|355|0.255|
|2%|1.439|`Ro120_q0.75_L8W_r1.20_th160_km-30`|322|0.515|
|5%|3.620|`Ro120_q0.85_L8W_r1.20_th170_km-20`|212|1.319|
|10%|3.853|`Ro80_q0.75_L8W_r1.40_th170_km-20`|87|2.751|

This full-search bound applies dF/dx<0 at every stored search knot through each target. Several MPa first becomes possible at 5% loss in this checkpoint-qualified sense. Among paths favorable through the full 20% interval, the maximum is only 1.319 MPa at 5% and reaches 2.751 MPa at 10%. No path builds several MPa by 1–2% loss.

## Files

- `cluster_quantile_competition.csv`: Λ, S, Aσ, AΓ, resolution and time at x10/x50/x90 for all ten paths.
- `mobility_and_root_sensitivity.csv`: mobility multiplier, equivalent site suppression and barrier increase at 1/2/5/10% loss.
- `bicrystal_cycles.csv` and `bicrystal_loading_paths.csv`: completed-cycle endpoint and path metrics.
- `maximum_thermodynamic_loading.csv`: full-search stress-gain envelope.
- `largest_thermodynamic_stress_slopes.csv`: twenty largest positive saved-segment dσ/dx values with favorable prefixes.
- `regime_map_points.csv`: common Δσ50–AΓ,50 comparison, with censoring basis explicit.

## Scope

The cluster projection remains a one-coordinate sharp-interface approximation. The full-search envelope uses the already saved 0/1/2/5/10/15/20% knots rather than rerunning geometry. Bicrystal results cover two completed cycles from one production realization; cycle 2's analytical median lies beyond its observed early root and is not extrapolated.
