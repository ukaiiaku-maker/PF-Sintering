# Earlier-stage initial-state design

This screen selects an initial geometry only. It contains no prescribed trajectory, kinetic projection, PF evolution, or stochastic draw.

450 of 729 cases are resolved through a conservative 10% topology probe and possess at least one independent local direction with dE/dq<0 and d sigma_local/dq>0.

|role|design|sigma0 (MPa)|two-contact clock (s)|rTJ/W|resolution margin|loading directions|best coordinate|
|---|---|---:|---:|---:|---:|---:|---|
|C1 (chosen)|`Ro180_q0.55_L35W_r0.75_th160_kc+0_ko+0`|39.79|8.87|18.6|2.82|2|center_curvature|
|C2|`Ro150_q0.55_L45W_r0.75_th160_kc+10_ko+10`|37.75|11.23|15.5|2.44|2|outer_curvature|
|C3|`Ro180_q0.45_L45W_r0.75_th160_kc+10_ko+10`|38.63|11.17|15.2|2.39|2|outer_curvature|
|C4|`Ro150_q0.45_L30W_r0.75_th160_kc+0_ko+0`|58.36|8.54|12.7|1.97|2|center_curvature|
|C5|`Ro120_q0.55_L30W_r0.75_th160_kc-10_ko-10`|69.69|7.04|12.4|1.92|2|center_curvature|

Chosen initial state: `Ro180_q0.55_L35W_r0.75_th160_kc+0_ko+0`. It is selected before mapping and before any stochastic seed is used.

The local derivatives demonstrate accessible headroom only. They neither predict nor impose the direction, rate, or magnitude of subsequent evolution. The exact bicrystal stress and root law are used unchanged.
