# Initial energy-conjugate stress

For the independent `gb_spacing` perturbation, the established virtual-work diagnostic is evaluated as `F_sint = -(dE/d ln Lc)/Lc` and `sigma_energy = F_sint/(pi r_TJ^2)`. Moving both contacts by half the spacing increment gives total work `F_sint dLc`. This diagnostic does not drive the PF evolution.

|design|energy-conjugate stress (MPa)|exact local activation stress (MPa)|
|---|---:|---:|
|`Ro180_q0.55_L35W_r0.75_th160_kc+0_ko+0`|16.2788|39.7902|
|`Ro150_q0.55_L45W_r0.75_th160_kc+10_ko+10`|-16.4585|37.7483|
|`Ro180_q0.45_L45W_r0.75_th160_kc+10_ko+10`|-16.7793|38.6325|
|`Ro150_q0.45_L30W_r0.75_th160_kc+0_ko+0`|-11.286|58.3590|
|`Ro120_q0.55_L30W_r0.75_th160_kc-10_ko-10`|-12.4445|69.6853|
