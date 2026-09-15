# Mapped-PF initial-state audit and events-off screen

Status: **COMPLETE; NO STOCHASTIC CAMPAIGN LAUNCHED**.

## Sharp-to-PF discrepancy

For C1, stress changes from 39.7902 MPa analytically to 54.6987 MPa under exact production metrology on the mapped field, 54.8269 MPa after cleanup, and 58.3246 MPa after the 0.01 s events-off qualification.

The offsets are 14.9085 MPa from finite-window mapped-PF metrology, 0.1283 MPa from cleanup, and 3.4976 MPa from actual PF evolution. At the qualified source the curvature term is 18.2981 MPa, compared with zero in the pointwise sharp construction; the TJ term changes by only 0.2362 MPa. The discrepancy is therefore a curvature-correspondence error: the production 3W fit samples the rapidly varying polynomial contour near the contact, whereas the selector used its pointwise contact jet.

Only C1 has a previously created qualified-source archive. C2-C5 were compared at the sharp, mapped, and five-step-cleanup stages and then evolved directly with the same events-off PF equations; no qualification stage is inferred for them.

## Completed-trajectory decomposition

The original 1.2796 MPa rise contains +1.7879 MPa from fitted curvature and -0.5082 MPa from the TJ term. After the maximum, curvature changes by -5.1649 MPa and the TJ term by -2.7363 MPa. Reversal of fitted mean meridional curvature controls the turnover; growing TJ radius and falling angle sum reinforce the later relaxation.

## Events-off mapped candidates

|candidate|post-cleanup sigma|screen/extended maximum|Delta sigma|time|survival|Delta ln Gamma barrier|Delta ln Gamma sites|
|---|---:|---:|---:|---:|---:|---:|---:|
|C1|54.827|59.604|4.777|1.01|0.8374|0.0977|0.0047|
|C2|43.154|48.126|4.972|0.05|0.9943|0.1173|-0.0001|
|C3|43.812|48.769|4.957|0.05|0.9943|0.1159|-0.0000|
|C4|64.694|69.643|4.949|0.05|0.9928|0.0914|0.0010|
|C5|70.706|75.525|4.818|0.00145|0.9998|0.0843|0.0025|

C2 and C3 remain endpoint-censored at 0.05 s in the common screen. C4 begins too highly stressed, and C5 turns over almost immediately.

## Recommendation

Recommend C2: `Ro150_q0.55_L45W_r0.75_th160_kc+10_ko+10`. Its ordinary events-off PF evolution reaches 53.1835 MPa at the 2 s cap from 43.1543 MPa, so its demonstrated loading headroom is at least 10.0292 MPa. Survival to that state is 0.7821. Barrier amplification (0.22899) exceeds the magnitude of site loss (0.01647) by a factor of 13.90. The maximum remains endpoint-censored.

Over the extension, the LEFT curvature term changes from -4.6748 to 4.4043 MPa and the TJ term from 47.8290 to 48.7792 MPa. The two fitted curvatures change from 2.6223/6.7273 to -5.8963/-2.9123 per micrometre; angles change from 162.247/161.204 to 161.160/167.889 degrees. Contact radius falls 1.633%, center volume falls 0.650%, and the per-contact root rate rises by 23.679%.

Surface area, GB area, and interfacial energy change by -0.1759%, -3.2401%, and -0.1596% respectively; total mass remains conserved to the reported solver tolerance.

This recommendation is for review only. Events and RNG remained disabled, no trajectory was prescribed, and no stochastic renewal campaign was started.
