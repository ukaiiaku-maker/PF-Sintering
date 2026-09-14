# Event-9 far-contact tangent qualification

Event 9 stopped at `q/b = 0.5625` because the integral-stress diagnostic
changed by more than its unchanged `0.10 MPa` per-step limit. The field,
ownership, mass, topology, local stress, and transport affinity remained
well behaved. Binary refinement showed that the apparent integral-stress
jump approached about `1.106 MPa` as the transfer increment approached zero.

The jump came from the center free-surface branch ending at the other field
TJ. Its far tangent was the last grid chord, so the tangent changed abruptly
when that endpoint crossed a z row. The active-TJ tangent already uses a
one-sided local fit for exactly this continuity reason. A branch terminated
by another field TJ now uses the same one-sided fit at its far endpoint.
Complete bicrystal branches retain their original resolved far tangent.

The exact retained Event-9 stop-frame audit in
`event9_far_endpoint_tangent_continuity.json` measures a `1.09623941 MPa`
legacy last-chord change for the rejected `0.0025b` trial. The fitted
far-contact tangent gives `0.04744810 MPa`, below the unchanged limit. The
same trial then completes with local-stress change `0.00334408 MPa`, bounded
fields, ownership closure `2.22e-16`, total-volume relative error
`-2.22e-16`, and no topology stop.

This is a metrology continuity correction. It does not change the local
stress or root-law definitions, any material or kinetic parameter, the
stored stochastic state, or the phase-field update. There is no clipping,
fitted physics correction, symmetry projection, or acceptance-limit
relaxation.
