# Two-times-native odd-step buffer reuse

The production event uses twice the native timestep with five native updates
per physical diagnostic block.  The reusable native kernel now supports this
odd step count by returning the scratch-side state and rotating the prior
input arrays into the next scratch set.

The unit regression matches the allocating and reusable kernels exactly for
five updates.  The event-level comparison in
`odd_buffer_reuse_2x_overlap.json` advances event 2's immutable origin by
`0.0025b`.  Both paths converge in 135 fast blocks, with zero difference in
the four fields, active local stress, transport clock, and total material
volume.  Convergence tolerances, physics, and stochastic state are unchanged.

The short comparison shows no material runtime change.  This is promoted for
allocation stability and scratch-memory reuse during long late-event
relaxations, without claiming a timestep or solver acceleration.
