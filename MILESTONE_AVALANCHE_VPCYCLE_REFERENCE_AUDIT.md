# Avalanche `Vp_over_Vp_cycle` bookkeeping audit

## Finding

The frozen stochastic production driver defines the volume-validity ratio
relative to the particle volume at the beginning of each post-event waiting
cycle. `wait_or_censor()` executes `vp0 = integral(state[1], setup)` every time
it is entered after a completed one-b event.

The first fixed-decrement avalanche trajectory instead calculated
`vp_cycle0` once at root loading and passed that same reference through all
five descendant correlation windows. Its reported value of
`Vp_over_Vp_cycle = 0.8791357347` after event 5 therefore compared against the
root-cycle reference, not the event-5 post-event reference. The resulting
0.88 stop was a bookkeeping-triggered premature validity stop.

## Correction

Only the diagnostic/validity reference is reset:

```text
Vp_cycle0 <- integral(particle, setup)
```

at exactly every completed `q=b` event, before the new facilitated sink-OFF
interval begins. The cutoff remains exactly `Vp/Vp_cycle >= 0.88`. Phase-field
state, coarsening, event mechanics, transport, barrier, clocks, thresholds,
`b=0.25 nm`, and `C4=0.05` are unchanged.

The continuation uses the exact event-5 checkpoint and recorded seed
`198902775941295178708233632700209944609`. Replaying the spawned RNG streams
reproduced the saved root threshold and all five saved descendant thresholds
exactly before continuation.
