# Event-4 transport-affinity pause qualification

The genuine stochastic trajectory stopped during event 4 after accepting
`q/b = 0.8775`.  The last accepted state remained bounded, conservative, and
topologically valid.  The following minimum `0.0025b` transfer trial stopped
because its current-state source-to-sink transport affinity became
nonpositive.

This is an interrupted transport transit, rather than avalanche extinction or
a numerical field failure.  The already-nucleated source remains active, and
root and descendant clocks remain frozen while the ordinary sink-off PF field
evolves.  After one existing bicrystal passive quadrature block (`0.025` model
time, `0.389498579 ms`), the unchanged LEFT affinity was `0.336221785 MPa`.
The unchanged qualified event integrator then accepted the continuation from
`0.8775b` to `0.88b`.

The overlap result in `event4_transport_pause_overlap.json` has zero measured
relative material-volume error, ownership closure of
`2.220446049250313e-16`, unchanged field extrema, and no topology stop.  The
driver therefore checkpoints an `EVENT_TRANSPORT_PAUSED` state, advances one
predeclared passive block with the source alive, records
`EVENT_TRANSPORT_RECOVERY`, and retries the same event restart.  There is no
affinity clipping, fitted recovery threshold, new stochastic draw, or change
to microscopic physics.
