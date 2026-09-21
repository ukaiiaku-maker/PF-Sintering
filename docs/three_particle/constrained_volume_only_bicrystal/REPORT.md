# Volume-only bicrystal conjugacy qualification

**Decision: `ACTIVE_DOMAIN_CONTROLS_FORCE`**

The constrained manifold fixes each grain volume and holds the displacement
through an ownership/GB frame evaluated directly at the absolute coordinate.
It does not constrain material centroids. Every archived state satisfies the
`1e-7` projected KKT gate without clipping; normalized volume errors are below
`1.5e-14`.

The four `u=0.5b` redistribution guesses converge on their common domain with
an energy spread of `1.109e-20 J`. Source
placement therefore does not explain the force failure.

The central forces do not converge: `Delta u/b = 0.10`, `0.05`, and `0.025`
give `3.695458e-07`, `2.247347e-07`, and `-6.489538e-08 N`, respectively. The independent
Cannon--Carter force is `3.370000e-07 N`.

Active-domain sensitivity is much larger than the virtual-work signal. At
`u=0`, expanding from 21,443 to 27,746 active cells changes the converged
energy by `6.168e-15 J`. At `u=0.5b`,
expanding from 18,931 to 25,242 cells changes it by
`6.919e-15 J`. All expanded states pass
the same KKT gate. A force derived from these energies is therefore not
invariant to the artificial active-domain boundary.

The lowest five projected-Hessian eigenvalues are positive for both old
moment-constrained `u=0` branches and for representative volume-only states.
The old branches are local minima on their recorded constrained domain, so the
earlier word “metastable” is second-order supported there. This does not rescue
work conjugacy because the active-domain energy is not converged.

The old moving-moment multiplier contributes
`1.123134e-07 N`, near Cannon--Carter but
not the old branch finite-difference force
`1.692737e-08 N`. That multiplier is only the
moment-target term: the archived objective and constraint basis also depend
explicitly on the advected ownership frame, and the neighboring stationary
states switch branches. It cannot by itself be identified with the complete
envelope derivative.

No C2 state was minimized or advanced. The completed stochastic trajectory,
root kinetics, barriers, and ordinary PF evolution were not modified.
