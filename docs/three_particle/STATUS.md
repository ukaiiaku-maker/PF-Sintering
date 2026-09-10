# Three-particle ripening development

Phase A is NOT QUALIFIED. Phase B is gated off.

Base: `stochastic-pr-current-state-production-v1`, commit
`0023db8509df151622ae94fec140e714b967ef31`. See `THREE_PARTICLE_BASE_AUDIT.json`.
The original dirty production worktree has not been edited. The complete
tracked diff is saved externally; its unrelated historical changes were not
applied. An AST audit of 80 transitive source files used by the live worker and
monitor found no missing untracked dependency to copy. No results were copied.

Production uses one conserved solid fraction f, normalized binary ownership,
and axisymmetric cell volumes 2*pi*r*dr*dz. Three ownership fractions are needed
for three separately measured grains; the binary dynamic ownership update
cannot be used unchanged. The new fixed-GB Phase-A representation uses three
compact obstacle fractions summing to one, with eta_i=f*phi_i.

The production `axisym_numba_kernel.flux_kernel` accepts only f and mu,
not grain masks. It projects chemical-potential gradients tangentially on all
internal faces; only external boundary fluxes vanish. Thus it permits transport
across BOTH TJs. There is no per-grain volume correction in this kernel.
Its radial-face transverse derivatives wrap axially even under noflux; preserve
large vacuum margins and check boundary clearance. Do not use the historical
Cartesian `model.py` threeparticle initializer for this axisymmetric chain.

The finite-neck seed has nominal cap radii 100/70/100 nm, W=10 nm,
dx=1.25 nm, neck radius 28 nm, nominal angle 160 degrees. Each neck-to-cap join
matches radius and two derivatives using a quintic. It is a geometric seed,
not a relaxed surface; no seed transient may be called ripening. Nominal cap
radii differ from volume-equivalent radii, which must be measured separately.

Next: initialization-only constrained energy relaxation at fixed grain volumes
and centers, followed by actual PF surface diffusion and all three deterministic
controls. Equal nominal radii do not themselves prove equal chemical potential:
the center has two contacts and outer grains only one; test this rather than
subtracting a null drift or forcing its sign.
