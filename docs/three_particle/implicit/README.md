# Adaptive implicit surface diffusion

The new solver advances the full current PF field on the accepted CMC grid.
The short native overlap passes for both the unequal chain and matched-μ null.
The long-time comparison is complete through the existing native bounds guards.
Sustained additional stress loading is observed, but the analytical null
develops material transfer. Full Phase A and Phase B remain gated. See
[results](RESULTS.md) and [long-time plots](long_time_plots.pdf).

For fixed ownership, write the native semidiscrete equations as

    f_dot = A(f) μ(f)
    μ(f) = W_f f(1-f)(1-2f) - k_f L f + g_GB
    H(f) = diag[W_f (1-6f+6f²)] - k_f L.

A is the cylindrical finite-volume divergence of the native face-projected
mobility times the chemical-potential gradient. It includes the native
radial-face transverse periodic stencil and zero external transport faces.
The base step solves

    [I - h A(f) H(f)] d = h A(f) μ(f).

It then applies μ(f)+H(f)d through the ORIGINAL native flux and divergence
kernels. The two adjacent cells receive the same common face flux, conserving
weighted global volume up to floating-point accumulation. There is no global
rescaling, grain-volume constraint, clipping, imposed symmetry or fitted
transfer law. The Hessian is exact for μ; derivatives of the mobility are
lagged in this base step. Full/half step doubling controls that approximation.
The accepted extrapolation 2*f_half_half-f_full is second-order consistent;
it is itself a linear combination of conservative flux increments.

The slow step is thus implicit, with the stiff curvature response solved on
the current field. It is not a separated phenomenological flux model. Both
geometry and chemical potential are measured again after each accepted step.
The error estimator also compares TJ angle and meridional curvature using
the unchanged frozen CMC calibration rule.

The sparse preconditioner uses ILU for h≤0.05 model time, and LU of a
thresholded matrix for larger steps. Large-step factors can be reused while
h stays within a factor of three and the field changes by less than 0.005.
Entries below 1e-7 may be removed from that preconditioner, never from the
solved matrix. GMRES uses rtol=1e-11 for small steps and 1e-9 for large steps,
atol=0, plus an independently checked relative residual below 1e-8.
The initial version used ILU at all timesteps; its late Krylov failures
limited speed. A saved null scout documents that computational limitation.
A small nonzero absolute solver tolerance also caused misleading relative-
residual failures as the bounds guard forced h toward zero; atol=0 fixes
that numerical diagnostic without changing the native bounds guard. See the official
[SciPy GMRES documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html)
for the residual/preconditioning semantics. A direct sparse LU benchmark
cost about 11 seconds per step; ILU/GMRES took 0.46–0.83 seconds over tested
step sizes 100–10000 times native dt. Full/half error estimation needs three
base solves, and the initial transient limits useful acceleration.

## Validation

35 focused tests pass, including sparse/native equivalence on an asymmetric
field with arbitrary potential and nontrivial boundaries, first-order base
consistency, cylindrical mass conservation, mirror symmetry, exact resumed
current-field continuation, CMC geometry, and production transfer/restart
regressions. The production kernels are unchanged.

The overlap uses all eleven native observations, not only the endpoints.
The maximum errors across both cases are:

| Quantity | Maximum absolute error | Gate |
|---|---:|---:|
| Each grain's relative volume | 3.92e-10 | 1e-8 |
| TJ total curvature | 10931 /m | 20000 /m |
| Contact radius | 2.99e-14 m | 2e-12 m |
| Cannon–Carter stress | 11161 Pa | 20000 Pa |
| PF geometric stress | 11391 Pa | 20000 Pa |
| Projected μC−μouter | 645 Pa | 5000 Pa |
| Maximum field difference | 3.71e-6 | 2e-5 |
| Total-volume difference | 6.67e-16 relative | 1e-11 |

The unequal final center-volume loss differs by 0.0155% of the native loss.
The null's final center-volume change remains about +1.893e-10 relative.
Neither number establishes a long-time null. Complete numerical evidence is
in [overlap.json](overlap.json); the later-analysis criteria are fixed in
[PROTOCOL.md](PROTOCOL.md).

Reproduction (repository venv; single numerical thread):

    python scripts/three_particle_implicit_run.py --case unequal --label overlap_v1
    python scripts/three_particle_implicit_run.py --case null --label overlap_v1
    python scripts/three_particle_implicit_qualify.py

Output directories must be new. Runs live under `runs/three_particle_implicit`
and never overwrite the accepted CMC or native explicit evidence. Atomic
checkpoints contain f, fixed ownership, grid, time, and next timestep.
Integrator state consists of the actual field and next h; preconditioner
factors are disposable numerical caches, not physical restart state.
