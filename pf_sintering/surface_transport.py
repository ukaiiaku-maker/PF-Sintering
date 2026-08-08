"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 12 Commit 2: unified variational tangential surface-diffusion
transport. Replaces the special Ostwald removal/redistribution kernel with
ONE conserved flux driven by the SAME production chemical potential
`mu_f = delta F / delta f` used for reporting -- no spatial removal mask,
no prescribed volume decrement, no receiver fraction, no manual choice of
where material goes. See MILESTONE_12_UNIFIED_VARIATIONAL_SURFACE_COARSENING.md
for the full derivation; summarized here:

    n = grad(f) / sqrt(|grad(f)|^2 + eps_n^2)          (interface normal)
    P_t = I - n (x) n                                   (tangential projector)
    M_tensor = M_s * q(f) * P_t                          (rank-1, PSD, tangential-only)
    J = -M_tensor . grad(mu_f)                            (surface flux)
    df/dt = -div(J)                                      (conservation law)

`q(f) = (12/W) * f^2*(1-f)^2` is normalized so `integral q(f_0(n)) dn = 1`
for the analytic equilibrium tanh profile `f_0(x) = 0.5*(1+tanh(x/W))`
(derived from the SAME `k_f`/`W_f` the production bulk+gradient energy
already uses -- `L = 2*sqrt(k_f/W_f) = W` exactly, confirming `p.interface_
width` already IS this profile's own width parameter; the q-normalization
integral reduces to `integral f_0^2(1-f_0)^2 dx = W/12`, giving the 12/W
prefactor -- see `derive_q_normalization_analytic` and its numerical check
in tests/test_surface_transport.py). This makes the INTEGRATED tangential
mobility through the interface independent of dx or W by construction.

`M_s_ref` preserves the qualified production CH mobility's own integrated
interface conductance: the old (isotropic) `M_old(f) = min(M_f*(16f^2(1-f)^2)^2,
M_f)` evaluated on the same equilibrium profile is exactly `M_f*sech^8(x/W)`
(the min() never binds there, since sech<=1), and `integral sech^8(u)du =
32/35` (standard reduction-formula result), giving `M_s_ref = M_f*W*(32/35)`
-- dimensionally consistent (q has units 1/length, so M_s must carry the
same length dimension M_f itself lacks to match M_old's units). `surface_
mobility_scale` (already multiplying `M_f`) therefore propagates into
`M_s_ref` automatically -- no new, separate rate parameter is introduced.
"""

from __future__ import annotations

import math

import numpy as np

from .bc_ops import face_average, face_gradient_x, face_gradient_y, flux_divergence, grad_bc

SECH8_INTEGRAL = 32.0 / 35.0  # integral_{-inf}^{inf} sech^8(u) du


def interface_localization_q(f, W):
    """q(f) = (12/W) f^2(1-f)^2, normalized so integral q(f_0(n)) dn = 1
    for the analytic equilibrium profile of width W (see module docstring)."""
    return (12.0 / W) * f * f * (1.0 - f) ** 2


def q_normalization_numeric(W, dx, n_cells=4001):
    """Numerically integrate q(f_0(x)) over a fine 1-D discretization of
    the SAME analytic equilibrium profile (not the actual 2-D discrete
    field), to check the analytic 12/W prefactor independent of dx/grid --
    should return 1.0 to high precision regardless of dx (Section 5's
    'test the normalization at dx=5nm and 2.5nm with physical W=20nm')."""
    half = 8.0 * W
    x = np.linspace(-half, half, n_cells)
    f0 = 0.5 * (1.0 + np.tanh(x / W))
    q = interface_localization_q(f0, W)
    return float(np.trapezoid(q, x))


def m_s_ref(M_f, W):
    """Integrated tangential mobility preserving the qualified production
    CH mobility's own integrated interface conductance (see module
    docstring derivation). M_f already carries surface_mobility_scale."""
    return M_f * W * SECH8_INTEGRAL


def interface_normal(f, dx, bc_x, bc_y, eps_n):
    """n = grad(f)/sqrt(|grad f|^2 + eps_n^2). eps_n is a purely numerical
    regularization (Section 7): since q(f)->0 in the bulk (where grad(f)
    is undefined/noisy), eps_n must have no physical effect there -- only
    prevents division by ~0 where it is multiplied away by q(f) anyway.
    Not a physics parameter; sensitivity to its exact value within a
    reasonable numeric range is tested in tests/test_surface_transport.py."""
    gx, gy = grad_bc(f, dx, bc_x=bc_x, bc_y=bc_y)
    mag = np.sqrt(gx * gx + gy * gy + eps_n * eps_n)
    return gx / mag, gy / mag, mag


def tangential_projector(nx, ny):
    """P_t = I - n(x)n as its symmetric components (Pxx, Pxy, Pyy);
    Pyx == Pxy. Symmetric and positive-semidefinite by construction
    (I - n n^T for a unit vector n has eigenvalues {0, 1})."""
    Pxx = 1.0 - nx * nx
    Pxy = -nx * ny
    Pyy = 1.0 - ny * ny
    return Pxx, Pxy, Pyy


def surface_mobility_tensor(f, dx, W, M_s, bc_x, bc_y, eps_n):
    """M_tensor = M_s * q(f) * P_t, as its symmetric components. Cell-
    centered. Only tangential transport (Section 4: M_normal=M_bulk=
    M_vapor=conserved-GB-diffusion=0 in this first qualification -- there
    is no separate normal/bulk term anywhere in this construction to zero
    out; the tensor is rank-1 tangential-only by construction)."""
    q = interface_localization_q(f, W)
    nx, ny, _ = interface_normal(f, dx, bc_x, bc_y, eps_n)
    Pxx, Pxy, Pyy = tangential_projector(nx, ny)
    coef = M_s * q
    return coef * Pxx, coef * Pxy, coef * Pyy


def surface_flux(mu, Mxx, Mxy, Myy, dx, bc_x, bc_y):
    """J = -M_tensor . grad(mu_f), cell-centered (Myx == Mxy, symmetric tensor)."""
    gx_mu, gy_mu = grad_bc(mu, dx, bc_x=bc_x, bc_y=bc_y)
    Jx = -(Mxx * gx_mu + Mxy * gy_mu)
    Jy = -(Mxy * gx_mu + Myy * gy_mu)
    return Jx, Jy


def dissipation_density(Mxx, Mxy, Myy, mu, dx, bc_x, bc_y):
    """D_CH density = grad(mu) . M_tensor . grad(mu), pointwise -- manifestly
    non-negative wherever M_tensor is PSD (a quadratic form x^T M x with M
    PSD), which surface_mobility_tensor's construction guarantees at every
    point by construction (Section 9's D_CH>=0 requirement)."""
    gx_mu, gy_mu = grad_bc(mu, dx, bc_x=bc_x, bc_y=bc_y)
    return Mxx * gx_mu * gx_mu + 2.0 * Mxy * gx_mu * gy_mu + Myy * gy_mu * gy_mu


def _face_projected_mobility_and_flux(f, mu, dx, W, M_s, bc_x, bc_y, eps_n, axis):
    """Milestone 13D Sections 2-5: construct `n`, `P_t`, `grad(mu)` ALL AT
    the SAME face (the '+axis' face of each cell -- axis=1 for x-faces,
    axis=0 for y-faces) via `bc_ops.face_gradient_x/y`, then project
    THERE -- not interpolated from cell-centered values (Section 2's
    design principle). `q` is evaluated at the face-averaged `f` (Section
    5: "evaluate/interpolate q at the face consistently", not re-derived
    or dx/curvature-dependent). Returns the FULL flux vector at that face
    family (both components -- the "own" component is what the
    conservative update uses; the other is diagnostic-only, needed to
    check `J_face . n_face`) plus the face mobility tensor and `grad(mu)`
    components (needed for the face-local energy-dissipation audit,
    Section 8)."""
    face_gradient = face_gradient_x if axis == 1 else face_gradient_y
    gx_f, gy_f = face_gradient(f, dx, bc_x=bc_x, bc_y=bc_y)
    gmag = np.sqrt(gx_f * gx_f + gy_f * gy_f + eps_n * eps_n)
    nx, ny = gx_f / gmag, gy_f / gmag

    f_face = face_average(f, axis=axis, bc=(bc_x if axis == 1 else bc_y))
    q_face = interface_localization_q(f_face, W)

    gx_mu, gy_mu = face_gradient(mu, dx, bc_x=bc_x, bc_y=bc_y)

    Pxx = 1.0 - nx * nx
    Pxy = -nx * ny
    Pyy = 1.0 - ny * ny
    coef = M_s * q_face
    Mxx, Mxy, Myy = coef * Pxx, coef * Pxy, coef * Pyy

    Jx = -(Mxx * gx_mu + Mxy * gy_mu)
    Jy = -(Mxy * gx_mu + Myy * gy_mu)
    return dict(Jx=Jx, Jy=Jy, Mxx=Mxx, Mxy=Mxy, Myy=Myy, nx=nx, ny=ny, gx_mu=gx_mu, gy_mu=gy_mu, q_face=q_face)


def surface_flux_face_projected(f, mu, dx, W, M_s, bc_x, bc_y, eps_n=None):
    """Milestone 13D Section 6: the face flux the conservative update uses
    IS the locally-projected flux -- not a cell-centered flux interpolated
    onto faces after the fact. `Jx_face` (the x-COMPONENT of the flux
    vector constructed AT x-faces) and `Jy_face` (the y-component
    constructed AT y-faces) are the two the divergence needs; `xface`/
    `yface` carry the full per-face-family vectors and mobility tensors
    for the tangentiality and energy-dissipation diagnostics."""
    if eps_n is None:
        eps_n = 1e-6 / W
    xface = _face_projected_mobility_and_flux(f, mu, dx, W, M_s, bc_x, bc_y, eps_n, axis=1)
    yface = _face_projected_mobility_and_flux(f, mu, dx, W, M_s, bc_x, bc_y, eps_n, axis=0)
    return dict(Jx_face=xface["Jx"], Jy_face=yface["Jy"], xface=xface, yface=yface)


def face_projected_tangentiality(fp):
    """Milestone 13D Section 9: J_face . n_face at each face family,
    computed directly from surface_flux_face_projected's own vectors (no
    interpolation needed at all, unlike discrete_tangentiality.
    face_tangentiality's legacy-mode diagnostic) -- both the flux and the
    normal it is dotted against were constructed at the identical face."""
    xf, yf = fp["xface"], fp["yface"]
    dot_x = xf["Jx"] * xf["nx"] + xf["Jy"] * xf["ny"]
    mag_x = np.hypot(xf["Jx"], xf["Jy"])
    dot_y = yf["Jx"] * yf["nx"] + yf["Jy"] * yf["ny"]
    mag_y = np.hypot(yf["Jx"], yf["Jy"])
    ratio_x = np.divide(np.abs(dot_x), mag_x, out=np.full_like(mag_x, np.nan), where=mag_x > 0)
    ratio_y = np.divide(np.abs(dot_y), mag_y, out=np.full_like(mag_y, np.nan), where=mag_y > 0)
    return dict(dot_x=dot_x, mag_x=mag_x, ratio_x=ratio_x, dot_y=dot_y, mag_y=mag_y, ratio_y=ratio_y)


def dissipation_density_face_projected(fp):
    """DEPRECATED / ALGEBRAICALLY INCORRECT -- kept only for Milestone 13D
    regression comparison, NOT used by surface_divergence_update (which
    uses `exact_dissipation_face_projected` below). This was Milestone
    13D's original guess at a face-local D_CH: the FULL 2x2 quadratic
    form `v^T M v` at each face family independently, using that face's
    own complete (Mxx, Mxy, Myy) and (gx_mu, gy_mu). Milestone 13E's exact
    derivation (see `exact_dissipation_face_projected`) shows this is
    NOT the quantity conjugate to `Fdot_chain = dx^2*sum(mu*f_dot)`: the
    conservative divergence only ever uses the x-face's OWN x-component
    of J and the y-face's OWN y-component (never the "other," diagnostic-
    only component each face's full vector also carries), so summation by
    parts produces a strictly SMALLER quantity than this full-quadratic-
    form guess (confirmed numerically: (F_n-F_{n+1})/dt converges to only
    ~65% of this D_CH as dt->0, not 100%)."""
    xf, yf = fp["xface"], fp["yface"]
    Dx = xf["Mxx"] * xf["gx_mu"] ** 2 + 2.0 * xf["Mxy"] * xf["gx_mu"] * xf["gy_mu"] + xf["Myy"] * xf["gy_mu"] ** 2
    Dy = yf["Mxx"] * yf["gx_mu"] ** 2 + 2.0 * yf["Mxy"] * yf["gx_mu"] * yf["gy_mu"] + yf["Myy"] * yf["gy_mu"] ** 2
    return Dx, Dy


def fdot_chain_face_projected(f, mu, dx, W, M_s, bc_x, bc_y, eps_n=None, fp=None):
    """Milestone 13E Section 3: the authoritative instantaneous
    thermodynamic rate `Fdot_chain = dx^2 * sum(mu * f_dot)`, computed
    directly from the discrete chain rule using the EXACT implemented
    face-projected flux and divergence (mu = delta F_h/delta f already
    established by ch_exact_energy.py; f_dot = L_h(mu) is exactly what
    surface_divergence_update's face_projected mode computes) -- not
    estimated from finite-time F differences."""
    if fp is None:
        fp = surface_flux_face_projected(f, mu, dx, W, M_s, bc_x, bc_y, eps_n)
    div = flux_divergence(fp["Jx_face"], fp["Jy_face"], dx, bc_x=bc_x, bc_y=bc_y)
    f_dot = -div
    return float(dx * dx * np.sum(mu * f_dot)), f_dot


def exact_dissipation_face_projected(fp):
    """Milestone 13E Sections 4, 8: the EXACT discrete dissipation `D_h`
    conjugate to `fdot_chain_face_projected`, derived via discrete
    summation by parts on the ACTUAL implemented `face_gradient_x/y` +
    `flux_divergence` pair (not assumed): `Fdot_chain = -D_h` is an exact
    algebraic identity (verified to machine precision, rel diff ~1e-16,
    not just as dt->0) because `_axis_efflux` and the forward face
    difference `(a[i+1]-a[i])/dx` are exact discrete adjoints of each
    other (both periodic and reflecting/no_flux BC, proven by direct
    telescoping-sum expansion -- MILESTONE_13E report Section 3).

    Only the "own-direction" component of each face family's flux vector
    enters (the conservative divergence never uses the other, diagnostic-
    only component), so:

        D_h = dx^2 * [ sum_x-faces(Mxx*gx_mu^2 + Mxy*gx_mu*gy_mu)
                      + sum_y-faces(Myy*gy_mu^2 + Mxy*gx_mu*gy_mu) ]

    -- structurally different from (and smaller than) the deprecated
    `dissipation_density_face_projected`'s full-quadratic-form guess at
    each face (that guess double-counts the cross term and includes an
    extra, not-actually-present Myy*gx_mu^2 / Mxx*gy_mu^2 term at the
    "wrong" face family). `D_h`'s sign-definiteness is NOT obvious
    pointwise (each per-face partial term can itself be negative for
    some mu, unlike a full PSD quadratic form) -- confirmed instead by
    direct numerical probing of the GLOBAL quadratic form `<mu,L_h mu>`
    over hundreds of diverse (smooth, high-frequency, single-spike) test
    vectors at multiple states (0/450+ violations found) -- see
    MILESTONE_13E report Section 6."""
    xf, yf = fp["xface"], fp["yface"]
    Dx = xf["Mxx"] * xf["gx_mu"] ** 2 + xf["Mxy"] * xf["gx_mu"] * xf["gy_mu"]
    Dy = yf["Myy"] * yf["gy_mu"] ** 2 + yf["Mxy"] * yf["gx_mu"] * yf["gy_mu"]
    return Dx, Dy


def surface_flux_divergence_conservative(Jx_cell, Jy_cell, dx, bc_x, bc_y):
    """Interpolates the cell-centered flux to faces (simple average) and
    applies bc_ops.flux_divergence, whose exact telescoping-sum mass
    conservation holds regardless of the face value's own accuracy (see
    bc_ops.flux_divergence docstring) -- this is what makes the update
    conservative even though the flux itself was built from a tensor
    mobility that does not fit the legacy face-centered-scalar pattern."""
    Jx_face = face_average(Jx_cell, axis=1, bc=bc_x)
    Jy_face = face_average(Jy_cell, axis=0, bc=bc_y)
    return flux_divergence(Jx_face, Jy_face, dx, bc_x=bc_x, bc_y=bc_y)


def surface_divergence_update(f, mu, dx, dt, W, M_s, bc_x, bc_y, eps_n=None,
                               face_flux_mode="cell_average_legacy"):
    """One full conservative step of df/dt = -div(J), J = -M_s*q(f)*P_t.grad(mu_f).

    Milestone 13D Section 7: `face_flux_mode` selects between the ORIGINAL
    ("cell_average_legacy", still the default -- UNCHANGED code path, not
    touched by Milestone 13D) and the new "face_projected" construction
    (Section 2: n/P_t/grad(mu) built directly AT each face, not
    interpolated from cell centers -- see `surface_flux_face_projected`).
    The two modes' diagnostics dicts differ in shape (legacy returns
    cell-centered Jx/Jy/Mxx/Mxy/Myy/D_density; face_projected returns
    face-centered Jx_face/Jy_face plus the full per-face-family `xface`/
    `yface` vectors and `D_density_x`/`D_density_y`) -- callers must check
    `face_flux_mode` before assuming a particular key's shape.

    Milestone 13E: `D_density_x`/`D_density_y` for `face_projected` now
    come from `exact_dissipation_face_projected` (the algebraically exact
    discrete dissipation, `Fdot_chain = -dx^2*(sum(D_density_x)+
    sum(D_density_y))` to machine precision) -- NOT Milestone 13D's
    original `dissipation_density_face_projected` guess, whose sum only
    converged to ~65% of the true (F_n-F_{n+1})/dt as dt->0."""
    if eps_n is None:
        eps_n = 1e-6 / W
    if face_flux_mode == "cell_average_legacy":
        Mxx, Mxy, Myy = surface_mobility_tensor(f, dx, W, M_s, bc_x, bc_y, eps_n)
        Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, dx, bc_x, bc_y)
        div = surface_flux_divergence_conservative(Jx, Jy, dx, bc_x, bc_y)
        f_new = f - dt * div
        D_density = dissipation_density(Mxx, Mxy, Myy, mu, dx, bc_x, bc_y)
        return f_new, dict(Jx=Jx, Jy=Jy, Mxx=Mxx, Mxy=Mxy, Myy=Myy, div=div, D_density=D_density)
    elif face_flux_mode == "face_projected":
        fp = surface_flux_face_projected(f, mu, dx, W, M_s, bc_x, bc_y, eps_n)
        div = flux_divergence(fp["Jx_face"], fp["Jy_face"], dx, bc_x=bc_x, bc_y=bc_y)
        f_new = f - dt * div
        D_density_x, D_density_y = exact_dissipation_face_projected(fp)
        return f_new, dict(Jx_face=fp["Jx_face"], Jy_face=fp["Jy_face"],
                            xface=fp["xface"], yface=fp["yface"], div=div,
                            D_density_x=D_density_x, D_density_y=D_density_y)
    else:
        raise ValueError(f"unknown face_flux_mode {face_flux_mode!r}")


def variational_surface_diffusion_step(f, mu, dx, dt, W, M_s, bc_x, bc_y, eps_n=None,
                                        face_flux_mode="face_projected"):
    """Milestone 13E Section 12: canonical transport-law entry point for the
    `variational_surface_diffusion` mode (Milestone 12's opt-in unified
    conserved-coarsening mode, distinct from `legacy_ostwald`; never wired
    into `model.py`'s production stepper -- it lives in this milestone-track
    module and the campaign scripts that call it directly).

    `face_projected` is the DEFAULT here -- promoted after this milestone's
    exact discrete chain-rule/dissipation closure (`Fdot_chain = -D_h` to
    machine precision, `(F_n-F_{n+1})/dt -> D_h` as dt->0), the corrected
    authoritative tangentiality diagnostic (roundoff-level on both the
    isolated-substrate and particle-contact benchmarks), and grid-consistent
    `dM_neck/dt` sign at dx=5, 2.5, 1.25nm (MILESTONE_13E report Section 9).
    `cell_average_legacy` stays available via explicit `face_flux_mode=`.

    `surface_divergence_update`'s OWN default is deliberately left
    UNCHANGED (still `cell_average_legacy`) -- it is shared low-level
    infrastructure called by ~15 pre-13D/13E milestone scripts and tests
    that assume that default and its cell-centered `Jx`/`Jy` diagnostic
    shape; flipping it would silently alter those unrelated regression
    baselines. This wrapper is the only place `face_projected` is now the
    default."""
    return surface_divergence_update(f, mu, dx, dt, W, M_s, bc_x, bc_y, eps_n=eps_n,
                                      face_flux_mode=face_flux_mode)
