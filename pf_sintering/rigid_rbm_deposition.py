"""Rigid-body event-operator experiments and regression paths.

.. deprecated:: recovery
   :func:`rigid_translation_sink_step` is retained only to reproduce the
   historical rigid-plus-additive-overlap regression.  It is not a physical
   production event operator: grain order parameters are ownership fields,
   not independent body occupancies, so adding and clipping them is invalid.

`axisym_sink_rbm.active_sink_transport_step`
builds its advection velocity as

    vz_field = -v_event * particle / (particle + substrate)

and then advects BOTH grain fields with this SAME spatially-varying
field. Far inside the particle bulk (particle/(particle+substrate)~1)
this is ~-v_event; deep in the substrate it is ~0 -- but through the
mixed GB/neck region it interpolates smoothly between the two, so the
"rigid" body is not actually translated rigidly: it is sheared/
compressed as it passes through the neck, and the substrate is ALSO
advected wherever the field is locally nonzero (i.e. near the interface,
not just far away). This is a genuine kinematic deformation field, not
the rigid relative-particle-translation the RBM model is supposed to
represent.

This module implements the intended mechanism directly:

    1. particle translated RIGIDLY by delta_sink (uniform shift, same
       displacement at every point -- via scipy.ndimage.shift, the SAME
       interpolation convention (order=3, mode="nearest") already
       vetted in this codebase for a rigid displacement
       (stress_estimators_axisym.py's sigma_reaction_instantaneous trial
       displacement -- a DIFFERENT, diagnostic-only use of the same
       mechanics, not reused here for kinetics, only the interpolation
       convention is shared).
    2. substrate held EXACTLY stationary (no motion at all).
    3. the resulting overlap (f>1, "excess") / vacancy (f<0, "deficit")
       is corrected via the SAME redistribution_fn interface
       active_sink_transport_step already uses (curvature_compatible_
       deposition.py's compliance-based Thomas-solved operator in
       production) -- this module changes ONLY the advection kinematics,
       not the redistribution mechanism, the Coble completion kinetics,
       the one-b quota contract, or any stress definition.

Same (f, e1, e2, completed, diag) return contract as
active_sink_transport_step for drop-in comparison.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import ndimage

from pf_sintering.axisym import axisym_mu_f_gb
from pf_sintering.axisym_sink_rbm import AxisymSink, HazardParams, SECONDS_PER_MODEL_TIME, _axisym_weighted_sum


def rigid_translation_sink_step(f, particle, substrate, sink: AxisymSink, hp: HazardParams, sigma_s: float,
                                 dt: float, dz: float, r_c, z, GB_z_hint,
                                 seconds_per_model_time: float = SECONDS_PER_MODEL_TIME,
                                 redistribution_fn=None, shift_order=3):
    if not sink.active:
        return f, particle, substrate, False, dict(paused=False, active=False)

    sigma_drive = max(0.0, sigma_s)
    dt_seconds = dt * seconds_per_model_time
    remaining = max(0.0, hp.b - sink.current_disp)

    if sigma_drive <= 0.0 or remaining <= 0.0:
        return f, particle, substrate, False, dict(
            paused=True, active=True, sigma_drive=sigma_drive, tau_Coble=math.inf, v_event=0.0,
            requested_d_delta=0.0, delta_sink_this_step=0.0, delta_event=sink.current_disp,
            remaining_to_b=remaining, shift_m=0.0, V_excess=0.0, V_deficit=0.0, V_net_correction=0.0,
            mass_conservation_residual=0.0, e1e2f_residual=float(np.max(np.abs(particle + substrate - f))),
            completed=False)

    xd = 0.5 * hp.GS / 2
    tau_Coble = (xd * xd * hp.kB * hp.T) / (sigma_drive * hp.Omega * hp.D_gb) + hp.tau_ex0
    v_event = hp.b / tau_Coble
    d_delta_full = v_event * dt_seconds
    d_delta_requested = min(d_delta_full, remaining)

    # rigid shift: same magnitude/sign convention as the mixture-weighted
    # operator's far-field particle velocity (-v_event, i.e. the particle
    # moves toward -z / into the substrate) so the two mechanisms are
    # comparable dose-for-dose.
    shift_cells = -d_delta_requested / dz
    particle_shifted = ndimage.shift(particle, shift=(shift_cells, 0.0), order=shift_order, mode="nearest")
    np.clip(particle_shifted, 0.0, 1.0, out=particle_shifted)
    substrate_new = substrate.copy()  # TRUE rigid mechanism: stationary, not advected at all

    f_raw = particle_shifted + substrate_new
    excess = np.maximum(0.0, f_raw - 1.0)
    deficit = np.maximum(0.0, -f_raw)
    f_new = np.clip(f_raw, 0.0, 1.0)
    V_excess = _axisym_weighted_sum(excess, r_c)
    V_deficit = _axisym_weighted_sum(deficit, r_c)
    V_net_correction = V_excess - V_deficit
    mass_resid = 0.0
    if redistribution_fn is not None and abs(V_net_correction) > 1e-30:
        # dt_seconds: this call's own physical duration (no internal
        # substep loop here, unlike active_sink_transport_step/
        # multi_sink_transport_step -- the rigid shift is applied once
        # per call, so the full per-call dt_seconds is the right elapsed
        # time for the deposit's diffusion-length width).
        f_new, particle_shifted, substrate_new, redist_diag = redistribution_fn(
            f_new, particle_shifted, substrate_new, r_c, z, GB_z_hint, V_net_correction, dt_seconds=dt_seconds)
        mass_resid = abs(redist_diag.get("closure_error", 0.0))

    sink.current_disp += d_delta_requested
    sink.cumulative_disp += d_delta_requested
    completed = False
    if sink.current_disp >= hp.b - 1e-15:
        sink.active = False
        sink.current_disp = 0.0
        sink.hazard = 0.0
        completed = True

    diag = dict(paused=False, active=sink.active, sigma_drive=sigma_drive, tau_Coble=tau_Coble, v_event=v_event,
                requested_d_delta=d_delta_requested, delta_sink_this_step=d_delta_requested,
                delta_event=(0.0 if completed else sink.current_disp), remaining_to_b=max(0.0, hp.b - sink.current_disp),
                shift_m=d_delta_requested, V_excess=V_excess, V_deficit=V_deficit, V_net_correction=V_net_correction,
                mass_conservation_residual=mass_resid,
                e1e2f_residual=float(np.max(np.abs(particle_shifted + substrate_new - f_new))), completed=completed)
    return f_new, particle_shifted, substrate_new, completed, diag


def conservative_shift_minus_z(field, displacement_m, dz):
    """Monotone finite-volume remap for a uniform shift toward ``-z``.

    The current bounded gate uses subcell doses (``0 <= alpha <= 1``).  A
    no-flux lower boundary retains material already in cell zero; the upper
    boundary supplies no inflow.  The resulting telescoping update conserves
    the unweighted z-column sum exactly up to floating-point roundoff.
    """
    alpha = float(displacement_m / dz)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"subcell conservative remap requires 0<=delta/dz<=1, got {alpha}")
    src = np.asarray(field)
    out = np.empty_like(src)
    out[0] = src[0] + alpha * src[1]
    out[1:-1] = (1.0 - alpha) * src[1:-1] + alpha * src[2:]
    out[-1] = (1.0 - alpha) * src[-1]
    return out


def ownership_to_kinematic_bodies(f, z, GB_z):
    """Make disjoint temporary body occupancies from total solid ``f``.

    These are deliberately *not* the diffuse grain order parameters.  The GB
    split is crisp and spatially anchored for the first sharp-union benchmark,
    giving ``body_particle + body_neighbor == f`` exactly.
    """
    particle_side = np.asarray(z)[:, None] >= float(GB_z)
    body_particle = np.where(particle_side, f, 0.0)
    body_neighbor = f - body_particle
    return body_particle, body_neighbor


def representation_corrected_union_transfer(
        f_pre, particle_owner_pre, substrate_owner_pre, displacement_m,
        dz, r_c, z, GB_z, redistribution_fn=None, dt_seconds=None):
    """Representation-corrected Candidate A using temporary body occupancies.

    The active body's occupancy is conservatively translated, the stationary
    body's occupancy is unchanged, and their sharp physical union is ``max``.
    Overlap is thereby counted once without eroding dense solid at the GB.
    Diffuse ownership is reconstructed only afterward from the spatially
    anchored pre-event ownership fraction.
    """
    body_p, body_n = ownership_to_kinematic_bodies(f_pre, z, GB_z)
    body_p_shifted = conservative_shift_minus_z(body_p, displacement_m, dz)
    # Sharp union for non-negative occupancies is max(a,b)=a+b-min(a,b).
    # Restrict the overlap term to cells where both bodies are positive so
    # tiny inherited PF undershoots are preserved exactly at zero dose rather
    # than silently clipped to zero by np.maximum.
    overlap = np.where(
        (body_p_shifted > 0.0) & (body_n > 0.0),
        np.minimum(body_p_shifted, body_n), 0.0)
    f_union = body_p_shifted + body_n - overlap

    V_pre_weighted = _axisym_weighted_sum(f_pre, r_c)
    V_union_weighted = _axisym_weighted_sum(f_union, r_c)
    V_source_weighted = V_pre_weighted - V_union_weighted
    if V_source_weighted < -1e-12 * max(abs(V_pre_weighted), 1e-300):
        raise RuntimeError(f"body union gained material: source={V_source_weighted:.6e}")
    V_source_weighted = max(0.0, V_source_weighted)

    owner_sum = particle_owner_pre + substrate_owner_pre
    q_particle = np.divide(
        particle_owner_pre, owner_sum, out=np.zeros_like(f_pre),
        where=owner_sum > 1e-30)
    particle_union = f_union * q_particle
    substrate_union = f_union - particle_union

    f_final, particle_final, substrate_final = f_union, particle_union, substrate_union
    redist_diag = dict(skipped=True)
    if redistribution_fn is not None and V_source_weighted > 1e-30:
        f_final, particle_final, substrate_final, redist_diag = redistribution_fn(
            f_union, particle_union, substrate_union, r_c, z, GB_z,
            V_source_weighted, dt_seconds=dt_seconds)

    return f_final, particle_final, substrate_final, dict(
        body_particle=body_p, body_neighbor=body_n,
        body_particle_shifted=body_p_shifted, f_union=f_union,
        particle_union=particle_union, substrate_union=substrate_union,
        V_pre_weighted=V_pre_weighted, V_union_weighted=V_union_weighted,
        V_source_weighted=V_source_weighted,
        body_volume_relative_error=(
            _axisym_weighted_sum(body_p_shifted, r_c) - _axisym_weighted_sum(body_p, r_c)
        ) / max(abs(_axisym_weighted_sum(body_p, r_c)), 1e-300),
        zero_or_dose_identity_residual=float(np.max(np.abs(f_union - f_pre))),
        union_bounds=(float(np.min(f_union)), float(np.max(f_union))),
        union_partition_residual=float(np.max(np.abs(particle_union + substrate_union - f_union))),
        final_partition_residual=float(np.max(np.abs(particle_final + substrate_final - f_final))),
        redistribution=redist_diag)


def normalized_axisym_support(raw_support, r_c):
    """Normalize a non-negative field under the axisymmetric volume weight.

    The common physical factor ``2*pi*dr*dz`` cancels, so the returned field
    satisfies ``sum(r_c * support) == 1``.  Multiplying it by a weighted
    volume therefore produces a conservative field increment.
    """
    raw = np.maximum(np.asarray(raw_support, dtype=float), 0.0)
    norm = _axisym_weighted_sum(raw, r_c)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("axisymmetric support has zero or non-finite weight")
    return raw / norm


def make_fixed_tj_source_support(
        f, particle, substrate, r_c, z, GB_z, r_neck, W,
        width_factor: float = 1.0):
    """Construct a fixed normalized free-surface arrival support at the TJ.

    Its physical width is ``width_factor*W``.  No diffusivity or elapsed time
    appears here: this field represents where GB-delivered material arrives,
    while the ordinary ``M_s`` phase-field operator subsequently determines
    how it spreads along the free surface.

    Returns the normalized total-solid support and the fixed fraction assigned
    to the moving grain.  Both are frozen from the pre-event state so changing
    the internal rigid-motion packetization cannot change the arrival field.
    """
    if W <= 0.0 or width_factor <= 0.0:
        raise ValueError("W and width_factor must be positive")
    f0 = np.asarray(f, dtype=float)
    p0 = np.asarray(particle, dtype=float)
    n0 = np.asarray(substrate, dtype=float)
    if f0.shape != p0.shape or f0.shape != n0.shape:
        raise ValueError("f and ownership fields must have identical shapes")
    sigma = width_factor * W
    Z = np.asarray(z, dtype=float)[:, None]
    RC = np.asarray(r_c, dtype=float)[None, :]
    interface = 16.0 * np.clip(f0, 0.0, 1.0) ** 2 * (1.0 - np.clip(f0, 0.0, 1.0)) ** 2
    distance2 = (Z - float(GB_z)) ** 2 + (RC - float(r_neck)) ** 2
    support = normalized_axisym_support(
        interface * np.exp(-0.5 * distance2 / (sigma * sigma)), r_c)
    owner_sum = p0 + n0
    particle_fraction = np.divide(
        p0, owner_sum, out=np.full_like(f0, 0.5), where=owner_sum > 1e-30)
    return support, particle_fraction


def apply_fixed_tj_source(
        f, particle, substrate, r_c, source_support,
        particle_fraction, source_volume_weighted):
    """Add a GB-delivered volume to a fixed TJ support without smoothing."""
    support = np.asarray(source_support, dtype=float)
    p_fraction = np.asarray(particle_fraction, dtype=float)
    if support.shape != np.shape(f) or p_fraction.shape != np.shape(f):
        raise ValueError("TJ support and ownership fraction must match f")
    support_norm = _axisym_weighted_sum(support, r_c)
    if not np.isclose(support_norm, 1.0, rtol=2e-13, atol=2e-15):
        raise ValueError(f"TJ source support is not normalized: {support_norm:.17g}")
    if source_volume_weighted < 0.0:
        raise ValueError("source_volume_weighted must be non-negative")
    added = float(source_volume_weighted) * support
    f_new = np.asarray(f) + added
    bound_tol = 2e-12
    if float(np.min(f_new)) < -bound_tol or float(np.max(f_new)) > 1.0 + bound_tol:
        raise RuntimeError(
            "fixed TJ source exceeded the admissible total-solid bounds: "
            f"range=({np.min(f_new):.6e},{np.max(f_new):.6e})")
    p_new = np.asarray(particle) + added * p_fraction
    n_new = np.asarray(substrate) + added * (1.0 - p_fraction)
    added_weighted = _axisym_weighted_sum(added, r_c)
    return f_new, p_new, n_new, dict(
        requested_volume_weighted=float(source_volume_weighted),
        added_volume_weighted=added_weighted,
        closure_error_weighted=added_weighted - float(source_volume_weighted),
        support_weighted_norm=support_norm,
        max_field_increment=float(np.max(added)),
        bounds=(float(np.min(f_new)), float(np.max(f_new))),
        partition_residual=float(np.max(np.abs(p_new + n_new - f_new))))


def local_chemical_potential_transport_drive(
        f, particle, substrate, pf_params, Wc, dr, dz, r_c, r_f, z,
        GB_z, r_neck, W, tj_source_support=None):
    """Return the instantaneous local GB-to-TJ chemical-potential affinity.

    ``axisym_mu_f_gb`` is ``delta F / delta f``.  Because ``f`` is a
    dimensionless occupancy, this code's chemical potential is an energy
    density (J/m^3 = Pa).  Thus ``mu_source - mu_sink`` is already the
    stress-equivalent transport drive.  Equivalently, an atomic chemical
    potential difference is ``Omega*Delta(mu_vol)`` and division by ``Omega``
    recovers this same Pa-valued affinity.

    The GB source average is localized to the current diffuse GB/contact.
    The sink average uses the fixed TJ arrival support when supplied.  No
    event-nucleation stress, previous-run value, or fitted constant enters.
    """
    mu = axisym_mu_f_gb(
        f, particle, substrate, pf_params, Wc, dr, dz, r_c, r_f,
        bc_z="noflux")
    Z = np.asarray(z, dtype=float)[:, None]
    RC = np.asarray(r_c, dtype=float)[None, :]
    gb_raw = (np.maximum(np.asarray(particle) * np.asarray(substrate), 0.0)
              * (np.abs(Z - float(GB_z)) <= 2.0 * W)
              * (RC <= float(r_neck) + W))
    gb_support = normalized_axisym_support(gb_raw, r_c)
    if tj_source_support is None:
        tj_source_support, _ = make_fixed_tj_source_support(
            f, particle, substrate, r_c, z, GB_z, r_neck, W)
    tj_support = np.asarray(tj_source_support, dtype=float)
    tj_norm = _axisym_weighted_sum(tj_support, r_c)
    if not np.isclose(tj_norm, 1.0, rtol=2e-13, atol=2e-15):
        raise ValueError(f"TJ source support is not normalized: {tj_norm:.17g}")
    mu_source = _axisym_weighted_sum(mu * gb_support, r_c)
    mu_sink = _axisym_weighted_sum(mu * tj_support, r_c)
    affinity = mu_source - mu_sink
    return dict(
        mu_source_Pa=float(mu_source), mu_sink_Pa=float(mu_sink),
        transport_affinity_Pa=float(affinity),
        positive_transport_drive_Pa=float(max(0.0, affinity)),
        gb_support_weighted_norm=_axisym_weighted_sum(gb_support, r_c),
        tj_support_weighted_norm=tj_norm,
        definition="instantaneous local mu_GB_source - mu_TJ_sink; mu=deltaF/deltaf in Pa")


def representation_corrected_flux_limited_event_step(
        f, particle, substrate, sink: AxisymSink, hp: HazardParams,
        transport_affinity_pa: float,
        dt: float, dz: float, r_c, z, GB_z_hint, contact_area_m2: float,
        tj_source_support, tj_particle_fraction,
        seconds_per_model_time: float = SECONDS_PER_MODEL_TIME,
        max_increment_fraction_b: float = 0.05,
        max_subincrements: int = 64, event_quota_m: float | None = None):
    """Advance an active event using separate GB-supply and surface channels.

    The Coble time sets a transported-volume capacity

    ``dV/dt = A_GB*b/tau_Coble``.

    Each internal increment inverts the *measured* source volume of the
    qualified body-union operator to obtain ``dq``.  Consequently the imposed
    rigid displacement is caused by the material that can leave the boundary
    during that increment, rather than imposing ``dq`` first and merely
    redistributing whatever overlap happens to result.  A full event is
    integrated in fractions of ``b``.  ``transport_affinity_pa`` must be the
    instantaneous local ``mu_GB_source - mu_TJ_sink`` stress equivalent from
    the current state, not an event-nucleation stress or a saved value from a
    previous simulation.

    The body-union increments only measure/remove the GB-supplied volume.  All
    packets accumulate into one fixed normalized TJ reservoir, which is added
    once after the kinematic integration.  There is deliberately no surface
    diffusion here and no use of ``D_gb`` to choose a surface width.  The
    ordinary ``M_s``-controlled PF step performs all subsequent spreading.
    """
    if not sink.active:
        return f, particle, substrate, False, dict(paused=False, active=False)
    if not 0.0 < max_increment_fraction_b <= 1.0:
        raise ValueError("max_increment_fraction_b must lie in (0,1]")
    if contact_area_m2 <= 0.0:
        raise ValueError("contact_area_m2 must be positive")
    quota = hp.b if event_quota_m is None else float(event_quota_m)
    if not 0.0 < quota <= hp.b:
        raise ValueError("event_quota_m must lie in (0,b]")

    support_norm = _axisym_weighted_sum(tj_source_support, r_c)
    if not np.isclose(support_norm, 1.0, rtol=2e-13, atol=2e-15):
        raise ValueError(f"TJ source support is not normalized: {support_norm:.17g}")
    transport_drive = max(0.0, float(transport_affinity_pa))
    dt_seconds_available = dt * seconds_per_model_time
    if transport_drive <= 0.0 or dt_seconds_available <= 0.0:
        return f, particle, substrate, False, dict(
            paused=True, active=True,
            transport_affinity_Pa=float(transport_affinity_pa),
            positive_transport_drive_Pa=transport_drive,
            delta_sink_this_step=0.0, transported_volume_m3=0.0,
            completed=False)

    xd = 0.5 * hp.GS / 2.0
    tau_Coble = ((xd * xd * hp.kB * hp.T)
                 / (transport_drive * hp.Omega * hp.D_gb) + hp.tau_ex0)
    qdot_nominal = hp.b / tau_Coble
    volume_rate_capacity = contact_area_m2 * qdot_nominal
    factor = 2.0 * math.pi * float(r_c[1] - r_c[0]) * dz
    remaining_initial = max(0.0, quota - sink.current_disp)
    _, _, _, horizon_probe = representation_corrected_union_transfer(
        f, particle, substrate, remaining_initial, dz, r_c, z, GB_z_hint)
    event_source_estimate = horizon_probe["V_source_weighted"] * factor
    event_time_estimate = event_source_estimate / volume_rate_capacity
    time_left = dt_seconds_available
    total_dq = 0.0
    total_volume = 0.0
    total_target_volume = 0.0
    total_time = 0.0
    increments = []
    dose_tolerance = max(1e-14 * hp.b, 1e-12 * quota)
    f_base, particle_base, substrate_base = f, particle, substrate
    cumulative_source_weighted = 0.0
    final_transfer = None

    for _ in range(max_subincrements):
        remaining = max(0.0, quota - sink.current_disp)
        if remaining <= dose_tolerance or time_left <= 1e-15 * dt_seconds_available:
            break
        dq_cap = min(remaining, max_increment_fraction_b * hp.b)
        q_trial = total_dq + dq_cap
        trial = representation_corrected_union_transfer(
            f_base, particle_base, substrate_base, q_trial,
            dz, r_c, z, GB_z_hint)
        source_probe_weighted = trial[3]["V_source_weighted"] - cumulative_source_weighted
        source_probe = source_probe_weighted * factor
        dq_probe = dq_cap
        if source_probe <= 0.0 or dq_probe <= 0.0:
            raise RuntimeError("body-union source vanished during an active flux-limited event")
        source_area_effective = source_probe / dq_probe

        source_at_cap = source_area_effective * dq_cap
        time_for_cap = source_at_cap / volume_rate_capacity
        if time_left >= time_for_cap:
            dq = dq_cap
            target_volume = source_at_cap
            dt_used = time_for_cap
            transfer = trial
        else:
            target_volume = volume_rate_capacity * time_left
            # Invert the measured cumulative V_source(q), not a prescribed
            # displacement rate.  Monotonic bisection is inexpensive here
            # and makes the partial-time volume closure independent of local
            # curvature in V(q).
            lo, hi = 0.0, dq_cap
            transfer = None
            for _bisect in range(52):
                dq_mid = 0.5 * (lo + hi)
                candidate = representation_corrected_union_transfer(
                    f_base, particle_base, substrate_base, total_dq + dq_mid,
                    dz, r_c, z, GB_z_hint)
                candidate_increment = ((candidate[3]["V_source_weighted"]
                                        - cumulative_source_weighted) * factor)
                if candidate_increment < target_volume:
                    lo = dq_mid
                else:
                    hi = dq_mid
                transfer = candidate
            dq = 0.5 * (lo + hi)
            transfer = representation_corrected_union_transfer(
                f_base, particle_base, substrate_base, total_dq + dq,
                dz, r_c, z, GB_z_hint)
            dt_used = time_left

        new_cumulative_weighted = transfer[3]["V_source_weighted"]
        source_volume = (new_cumulative_weighted - cumulative_source_weighted) * factor
        sink.current_disp += dq
        sink.cumulative_disp += dq
        total_dq += dq
        total_volume += source_volume
        cumulative_source_weighted = new_cumulative_weighted
        total_target_volume += target_volume
        total_time += dt_used
        time_left = max(0.0, time_left - dt_used)
        final_transfer = transfer
        increments.append(dict(
            dq_m=dq, fraction_b=dq / hp.b, transport_dt_seconds=dt_used,
            source_volume_m3=source_volume, target_volume_m3=target_volume,
            source_area_effective_m2=source_volume / dq))
        if quota - sink.current_disp <= dose_tolerance:
            break
    else:
        raise RuntimeError("flux-limited event exceeded max_subincrements")

    dose_snap = 0.0
    if 0.0 < quota - sink.current_disp <= dose_tolerance:
        dose_snap = quota - sink.current_disp
        sink.current_disp += dose_snap
        sink.cumulative_disp += dose_snap
        total_dq += dose_snap
    completed = sink.current_disp >= quota - dose_tolerance
    event_progress = min(sink.current_disp, quota)
    if final_transfer is not None:
        f, particle, substrate = final_transfer[:3]
    f, particle, substrate, tj_diag = apply_fixed_tj_source(
        f, particle, substrate, r_c, tj_source_support,
        tj_particle_fraction, cumulative_source_weighted)
    if completed:
        sink.active = False
        sink.current_disp = 0.0
        sink.hazard = 0.0

    diag = dict(
        paused=False, active=sink.active,
        transport_affinity_Pa=float(transport_affinity_pa),
        positive_transport_drive_Pa=transport_drive,
        tau_Coble=tau_Coble, qdot_nominal_m_per_s=qdot_nominal,
        transported_volume_rate_capacity_m3_per_s=volume_rate_capacity,
        event_quota_m=quota, event_quota_fraction_b=quota / hp.b,
        numerical_dose_snap_m=dose_snap,
        delta_sink_this_step=total_dq, event_progress_m=event_progress,
        remaining_to_quota_m=(0.0 if completed else quota - event_progress),
        transported_volume_m3=total_volume,
        target_transported_volume_m3=total_target_volume,
        volume_rate_closure_relative=(total_volume - total_target_volume)
                                     / max(abs(total_target_volume), 1e-300),
        physical_time_consumed_s=total_time,
        physical_time_available_s=dt_seconds_available,
        estimated_event_time_s=event_time_estimate,
        n_subincrements=len(increments), increments=increments,
        tj_source=tj_diag,
        surface_spreading="none during event; delegated to ordinary M_s PF operator",
        max_increment_fraction_b=max_increment_fraction_b,
        completed=completed)
    return f, particle, substrate, completed, diag


def representation_corrected_dynamic_affinity_event(
        f, particle, substrate, hp: HazardParams,
        dz: float, r_c, z, GB_z_hint,
        tj_source_support, tj_particle_fraction,
        affinity_evaluator,
        max_increment_fraction_b: float = 0.01,
        event_quota_m: float | None = None,
        max_subincrements: int = 512,
        max_physical_time_s: float | None = None):
    """Complete a cumulative event while updating transport affinity.

    This deterministic full-event integrator retains a single pre-event body
    reference for every cumulative union state, so changing the packet size
    cannot compound rigid-remap representations.  At the beginning of each
    fractional-``b`` increment, ``affinity_evaluator(f,p,n)`` must return the
    *current* ``transport_affinity_Pa`` and ``contact_area_m2``.  ``D_gb`` then
    controls only the elapsed delivery time for that increment.  The fixed TJ
    reservoir is applied cumulatively at every intermediate state so the
    affinity sees all mass transported so far.

    No PF surface-relaxation step occurs inside this sub-PF-interval event;
    ordinary ``M_s`` evolution remains the separate post-event channel.
    """
    if not 0.0 < max_increment_fraction_b <= 1.0:
        raise ValueError("max_increment_fraction_b must lie in (0,1]")
    quota = hp.b if event_quota_m is None else float(event_quota_m)
    if not 0.0 < quota <= hp.b:
        raise ValueError("event_quota_m must lie in (0,b]")
    support_norm = _axisym_weighted_sum(tj_source_support, r_c)
    if not np.isclose(support_norm, 1.0, rtol=2e-13, atol=2e-15):
        raise ValueError(f"TJ source support is not normalized: {support_norm:.17g}")

    f_base = np.asarray(f, dtype=float)
    particle_base = np.asarray(particle, dtype=float)
    substrate_base = np.asarray(substrate, dtype=float)
    factor = 2.0 * math.pi * float(r_c[1] - r_c[0]) * dz
    cumulative_q = 0.0
    cumulative_source_weighted = 0.0
    total_time = 0.0
    time_left = (math.inf if max_physical_time_s is None
                 else float(max_physical_time_s))
    if time_left <= 0.0:
        raise ValueError("max_physical_time_s must be positive when supplied")
    packets = []
    state = (f_base, particle_base, substrate_base)
    dose_tolerance = max(1e-14 * hp.b, 1e-12 * quota)

    for _ in range(max_subincrements):
        remaining = quota - cumulative_q
        if remaining <= dose_tolerance or time_left <= 1e-15 * max(1.0, total_time):
            break
        current = affinity_evaluator(*state)
        affinity = float(current["transport_affinity_Pa"])
        contact_area = float(current["contact_area_m2"])
        if affinity <= 0.0 or contact_area <= 0.0:
            return *state, False, dict(
                completed=False, paused=True,
                event_progress_m=cumulative_q,
                remaining_to_quota_m=max(0.0, remaining),
                physical_time_consumed_s=total_time,
                packets=packets,
                max_physical_time_s=max_physical_time_s,
                time_window_exhausted=False,
                stop_state=current)
        dq = min(remaining, max_increment_fraction_b * hp.b)
        q_next = cumulative_q + dq
        transfer = representation_corrected_union_transfer(
            f_base, particle_base, substrate_base, q_next,
            dz, r_c, z, GB_z_hint)
        source_next_weighted = transfer[3]["V_source_weighted"]
        source_increment_m3 = (
            source_next_weighted - cumulative_source_weighted) * factor
        if source_increment_m3 <= 0.0:
            raise RuntimeError("cumulative body-union source is not increasing")

        xd = 0.5 * hp.GS / 2.0
        tau_coble = ((xd * xd * hp.kB * hp.T)
                     / (affinity * hp.Omega * hp.D_gb) + hp.tau_ex0)
        qdot_nominal = hp.b / tau_coble
        volume_rate = contact_area * qdot_nominal
        packet_time = source_increment_m3 / volume_rate
        partial_time_window = False
        if packet_time > time_left:
            target_source_m3 = volume_rate * time_left
            fraction = target_source_m3 / source_increment_m3
            dq *= fraction
            q_next = cumulative_q + dq
            transfer = representation_corrected_union_transfer(
                f_base, particle_base, substrate_base, q_next,
                dz, r_c, z, GB_z_hint)
            source_next_weighted = transfer[3]["V_source_weighted"]
            source_increment_m3 = (
                source_next_weighted - cumulative_source_weighted) * factor
            # The qualified subcell cumulative-union source is linear in q.
            # Retain a guarded bisection fallback if a future geometry leaves
            # that regime, rather than exceeding the physical time window.
            if source_increment_m3 > target_source_m3 * (1.0 + 1e-10):
                lo, hi = cumulative_q, q_next
                for _bisect in range(36):
                    q_mid = 0.5 * (lo + hi)
                    candidate = representation_corrected_union_transfer(
                        f_base, particle_base, substrate_base, q_mid,
                        dz, r_c, z, GB_z_hint)
                    candidate_source = (
                        candidate[3]["V_source_weighted"]
                        - cumulative_source_weighted) * factor
                    if candidate_source < target_source_m3:
                        lo = q_mid
                    else:
                        hi = q_mid
                q_next = 0.5 * (lo + hi)
                dq = q_next - cumulative_q
                transfer = representation_corrected_union_transfer(
                    f_base, particle_base, substrate_base, q_next,
                    dz, r_c, z, GB_z_hint)
                source_next_weighted = transfer[3]["V_source_weighted"]
                source_increment_m3 = (
                    source_next_weighted - cumulative_source_weighted) * factor
            packet_time = source_increment_m3 / volume_rate
            partial_time_window = True
        total_time += packet_time
        time_left = max(0.0, time_left - packet_time)

        state = apply_fixed_tj_source(
            *transfer[:3], r_c, tj_source_support, tj_particle_fraction,
            source_next_weighted)[:3]
        packets.append(dict(
            q_start_m=cumulative_q,
            q_end_m=q_next,
            fraction_b=dq / hp.b,
            transport_affinity_Pa=affinity,
            contact_area_m2=contact_area,
            tau_Coble_s=tau_coble,
            qdot_nominal_m_per_s=qdot_nominal,
            source_increment_m3=source_increment_m3,
            transport_dt_seconds=packet_time,
            partial_time_window=partial_time_window,
            state_coordinates=current))
        cumulative_q = q_next
        cumulative_source_weighted = source_next_weighted
    else:
        raise RuntimeError("dynamic-affinity event exceeded max_subincrements")

    if 0.0 < quota - cumulative_q <= dose_tolerance:
        cumulative_q = quota
    completed = cumulative_q >= quota - dose_tolerance
    final_coordinates = affinity_evaluator(*state)
    diag = dict(
        completed=completed,
        paused=False,
        event_quota_m=quota,
        event_quota_fraction_b=quota / hp.b,
        event_progress_m=cumulative_q,
        remaining_to_quota_m=(0.0 if completed else quota - cumulative_q),
        physical_time_consumed_s=total_time,
        n_subincrements=len(packets),
        max_increment_fraction_b=max_increment_fraction_b,
        cumulative_source_weighted=cumulative_source_weighted,
        transported_volume_m3=cumulative_source_weighted * factor,
        packets=packets,
        initial_coordinates=(packets[0]["state_coordinates"] if packets else final_coordinates),
        final_coordinates=final_coordinates,
        transport_affinity_updated_each_increment=True,
        max_physical_time_s=max_physical_time_s,
        time_window_exhausted=bool(
            max_physical_time_s is not None
            and total_time >= max_physical_time_s * (1.0 - 1e-10)
            and not completed),
        cumulative_union_uses_single_pre_event_reference=True,
        surface_spreading="none during event; delegated to ordinary M_s PF operator")
    return *state, completed, diag


def representation_corrected_concurrent_transport_event(
        f, particle, substrate, hp: HazardParams,
        dz: float, r_c, z, GB_z_hint,
        tj_source_support, tj_particle_fraction,
        affinity_evaluator, surface_relaxation_step,
        max_increment_fraction_b: float = 0.001,
        event_quota_m: float | None = None,
        max_subincrements: int = 4096,
        seconds_per_model_time: float = SECONDS_PER_MODEL_TIME,
        attachment_step=None):
    """Couple each GB-delivery packet to surface PF over the same time.

    For packet ``j`` the ordering is

    ``DeltaV_GB -> Deltaq -> conservative TJ arrival -> M_s PF(Delta t_j)``.

    The cumulative body-union geometry is always evaluated from one pre-event
    reference.  Its *increment* between ``q_j`` and ``q_(j+1)`` is applied to
    the current, already surface-relaxed state, followed by exactly the newly
    removed source volume at the fixed arrival support.  With an identity
    surface callback this reduces algebraically to the qualified cumulative
    union plus cumulative reservoir and is independent of packetization.

    By default, arrival uses the legacy additive ``tj_source_support`` and
    mixed ``tj_particle_fraction``.  A shape-preserving replacement may be
    supplied as ``attachment_step(f,p,n,source_increment_weighted)``; it must
    return ``(f,p,n)`` or ``(f,p,n,diag)`` and exactly add that weighted
    volume.  This callback keeps GB timing/body mechanics independent of the
    diffuse-interface attachment representation.

    ``surface_relaxation_step(f,p,n,dt_model)`` receives
    ``dt_model = Delta t_j / seconds_per_model_time``.  It must use the frozen
    ordinary phase-field operator and return the three updated fields.  Thus
    ``D_gb`` controls delivery time while ``M_s`` acts concurrently over the
    mapped interval; the physical qualification of ``seconds_per_model_time``
    is a separate calibration, and diffusivity never selects an arrival width.
    """
    if not 0.0 < max_increment_fraction_b <= 1.0:
        raise ValueError("max_increment_fraction_b must lie in (0,1]")
    quota = hp.b if event_quota_m is None else float(event_quota_m)
    if not 0.0 < quota <= hp.b:
        raise ValueError("event_quota_m must lie in (0,b]")
    if seconds_per_model_time <= 0.0:
        raise ValueError("seconds_per_model_time must be positive")
    if attachment_step is None:
        support_norm = _axisym_weighted_sum(tj_source_support, r_c)
        if not np.isclose(support_norm, 1.0, rtol=2e-13, atol=2e-15):
            raise ValueError(f"TJ source support is not normalized: {support_norm:.17g}")

    base = tuple(np.asarray(field, dtype=float) for field in (f, particle, substrate))
    state = tuple(field.copy() for field in base)
    union_previous = base
    factor = 2.0 * math.pi * float(r_c[1] - r_c[0]) * dz
    mass_initial_weighted = _axisym_weighted_sum(base[0], r_c)
    cumulative_q = 0.0
    cumulative_source_weighted = 0.0
    total_time_s = 0.0
    total_surface_model_time = 0.0
    packets = []
    dose_tolerance = max(1e-14 * hp.b, 1e-12 * quota)

    for _ in range(max_subincrements):
        remaining = quota - cumulative_q
        if remaining <= dose_tolerance:
            break
        before = affinity_evaluator(*state)
        affinity = float(before["transport_affinity_Pa"])
        contact_area = float(before["contact_area_m2"])
        if affinity <= 0.0 or contact_area <= 0.0:
            return *state, False, dict(
                completed=False, paused=True,
                event_progress_m=cumulative_q,
                event_progress_over_b=cumulative_q / hp.b,
                remaining_to_quota_m=max(0.0, remaining),
                physical_time_consumed_s=total_time_s,
                surface_PF_model_time=total_surface_model_time,
                cumulative_source_weighted=cumulative_source_weighted,
                transported_volume_m3=cumulative_source_weighted * factor,
                packets=packets, stop_state=before,
                stop_reason="nonpositive current GB-to-TJ transport affinity",
                cumulative_union_uses_single_pre_event_reference=True,
                surface_relaxation_concurrent_with_each_packet=True)

        dq = min(remaining, max_increment_fraction_b * hp.b)
        q_next = cumulative_q + dq
        transfer = representation_corrected_union_transfer(
            *base, q_next, dz, r_c, z, GB_z_hint)
        union_next = transfer[:3]
        source_next_weighted = transfer[3]["V_source_weighted"]
        source_increment_weighted = source_next_weighted - cumulative_source_weighted
        if source_increment_weighted <= 0.0:
            raise RuntimeError("cumulative body-union source is not increasing")
        source_increment_m3 = source_increment_weighted * factor

        xd = hp.GS / 4.0
        tau_coble = ((xd * xd * hp.kB * hp.T)
                     / (affinity * hp.Omega * hp.D_gb) + hp.tau_ex0)
        qdot_nominal = hp.b / tau_coble
        volume_rate = contact_area * qdot_nominal
        packet_time_s = source_increment_m3 / volume_rate
        packet_model_time = packet_time_s / seconds_per_model_time

        state_after_union = tuple(
            current + (new_union - old_union)
            for current, new_union, old_union
            in zip(state, union_next, union_previous))
        partition_before_arrival = float(np.max(np.abs(
            state_after_union[1] + state_after_union[2] - state_after_union[0])))
        if attachment_step is None:
            attached = apply_fixed_tj_source(
                *state_after_union, r_c, tj_source_support, tj_particle_fraction,
                source_increment_weighted)
        else:
            attached = attachment_step(*state_after_union, source_increment_weighted)
            if not isinstance(attached, (tuple, list)) or len(attached) not in (3, 4):
                raise ValueError("attachment_step must return (f,p,n) or (f,p,n,diag)")
        state_after_arrival = tuple(np.asarray(field, dtype=float) for field in attached[:3])
        attachment_diag = dict(attached[3]) if len(attached) == 4 else {}
        arrival_increment_weighted = _axisym_weighted_sum(
            state_after_arrival[0] - state_after_union[0], r_c)
        arrival_closure_error = arrival_increment_weighted - source_increment_weighted
        if not np.isclose(
                arrival_increment_weighted, source_increment_weighted,
                rtol=1e-9, atol=1e-14 * max(abs(source_increment_weighted), 1e-300)):
            raise ValueError(
                "attachment_step did not add exactly the transported weighted volume")
        partition_after_arrival = float(np.max(np.abs(
            state_after_arrival[1] + state_after_arrival[2]
            - state_after_arrival[0])))
        if partition_after_arrival > max(1e-10, 10.0 * partition_before_arrival):
            raise ValueError("attachment_step violated grain-ownership partition")
        attachment_diag.update(
            event_source_increment_weighted=source_increment_weighted,
            event_added_increment_weighted=arrival_increment_weighted,
            event_closure_error_weighted=arrival_closure_error,
            event_partition_residual=partition_after_arrival)
        arrival = affinity_evaluator(*state_after_arrival)
        relaxed = surface_relaxation_step(*state_after_arrival, packet_model_time)
        if not isinstance(relaxed, (tuple, list)) or len(relaxed) != 3:
            raise ValueError("surface_relaxation_step must return (f,particle,substrate)")
        state = tuple(np.asarray(field, dtype=float).copy() for field in relaxed)
        if any(field.shape != base[0].shape for field in state):
            raise ValueError("surface_relaxation_step returned a mismatched field shape")
        after = affinity_evaluator(*state)
        cumulative_q = q_next
        cumulative_source_weighted = source_next_weighted
        union_previous = union_next
        total_time_s += packet_time_s
        total_surface_model_time += packet_model_time
        mass_weighted = _axisym_weighted_sum(state[0], r_c)
        packets.append(dict(
            q_start_m=q_next - dq, q_end_m=q_next,
            q_start_over_b=(q_next - dq) / hp.b,
            q_end_over_b=q_next / hp.b,
            dq_m=dq, fraction_b=dq / hp.b,
            transport_affinity_before_Pa=affinity,
            contact_area_m2=contact_area,
            x_d_m=xd, tau_Coble_s=tau_coble,
            Vdot_GB_m3_per_s=volume_rate,
            source_increment_weighted=source_increment_weighted,
            source_increment_m3=source_increment_m3,
            transport_dt_seconds=packet_time_s,
            surface_PF_dt_model=packet_model_time,
            coordinates_before=before,
            coordinates_after_arrival=arrival,
            coordinates_after_surface_PF=after,
            attachment=attachment_diag,
            partition_residual_before_arrival=partition_before_arrival,
            partition_residual_after_surface_PF=float(np.max(np.abs(
                state[1] + state[2] - state[0]))),
            mass_relative_error_after_surface_PF=(
                mass_weighted - mass_initial_weighted)
                / max(abs(mass_initial_weighted), 1e-300)))
    else:
        raise RuntimeError("concurrent transport event exceeded max_subincrements")

    if 0.0 < quota - cumulative_q <= dose_tolerance:
        cumulative_q = quota
    completed = cumulative_q >= quota - dose_tolerance
    final = affinity_evaluator(*state)
    return *state, completed, dict(
        completed=completed, paused=False,
        event_quota_m=quota, event_quota_fraction_b=quota / hp.b,
        event_progress_m=cumulative_q,
        event_progress_over_b=cumulative_q / hp.b,
        remaining_to_quota_m=(0.0 if completed else quota - cumulative_q),
        physical_time_consumed_s=total_time_s,
        surface_PF_model_time=total_surface_model_time,
        n_subincrements=len(packets), packets=packets,
        cumulative_source_weighted=cumulative_source_weighted,
        transported_volume_m3=cumulative_source_weighted * factor,
        final_coordinates=final,
        cumulative_union_uses_single_pre_event_reference=True,
        incremental_arrival_is_exact_source_increment=True,
        attachment_operator=(
            "legacy additive fixed support" if attachment_step is None
            else "caller-supplied conservative interface attachment"),
        surface_relaxation_concurrent_with_each_packet=True,
        D_gb_controls_arrival_time_only=True,
        M_s_surface_time_equals_transport_time=True)


def smooth_union_ownership_transfer(
        particle_shifted, substrate_stationary, r_c, z, GB_z_hint,
        redistribution_fn=None, dt_seconds=None):
    """INVALID HISTORICAL PATH: probabilistic smooth-union projection.

    .. deprecated:: recovery
       Retained only for regression/provenance.  Use the cumulative
       kinematic-body union, current chemical-potential GB supply, fixed TJ
       reservoir, and ordinary ``M_s`` evolution in
       :func:`representation_corrected_flux_limited_event_step`.

    Unlike :func:`rigid_translation_sink_step`, this construction never forms
    ``clip(particle + substrate)``.  The two diffuse occupancies are combined
    with the probabilistic union ``p + n - p*n``.  Grain ownership is then
    projected smoothly using the pre-projection occupancy fractions, so that
    ``particle_new + substrate_new == f_union`` pointwise.

    The material removed by non-overlap is exactly ``p*n`` in field units.
    Its axisymmetric weighted volume is passed unchanged to the existing
    free-surface/TJ redistribution interface.  This keeps rigid translation,
    GB source removal, and TJ deposition as separately inspectable states.
    """
    warnings.warn(
        "smooth_union_ownership_transfer is an invalid historical recovery "
        "path; use representation_corrected_flux_limited_event_step",
        DeprecationWarning, stacklevel=2)
    p = np.clip(np.asarray(particle_shifted), 0.0, 1.0)
    n = np.clip(np.asarray(substrate_stationary), 0.0, 1.0)
    f_union = p + n - p * n

    occupancy_sum = p + n
    p_fraction = np.divide(p, occupancy_sum, out=np.zeros_like(p), where=occupancy_sum > 1e-30)
    particle_owned = f_union * p_fraction
    substrate_owned = f_union - particle_owned

    removed = p * n
    V_removed_weighted = _axisym_weighted_sum(removed, r_c)
    redist_diag = dict(skipped=True)
    f_final = f_union
    particle_final = particle_owned
    substrate_final = substrate_owned
    if redistribution_fn is not None and V_removed_weighted > 1e-30:
        f_final, particle_final, substrate_final, redist_diag = redistribution_fn(
            f_union, particle_owned, substrate_owned, r_c, z, GB_z_hint,
            V_removed_weighted, dt_seconds=dt_seconds)

    return f_final, particle_final, substrate_final, dict(
        f_union=f_union,
        particle_owned=particle_owned,
        substrate_owned=substrate_owned,
        removed=removed,
        V_removed_weighted=V_removed_weighted,
        union_bounds_residual=max(float(np.max(-f_union)), float(np.max(f_union - 1.0)), 0.0),
        ownership_residual=float(np.max(np.abs(particle_owned + substrate_owned - f_union))),
        final_partition_residual=float(np.max(np.abs(particle_final + substrate_final - f_final))),
        redistribution=redist_diag,
    )


def candidate_a_rigid_translation_sink_step(
        f, particle, substrate, sink: AxisymSink, hp: HazardParams, sigma_s: float,
        dt: float, dz: float, r_c, z, GB_z_hint,
        seconds_per_model_time: float = SECONDS_PER_MODEL_TIME,
        redistribution_fn=None, shift_order=3):
    """INVALID HISTORICAL PATH retained only for regression/provenance.

    .. deprecated:: recovery
       This path uses nucleation-style ``sigma_s``, interpolation-based grain
       shifting, and the invalid smooth-union projection.  It must not be used
       for scientific PR events.
    """
    warnings.warn(
        "candidate_a_rigid_translation_sink_step is an invalid historical "
        "recovery path; use representation_corrected_flux_limited_event_step",
        DeprecationWarning, stacklevel=2)
    if not sink.active:
        return f, particle, substrate, False, dict(paused=False, active=False)

    sigma_drive = max(0.0, sigma_s)
    dt_seconds = dt * seconds_per_model_time
    remaining = max(0.0, hp.b - sink.current_disp)
    if sigma_drive <= 0.0 or remaining <= 0.0:
        return f, particle, substrate, False, dict(
            paused=True, active=True, sigma_drive=sigma_drive,
            delta_sink_this_step=0.0, completed=False)

    xd = 0.5 * hp.GS / 2
    tau_Coble = (xd * xd * hp.kB * hp.T) / (sigma_drive * hp.Omega * hp.D_gb) + hp.tau_ex0
    v_event = hp.b / tau_Coble
    d_delta_requested = min(v_event * dt_seconds, remaining)

    particle_shifted = ndimage.shift(
        particle, shift=(-d_delta_requested / dz, 0.0),
        order=shift_order, mode="nearest")
    np.clip(particle_shifted, 0.0, 1.0, out=particle_shifted)
    substrate_stationary = substrate.copy()
    f_new, particle_new, substrate_new, projection = smooth_union_ownership_transfer(
        particle_shifted, substrate_stationary, r_c, z, GB_z_hint,
        redistribution_fn=redistribution_fn, dt_seconds=dt_seconds)

    sink.current_disp += d_delta_requested
    sink.cumulative_disp += d_delta_requested
    completed = sink.current_disp >= hp.b - 1e-15
    if completed:
        sink.active = False
        sink.current_disp = 0.0
        sink.hazard = 0.0

    diag = dict(
        paused=False, active=sink.active, sigma_drive=sigma_drive,
        tau_Coble=tau_Coble, v_event=v_event,
        requested_d_delta=d_delta_requested,
        delta_sink_this_step=d_delta_requested,
        delta_event=(0.0 if completed else sink.current_disp),
        remaining_to_b=max(0.0, hp.b - sink.current_disp),
        shift_m=d_delta_requested,
        V_removed_weighted=projection["V_removed_weighted"],
        union_bounds_residual=projection["union_bounds_residual"],
        ownership_residual=projection["ownership_residual"],
        e1e2f_residual=projection["final_partition_residual"],
        redistribution=projection["redistribution"], completed=completed)
    return f_new, particle_new, substrate_new, completed, diag


def gb_swept_volume_erosion_transfer(
        particle_shifted, substrate_stationary, displacement_m, contact_radius_m,
        r_c, z, GB_z_hint, redistribution_fn=None, dt_seconds=None):
    """INVALID HISTORICAL PATH: prescribed swept-volume GB erosion.

    .. deprecated:: recovery
       Retained only for regression/provenance.  It prescribes a swept volume
       instead of measuring the cumulative kinematic-body union and closing
       displacement causally through GB-delivered mass.

    The source shape is the smooth diffuse-GB overlap ``particle*neighbor``;
    only its normalization comes from the swept contact geometry.  Thus the
    transferred volume is not inferred from clipped occupancy overflow.
    """
    warnings.warn(
        "gb_swept_volume_erosion_transfer is an invalid historical recovery "
        "path; use representation_corrected_flux_limited_event_step",
        DeprecationWarning, stacklevel=2)
    p = np.clip(np.asarray(particle_shifted), 0.0, 1.0)
    n = np.clip(np.asarray(substrate_stationary), 0.0, 1.0)
    raw = p + n
    dr = float(r_c[1] - r_c[0])
    dz = float(z[1] - z[0])
    swept_volume_m3 = math.pi * contact_radius_m ** 2 * displacement_m
    target_weighted = swept_volume_m3 / (2.0 * math.pi * dr * dz)

    source_weight = p * n
    source_norm = _axisym_weighted_sum(source_weight, r_c)
    if source_norm <= 1e-30:
        raise RuntimeError("Candidate B has no diffuse GB source support")
    removed = source_weight * (target_weighted / source_norm)
    f_eroded = raw - removed
    if float(np.min(f_eroded)) < -1e-12 or float(np.max(f_eroded)) > 1.0 + 1e-12:
        raise RuntimeError(
            "Candidate B explicit swept-volume erosion did not produce a bounded total-solid field: "
            f"range=({np.min(f_eroded):.6e},{np.max(f_eroded):.6e})")
    # Floating roundoff only; this is not the historical overlap clip.
    f_eroded = np.minimum(1.0, np.maximum(0.0, f_eroded))
    ownership_sum = raw
    p_fraction = np.divide(p, ownership_sum, out=np.zeros_like(p), where=ownership_sum > 1e-30)
    p_eroded = f_eroded * p_fraction
    n_eroded = f_eroded - p_eroded

    f_final, p_final, n_final = f_eroded, p_eroded, n_eroded
    redist_diag = dict(skipped=True)
    if redistribution_fn is not None and target_weighted > 1e-30:
        f_final, p_final, n_final, redist_diag = redistribution_fn(
            f_eroded, p_eroded, n_eroded, r_c, z, GB_z_hint,
            target_weighted, dt_seconds=dt_seconds)
    return f_final, p_final, n_final, dict(
        f_eroded=f_eroded, particle_eroded=p_eroded, substrate_eroded=n_eroded,
        removed=removed, swept_volume_m3=swept_volume_m3,
        target_weighted=target_weighted,
        removed_weighted=_axisym_weighted_sum(removed, r_c),
        eroded_bounds=(float(np.min(f_eroded)), float(np.max(f_eroded))),
        erosion_partition_residual=float(np.max(np.abs(p_eroded + n_eroded - f_eroded))),
        final_partition_residual=float(np.max(np.abs(p_final + n_final - f_final))),
        redistribution=redist_diag)
