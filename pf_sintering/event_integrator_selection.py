"""Physical-timescale selection between packet and quasi-static 1b events."""
from __future__ import annotations

import math


VALID_REQUESTS = ("auto", "packet", "quasistatic")


def select_event_integrator(
        requested: str, *, tau_GB: float, tau_surface: float,
        threshold_low: float = 0.1, threshold_high: float = 10.0,
        intermediate_fallback: str = "packet") -> dict:
    """Return one frozen per-event propagator decision.

    The two input times may use any common unit.  Manual choices bypass the
    automatic thresholds exactly.  The intermediate regime defaults to the
    packet method because it assumes less timescale separation.
    """
    requested = str(requested).lower()
    if requested not in VALID_REQUESTS:
        raise ValueError(f"event integrator must be one of {VALID_REQUESTS}")
    values = (tau_GB, tau_surface, threshold_low, threshold_high)
    if any(not math.isfinite(float(value)) or float(value) <= 0.0
           for value in values):
        raise ValueError("times and selection thresholds must be positive and finite")
    if not float(threshold_low) < float(threshold_high):
        raise ValueError("selection thresholds must increase")
    if intermediate_fallback not in ("packet", "quasistatic"):
        raise ValueError("intermediate fallback must be packet or quasistatic")
    ratio = float(tau_GB)/float(tau_surface)
    warning = False
    if requested != "auto":
        selected = requested
        reason = "manual override"
    elif ratio <= float(threshold_low):
        selected = "packet"
        reason = "GB delivery is asymptotically faster than surface relaxation"
    elif ratio >= float(threshold_high):
        selected = "quasistatic"
        reason = "surface relaxation is asymptotically faster than GB delivery"
    else:
        selected = intermediate_fallback
        warning = True
        reason = (
            "INTERMEDIATE TIMESCALE REGIME: neither packet nor quasi-static "
            "separation is asymptotically well justified")
    return dict(
        event_integrator_requested=requested,
        event_integrator_selected=selected,
        tau_GB_estimate=float(tau_GB),
        tau_surface_estimate=float(tau_surface),
        timescale_ratio=ratio,
        selection_threshold_low=float(threshold_low),
        selection_threshold_high=float(threshold_high),
        intermediate_timescale_warning=bool(warning),
        intermediate_timescale_fallback=intermediate_fallback,
        selection_reason=reason)


def estimate_and_select_event_integrator(
        requested: str, *, transport, affinity_Pa: float,
        surface_length_m: float, surface_B_m4_per_model_time: float,
        seconds_per_model_time: float, threshold_low: float = 0.1,
        threshold_high: float = 10.0,
        intermediate_fallback: str = "packet") -> dict:
    """Evaluate the physical GB and Mullins times and select one integrator."""
    length = float(surface_length_m)
    coefficient = float(surface_B_m4_per_model_time)
    seconds = float(seconds_per_model_time)
    if length <= 0.0 or coefficient <= 0.0 or seconds <= 0.0:
        raise ValueError("surface scale, Mullins B, and time mapping must be positive")
    tau_gb_model = float(transport.tau_gb_model(float(affinity_Pa)))
    tau_surface_model = length**4/coefficient
    decision = select_event_integrator(
        requested, tau_GB=tau_gb_model*seconds,
        tau_surface=tau_surface_model*seconds,
        threshold_low=threshold_low, threshold_high=threshold_high,
        intermediate_fallback=intermediate_fallback)
    decision.update(
        tau_GB_estimate_model_time=tau_gb_model,
        tau_surface_estimate_model_time=tau_surface_model,
        timescale_estimate_common_unit="physical seconds",
        surface_active_length_m=length,
        surface_Mullins_B_m4_per_model_time=coefficient,
        surface_Mullins_B_m4_per_s=coefficient/seconds,
        GB_timescale_definition="b/qdot(A_start)",
        surface_timescale_definition="L_surface^4/B_surface")
    return decision
