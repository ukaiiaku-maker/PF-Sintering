"""Zero-storage triple-junction transport node for asymmetric PR branches.

The real PR trough is not mirror symmetric, so its two surface fluxes are not
prescribed as equal.  A sharp node potential is instead determined by local
volume conservation between the GB supply and the two branch conductances.
"""
from __future__ import annotations

import math

from scipy import optimize

from pf_sintering.model_time_transport import ModelTimeGBTransport


def branch_node_conductances(
        branches, surface_flux_mobility_m6_per_J_model_time: float) -> dict:
    """Return ``C=2*pi*r_TJ*K_s/ds_TJ`` and first-cell potentials."""
    mobility = float(surface_flux_mobility_m6_per_J_model_time)
    if not math.isfinite(mobility) or mobility <= 0.0:
        raise ValueError("surface flux mobility must be positive and finite")
    if len(branches) != 2 or {branch.side for branch in branches} != {
            "positive", "negative"}:
        raise ValueError("one positive and one negative branch are required")
    out = {}
    for branch in branches:
        distance = float(branch.s_centers_m[0])
        radius = float(branch.r_faces_m[0])
        conductance = 2.0 * math.pi * radius * mobility / distance
        if not all(math.isfinite(value) and value > 0.0 for value in (
                distance, radius, conductance)):
            raise ValueError("branch node geometry and conductance must be positive")
        out[branch.side] = dict(
            conductance_m3_per_Pa_model_time=conductance,
            first_cell_mu_Pa=float(branch.mu_Pa[0]),
            node_to_first_center_distance_m=distance,
            r_TJ_face_m=radius)
    return out


def _surface_only_node(terms):
    total = sum(row["conductance_m3_per_Pa_model_time"] for row in terms.values())
    weighted = sum(
        row["conductance_m3_per_Pa_model_time"] * row["first_cell_mu_Pa"]
        for row in terms.values())
    return weighted / total, total


def _finish_node(mu_node, gb_rate, terms, *, mode, mu_gb=None,
                 gb_conductance=None, nonlinear=False):
    branch_rates = {
        side: row["conductance_m3_per_Pa_model_time"]
        * (mu_node - row["first_cell_mu_Pa"])
        for side, row in terms.items()}
    for side, row in terms.items():
        row["one_sided_node_gradient_Pa_per_m"] = (
            (mu_node - row["first_cell_mu_Pa"])
            / row["node_to_first_center_distance_m"])
    surface_sum = sum(branch_rates.values())
    closure = surface_sum - gb_rate
    scale = max(
        abs(surface_sum), abs(gb_rate),
        *(abs(rate) for rate in branch_rates.values()), 1e-300)
    normalized_signed_rates = (
        {side: rate / gb_rate for side, rate in branch_rates.items()}
        if gb_rate > 0.0 else {side: None for side in branch_rates})
    mu_off, _ = _surface_only_node(terms)
    return dict(
        node_mode=mode,
        mu_TJ_Pa=float(mu_node),
        mu_TJ_OFF_Pa=float(mu_off),
        mu_GB_Pa=(None if mu_gb is None else float(mu_gb)),
        transport_affinity_Pa=(
            None if mu_gb is None else float(mu_gb - mu_node)),
        Vdot_GB_m3_per_model_time=float(gb_rate),
        branch_volume_rates_m3_per_model_time=branch_rates,
        branch_rate_over_net_GB=normalized_signed_rates,
        branch=terms,
        surface_rate_sum_m3_per_model_time=float(surface_sum),
        zero_storage_closure_m3_per_model_time=float(closure),
        zero_storage_closure_relative=float(closure / scale),
        L_GB_m3_per_Pa_model_time=gb_conductance,
        nonlinear_tau_ex_solve=bool(nonlinear),
        imposed_half_partition=False,
        endpoint_width_m=None,
        Young_Herring_force_balance_imposed=False,
        kinetic_time_basis="model_time_demonstration")


def solve_prescribed_rate_tj_node(
        branches, surface_flux_mobility_m6_per_J_model_time: float,
        total_incoming_volume_rate_m3_per_model_time: float):
    """Solve the asymmetric node for a prescribed total GB input rate."""
    incoming = float(total_incoming_volume_rate_m3_per_model_time)
    if not math.isfinite(incoming):
        raise ValueError("prescribed total input must be finite")
    terms = branch_node_conductances(
        branches, surface_flux_mobility_m6_per_J_model_time)
    mu_off, total_conductance = _surface_only_node(terms)
    mu_node = mu_off + incoming / total_conductance
    return _finish_node(
        mu_node, incoming, terms, mode="prescribed total GB rate")


def solve_model_time_tj_node(
        branches, surface_flux_mobility_m6_per_J_model_time: float,
        mu_GB_Pa: float, contact_area_m2: float,
        transport: ModelTimeGBTransport, gb_path_active: bool = True):
    """Couple model-time GB supply and two asymmetric surface branches.

    With ``tau_ex=0`` this recovers the analytical conductance-weighted node.
    For nonzero ``tau_ex`` the same zero-storage equation is solved as a
    monotonic scalar root.  With the sink OFF, ``L_GB=0`` and the node is the
    conductance-weighted surface-only value; the two branches may exchange
    equal and opposite volume through the node.
    """
    mu_gb = float(mu_GB_Pa)
    area = float(contact_area_m2)
    if not math.isfinite(mu_gb) or not math.isfinite(area) or area <= 0.0:
        raise ValueError("mu_GB must be finite and contact area positive")
    terms = branch_node_conductances(
        branches, surface_flux_mobility_m6_per_J_model_time)
    mu_off, surface_conductance = _surface_only_node(terms)
    L_gb = area * transport.b_m / transport.K_gb_Pa_model_time
    if not gb_path_active or mu_gb <= mu_off:
        return _finish_node(
            mu_off, 0.0, terms,
            mode=("sink OFF surface-only node" if not gb_path_active
                  else "active GB path with nonpositive drive"),
            mu_gb=mu_gb, gb_conductance=(0.0 if not gb_path_active else L_gb),
            nonlinear=transport.tau_ex_model > 0.0)

    if transport.tau_ex_model == 0.0:
        mu_node = (
            L_gb * mu_gb
            + sum(row["conductance_m3_per_Pa_model_time"]
                  * row["first_cell_mu_Pa"] for row in terms.values())) / (
                      L_gb + surface_conductance)
        gb_rate = L_gb * (mu_gb - mu_node)
        return _finish_node(
            mu_node, gb_rate, terms, mode="coupled linear GB/surface node",
            mu_gb=mu_gb, gb_conductance=L_gb, nonlinear=False)

    def residual(mu_node):
        affinity = mu_gb - mu_node
        gb_rate = transport.volume_rate_m3_per_model_time(affinity, area)
        surface_rate = surface_conductance * (mu_node - mu_off)
        return gb_rate - surface_rate

    mu_node = optimize.brentq(
        residual, mu_off, mu_gb, xtol=1e-10 * max(1.0, abs(mu_gb), abs(mu_off)),
        rtol=4.0 * math.ulp(1.0), maxiter=100)
    affinity = mu_gb - mu_node
    gb_rate = transport.volume_rate_m3_per_model_time(affinity, area)
    return _finish_node(
        mu_node, gb_rate, terms, mode="coupled nonlinear tau_ex GB/surface node",
        mu_gb=mu_gb, gb_conductance=L_gb, nonlinear=True)
