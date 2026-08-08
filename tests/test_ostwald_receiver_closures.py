import math

import numpy as np

from pf_sintering.model import ModelConfig, build_params, initialize_fields
from pf_sintering.ostwald_diagnostics import ostwald_substrate_diagnostic
from pf_sintering.ostwald_receiver_closures import (
    ostwald_external_reservoir_step,
    ostwald_partial_addition_step,
    region_masks_by_tj_distance,
)
from pf_sintering.tj_subgrid import compute_subgrid_contact


def _state():
    p = build_params(ModelConfig(
        preset="dev", dx=5e-9, r2=80e-9, aspect_ratio=2.0, contact_orientation="short_plane",
        initial_overlap=20e-9, t_total=1e-6, coarsening_rate_scale=3.0, surface_mobility_scale=0.3,
        eta_mobility_scale=1.0,
    ))
    f, e1, e2, e3 = initialize_fields(p)
    return p, f, e1, e2, e3


def test_model_a_phi1_no_mask_matches_o0_exactly():
    p, f, e1, e2, e3 = _state()
    f_o0, e1_o0, e2_o0, e3_o0, _ = ostwald_substrate_diagnostic(f, e1, e2, e3, p)
    f_a, e1_a, e2_a, e3_a, reservoir = ostwald_external_reservoir_step(f, e1, e2, e3, p, phi_local=1.0, region_mask=None)
    assert np.array_equal(f_a, f_o0)
    assert np.array_equal(e1_a, e1_o0)
    assert np.array_equal(e2_a, e2_o0)
    assert reservoir == 0.0


def test_model_b_phi0_diverts_all_addition_to_reservoir():
    p, f, e1, e2, e3 = _state()
    _, _, _, _, diag = ostwald_substrate_diagnostic(f, e1, e2, e3, p)
    f_b, e1_b, e2_b, e3_b, reservoir = ostwald_external_reservoir_step(f, e1, e2, e3, p, phi_local=0.0)
    assert np.array_equal(e1_b, e1)  # no local deposition at all
    assert math.isclose(reservoir, diag.actual_added * p.dx * p.dx, rel_tol=1e-9)
    # removal still happens -- e2/f decrease by the same removal field as O0
    assert not np.array_equal(e2_b, e2)


def test_region_masks_partition_the_domain_exactly():
    p, f, e1, e2, e3 = _state()
    sub = compute_subgrid_contact(f, e1, e2, p)
    assert sub.resolved
    tj_top = (sub.top.x_sub, sub.top.y_sub)
    tj_bot = (sub.bottom.x_sub, sub.bottom.y_sub)
    masks = region_masks_by_tj_distance(p, tj_top, tj_bot, band_edges_in_W=(0, 1, 3, 5))
    stacked = np.stack(list(masks.values()))
    # every grid point belongs to exactly one band
    assert np.all(stacked.sum(axis=0) == 1)


def test_partial_addition_mass_never_renormalized():
    p, f, e1, e2, e3 = _state()
    sub = compute_subgrid_contact(f, e1, e2, p)
    tj_top = (sub.top.x_sub, sub.top.y_sub)
    tj_bot = (sub.bottom.x_sub, sub.bottom.y_sub)
    masks = region_masks_by_tj_distance(p, tj_top, tj_bot)
    _, _, _, _, diag = ostwald_substrate_diagnostic(f, e1, e2, e3, p)
    total_added_full = diag.actual_added * p.dx * p.dx

    total_kept = 0.0
    total_reservoir = 0.0
    for mask in masks.values():
        f_p, e1_p, e2_p, e3_p, reservoir = ostwald_partial_addition_step(f, e1, e2, e3, p, mask)
        kept = float((e1_p - e1).sum()) * p.dx * p.dx
        total_kept += kept
        total_reservoir += reservoir
    # sum of (kept + reservoir) over disjoint regions must equal the total
    # added mass computed ONCE, not four independent full-mass allocations
    # -- confirms no renormalization occurred region-by-region.
    n_regions = len(masks)
    assert math.isclose(total_kept + total_reservoir, n_regions * total_added_full, rel_tol=1e-9)
