import numpy as np

from pf_sintering.three_particle_sharp_design import SharpDesign
from pf_sintering.three_particle_sharp_initial import sharp_radius
from scripts.three_particle_initial_state_design import local_headroom, perturb


def design():
    return SharpDesign(
        Ro_m=180e-9, Rc_over_Ro=.55, Lc_m=138.6e-9,
        rTJ_over_Rc=.75, theta_center_deg=160., theta_outer_deg=160.,
        kRc_center=0., kRo_outer=0., W_m=4e-9)


def test_sharp_radius_is_symmetric_and_hits_the_selected_contact_radius():
    d = design()
    z = np.array([-.5*d.Lc_m, 0., .5*d.Lc_m])
    radius = sharp_radius(z, d)
    assert radius[0] == radius[2]
    np.testing.assert_allclose(radius[0], d.rTJ0_m)
    assert radius[1] > radius[0]


def test_center_outer_exchange_conserves_sharp_three_particle_volume():
    d = design()
    changed = perturb(d, "center_outer_exchange", .002)
    before = 2*d.Ro_m**3 + d.Rc_m**3
    after = 2*changed.Ro_m**3 + changed.Rc_m**3
    np.testing.assert_allclose(after, before, rtol=5e-15)


def test_selected_state_has_independent_energy_lowering_loading_headroom():
    import json
    from pathlib import Path

    manifest = json.loads(Path(
        "docs/three_particle/production_screen/bicrystal_launch_manifest.json"
    ).read_text())
    accessible = [row for row in local_headroom(design(), manifest)
                  if row["accessible_loading"]]
    assert len(accessible) >= 2
    assert all(row["directional_minus_dE_dq_J"] > 0 for row in accessible)
    assert all(row["directional_dsigma_dq_Pa"] > 0 for row in accessible)
