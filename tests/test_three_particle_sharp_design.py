import math

import numpy as np

from pf_sintering.three_particle_sharp_design import (
    SharpDesign, admissibility, center_squared_radius_coefficients,
    contact_state, evaluate_even_squared_radius, evaluate_polynomial,
    inverse_target_radius, outer_squared_radius_coefficients,
)


def example():
    return SharpDesign(120e-9, .75, 64e-9, 1.0, 160., 150., .5, .25)


def test_center_profile_and_loading_satisfy_exact_volume_constraint():
    d=example(); u=np.linspace(-1.,1.,20001)
    coeff=center_squared_radius_coefficients(d)
    for x in (0.,.01,.02,.05,.10,.15,.20):
        loaded=coeff.copy();loaded[0]-=x*d.Vc0_m3/(math.pi*d.Lc_m)
        volume=math.pi*.5*d.Lc_m*np.trapezoid(
            evaluate_even_squared_radius(loaded,abs(u)),u)
        np.testing.assert_allclose(volume,d.Vc0_m3*(1-x),rtol=2e-9)
        np.testing.assert_allclose(
            evaluate_even_squared_radius(loaded,np.array([1.]))[0],
            contact_state(d,x)["r_TJ_m"]**2,rtol=1e-13)


def test_initial_contact_jet_and_exact_production_mean_are_recovered():
    d=example(); s=contact_state(d,0.)
    assert s["center_theta_deg"] == 160.
    assert s["outer_theta_deg"] == 150.
    np.testing.assert_allclose(s["center_curvature_per_m"],d.k_center0_per_m)
    np.testing.assert_allclose(s["outer_curvature_per_m"],d.k_outer0_per_m)
    expected=.5*(s["sigma_center_side_Pa"]+s["sigma_outer_side_Pa"])
    np.testing.assert_allclose(s["sigma_GB_Pa"],expected,rtol=1e-14)
    np.testing.assert_allclose(s["sigma_GB_Pa"],s["sigma_kappa_Pa"]+s["sigma_TJ_Pa"])


def test_outer_particle_plus_neck_profile_closes_and_has_exact_volume():
    d=example(); t=np.linspace(0.,1.,20001)
    for x in (0.,.20):
        coeff,length=outer_squared_radius_coefficients(d,x)
        g=evaluate_polynomial(coeff,t)
        volume=math.pi*length*np.trapezoid(g,t)
        target=d.Vo0_m3+.5*x*d.Vc0_m3
        np.testing.assert_allclose(volume,target,rtol=2e-9)
        assert abs(g[-1]) < 1e-24
        np.testing.assert_allclose(g[0],contact_state(d,x)["r_TJ_m"]**2)


def test_inverse_radius_round_trip_and_requested_reference_values():
    theta=160.; curvature=3.36e6
    expected={20.:126.47359841766368,25.:104.17571435249026,
              30.:88.56184829246475}
    for stress_MPa,radius_nm in expected.items():
        radius=inverse_target_radius(stress_MPa*1e6,curvature,theta,theta)
        np.testing.assert_allclose(radius*1e9,radius_nm,rtol=1e-12)
        recovered=-curvature+3*math.sin(math.radians(theta)/2)/radius
        np.testing.assert_allclose(recovered,stress_MPa*1e6,rtol=1e-14)


def test_resolution_gate_rejects_contact_that_collapses_before_20_percent():
    d=SharpDesign(100e-9,.65,40e-9,.60,140.,140.,0.,.25)
    a=admissibility(d)
    assert not a["admissible"]
    assert "topology" in a["reason"]
