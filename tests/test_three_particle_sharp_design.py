import math

import numpy as np

from pf_sintering.three_particle_sharp_design import (
    SharpDesign, admissibility, center_squared_radius_coefficients,
    contact_state, evaluate_even_squared_radius, evaluate_polynomial,
    inverse_target_radius, outer_squared_radius_coefficients,
    sharp_free_energy, surface_diffusion_projection,
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


def test_sharp_energy_is_exact_sum_of_surface_and_two_gb_terms():
    d=example();gamma_gb=.34729635533386083
    e=sharp_free_energy(d,.1,gamma_gb,801)
    np.testing.assert_allclose(e["surface_energy_J"],d.gamma_s*e["surface_area_m2"])
    np.testing.assert_allclose(e["GB_energy_J"],gamma_gb*e["GB_area_m2"])
    np.testing.assert_allclose(e["free_energy_J"],e["surface_energy_J"]+e["GB_energy_J"])


def test_surface_projection_obeys_mobility_scaling_and_flux_closure():
    # A geometry on the thermodynamically favorable signed-curvature branch.
    d=SharpDesign(100e-9,.65,8*4e-9,1.6,160.,160.,
                  -10e6*(100e-9*.65),-10e6*100e-9)
    kwargs=dict(gamma_gb=.34729635533386083,points_per_branch=801)
    p=surface_diffusion_projection(d,.1,mobility_m6_per_J_s=4e-32,**kwargs)
    q=surface_diffusion_projection(d,.1,mobility_m6_per_J_s=8e-32,**kwargs)
    assert p["dF_dx_J"] < 0 and p["xdot_per_s"] > 0
    assert p["relative_half_chain_closure"] < 3e-6
    np.testing.assert_allclose(q["zeta_J_s"],.5*p["zeta_J_s"],rtol=1e-14)
    np.testing.assert_allclose(q["xdot_per_s"],2*p["xdot_per_s"],rtol=1e-14)


def test_projection_rejects_assumed_shrinkage_thermodynamically_by_sign():
    # A resolved geometry from the old stress-only screen whose energy rises
    # along the prescribed center-shrink path.
    d=SharpDesign(120e-9,.65,48e-9,1.2,160.,160.,-3.,-3.)
    p=surface_diffusion_projection(d,.1,gamma_gb=.34729635533386083,
        mobility_m6_per_J_s=6e-34/.01557994316955921,points_per_branch=801)
    assert p["dF_dx_J"] > 0 and p["xdot_per_s"] < 0
