import numpy as np
import pytest
from scipy.integrate import simpson
from pf_sintering.three_particle_cmc import outer_cap,solve_center,compatible_chain,chemical_potential_null,map_to_pf

def test_outer_cap_volume_and_centroid():
    c=outer_cap();z=np.linspace(0,c['pole_distance_m'],10001)
    r2=c['sphere_radius_m']**2-(z-c['sphere_center_distance_m'])**2
    v=simpson(np.pi*r2,x=z)
    np.testing.assert_allclose(v,4*np.pi*(100e-9)**3/3,rtol=1e-12)
    np.testing.assert_allclose(simpson(z*np.pi*r2,x=z)/v,c['centroid_distance_m'],rtol=1e-12)

def test_cmc_volume_curvature_and_shared_angle():
    c,o=compatible_chain(.7);x=np.linspace(0,1,2001);y=c.solution.sol(x);d=c.solution.sol(x,1)
    r=y[0]*c.radius_scale;u=y[1];rpp=d[1]/c.half_length
    k=1/(r*np.sqrt(1+u*u))-rpp/(1+u*u)**1.5
    np.testing.assert_allclose(k,c.K,rtol=1e-7)
    v=2*np.pi*simpson(r*r,x=x*c.half_length)
    np.testing.assert_allclose(v,4*np.pi*(70e-9)**3/3,rtol=1e-9)
    np.testing.assert_allclose(c.contact_radius,o['contact_radius_m'],rtol=1e-10)
    psi=np.pi-(np.arctan(o['slope'])-np.arctan(u[-1]))
    np.testing.assert_allclose(np.rad2deg(psi),160,atol=1e-9)
    assert c.K>o['K_per_m']

def test_exact_null_and_equal_radius_comparison():
    c,o=chemical_potential_null()
    np.testing.assert_allclose(c.K,o['K_per_m'],rtol=2e-9)
    assert .7<c.ratio<.8
    equal,o=compatible_chain(1)
    assert equal.K<o['K_per_m']

def test_arbitrary_spacing_not_automatically_connected():
    c=solve_center(.7,64.15605972938175e-9);o=outer_cap()
    assert abs(c.contact_radius/o['contact_radius_m']-1)>.1
    good,o=compatible_chain(.7)
    with pytest.raises(ValueError,match='under-resolved'):map_to_pf(good,o,10e-9,1.25e-9)
