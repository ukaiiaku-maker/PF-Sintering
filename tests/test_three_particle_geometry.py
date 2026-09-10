import numpy as np
import pytest
from pf_sintering.three_particle_geometry import (ThreeParticleConfig,build_three_particle,grain_volumes,meridian,geometry_parameters,topology_status)
from pf_sintering.axisym import axisym_volume

@pytest.fixture(scope='module')
def geometry():
    return build_three_particle(ThreeParticleConfig(spacing=2.5e-9))

def test_partition_volume_and_mirror(geometry):
    g=geometry; f=g['f']
    np.testing.assert_allclose(g['eta'].sum(axis=0),f,atol=2e-16)
    np.testing.assert_allclose(f,f[::-1],atol=3e-14)
    v=grain_volumes(f,g)
    assert abs(v[0]/v[2]-1)<1e-13
    assert v[1]<v[0]
    assert abs(v.sum()/axisym_volume(f,g['r_c'],g['dr'],g['dz'])-1)<1e-14
    assert not topology_status(f,g)['stop']

def test_two_contacts_and_dihedral():
    c=ThreeParticleConfig(); g=geometry_parameters(c); h=1e-13
    for b in g['gb']:
        r=meridian(np.array([b-h,b,b+h]),c)
        np.testing.assert_allclose(r[1],g['neck'],rtol=1e-12)
        slopes=(r[[0,2]]-r[1])/h
        np.testing.assert_allclose(slopes,np.tan(np.deg2rad(10)),rtol=2e-5)

def test_controls_and_resolution():
    equal=geometry_parameters(ThreeParticleConfig(center_ratio=1))
    np.testing.assert_array_equal(equal['radii'],np.full(3,100e-9))
    with pytest.raises(ValueError): build_three_particle(ThreeParticleConfig(center_ratio=.1))
    with pytest.raises(ValueError): build_three_particle(ThreeParticleConfig(spacing=5e-9))
