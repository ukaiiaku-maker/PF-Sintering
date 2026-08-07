import numpy as np

from pf_sintering import ModelConfig, SinteringModel
from pf_sintering.model import build_params, initialize_fields, laplacian_9pt
from pf_sintering.runner import hard_eta_consistency, selective_reproject_eta_to_f


def test_dev_preset_is_small_and_configurable():
    p=build_params(ModelConfig(preset="dev",nx=72,ny=80,dx=5e-9,r2=70e-9,t_total=1e-6))
    assert (p.Nx,p.Ny)==(72,80)
    assert np.isclose(p.R2,70e-9)


def test_initial_field_shape_and_finite():
    p=build_params(ModelConfig(preset="dev",nx=72,ny=80,t_total=1e-6))
    f,e1,e2,e3=initialize_fields(p)
    assert f.shape==(80,72)
    assert np.isfinite(f).all()
    assert np.isfinite(e1).all() and np.isfinite(e2).all()


def test_laplacian_constant_is_zero():
    a=np.ones((20,30))
    assert np.max(np.abs(laplacian_9pt(a,2e-9)))<1e-6


def test_selective_and_hard_projection_are_distinct():
    p=build_params(ModelConfig(preset="dev",nx=8,ny=8,t_total=1e-6))
    f=np.full((8,8),0.5)
    e1=np.full((8,8),0.4)
    e2=np.full((8,8),0.4)
    e3=np.zeros((8,8))

    s1,s2,s3=selective_reproject_eta_to_f(f,e1,e2,e3,p)
    assert np.allclose(s1+s2+s3,0.8)

    h1,h2,h3=hard_eta_consistency(f,e1,e2,e3,p)
    assert np.allclose(h1+h2+h3,0.5)


def test_tiny_run_completes(tmp_path):
    cfg=ModelConfig(preset="dev",nx=72,ny=80,t_total=2e-7,dt_override=1e-7,save_interval_steps=10,diag_every_steps=1,checkpoint_interval_steps=10,output_dir=tmp_path,out_tag="tiny",status_prints=False,event_prints=False,checkpoint=False,use_aniso_surface=False,psi_measure=False)
    m=SinteringModel(cfg)
    out=m.run()
    assert out["final_step"]==2
    assert np.isfinite(out["Vsolid_ratio"])
    assert "V2_ratio" in out
    assert "quota_progress" in out
