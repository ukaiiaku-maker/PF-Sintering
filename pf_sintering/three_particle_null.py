"""Finite-width null diagnostics; no transfer or stress correction.

Virtual-work multipliers are thermodynamic chemical potentials ONLY if the
constrained equilibrium residual is small and independent of the admissible
variation basis. A zero instantaneous grain integral does not imply F(f)=0.
"""
import numpy as np
from .axisym_numba_kernel import flux_kernel,div_and_update_kernel
from .axisym import axisym_volume


def native_rhs(f,op):
    g=op.g
    flux_kernel(f,op.potential(f),g['dr'],g['dz'],op.W,op.physics.M_s,1e-6/op.W,op.Jr,op.Jz)
    div_and_update_kernel(np.zeros_like(f),op.Jr,op.Jz,g['r_c'],g['r_f'],g['dr'],g['dz'],1.,op.out)
    return op.out.copy()


def virtual_work_multipliers(mu,phi,weight,cell_volume):
    """G_ki=dV_k/db_i and e_i=dE/db_i for b_i=weight*phi_i.

    Solve G^T lambda=e; same phi means symmetric Gram G. This exactly matches
    first variations, but its interpretation as dE_min/dV needs equilibrium.
    """
    gram=np.einsum('kji,lji,ji,ji->kl',phi,phi,weight,cell_volume)
    rhs=np.einsum('kji,ji,ji,ji->k',phi,mu,weight,cell_volume)
    lam=np.linalg.solve(gram,rhs)
    residual=mu-np.einsum('k,kji->ji',lam,phi)
    rms=np.sqrt(np.sum(residual**2*weight*cell_volume)/np.sum(weight*cell_volume))
    return lam,float(rms)


def inspect_state(f,op):
    g=op.g;mu=op.potential(f).copy();phi=g['ownership']
    cv=np.broadcast_to(2*np.pi*g['r_c'][None,:]*g['dr']*g['dz'],f.shape)
    dz,dr=np.gradient(f,g['dz'],g['dr'])
    weights={'profile':f*(1-f),'localization':(f*(1-f))**2,'normal_displacement':op.W*np.hypot(dr,dz)}
    thermo={}
    for name,w in weights.items():
        lam,res=virtual_work_multipliers(mu,phi,np.maximum(w,0),cv)
        thermo[name]=dict(multipliers_Pa=lam.tolist(),delta_Pa=float(lam[1]-.5*(lam[0]+lam[2])),weighted_KKT_rms_Pa=res)
    F=native_rhs(f,op)
    rates=[axisym_volume(F*p,g['r_c'],g['dr'],g['dz']) for p in phi]
    faces=[int(np.argmin(abs(g['z']+g['dz']/2-b))) for b in g['gb']]
    plane_rate=float(np.sum((op.Jz[faces[0]]-op.Jz[faces[1]])*2*np.pi*g['r_c']*g['dr']))
    return dict(virtual_work=thermo,grain_volume_rates_m3_per_model_time=rates,
                sharp_GB_center_rate_m3_per_model_time=plane_rate,
                native_rhs_max_per_model_time=float(np.max(abs(F))),
                virtual_work_delta_spread_Pa=float(np.ptp([x['delta_Pa'] for x in thermo.values()])))
