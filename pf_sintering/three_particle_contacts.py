"""Geometry-indexed reuse of bicrystal production metrology and root law.

The two-side mean below comes ONLY from the production _side_first_stresses.
A second contact terminates a grain's free branch; it is never crossed by a fit.
"""
import math
import numpy as np
from .continuous_field_tj import ContinuousFieldTJTracker
from .pr_experimental_geometry import (_radius_at_half_level, _branch, _local_fit,
    _branch_integrals, _continuous_endpoint_turning)
from .experimental_pr_metrology import _side_first_stresses, _particle_silhouette
from .exp_barrier_nucleation import (CompleteExpFloorParams,
    delta_G_complete_exp_floor_eV, gamma_complete_exp_floor_per_model_time)
from .corrected_pr_thermodynamics import _gb_plane_values, _radial_area_weights_to_cutoff
from .axisym_branch_boundary_flux import extract_axisymmetric_surface_branches


def contact_stresses(f, eta_negative, eta_positive, setup, z_tj, r_tj,
                     lower=-np.inf, upper=np.inf, other_contacts=()):
    """Apply unchanged branch formulas on each adjacent grain's free surface.

    Infinite limits recover the complete original bicrystal branches exactly.
    Finite limits insert the other field TJ as that grain's endpoint.
    """
    z=setup['z'];r=setup['r_c'];radius=_radius_at_half_level(f,r)
    radius[(z<=lower)|(z>=upper)]=np.nan
    geometry=dict(r_n_m=float(r_tj),gamma_J_per_m2=setup['gamma_s'])
    branches={}
    for side,sign in [('negative',-1),('positive',1)]:
        # Prevent original positive-tip closure from reaching a third grain.
        branch=_branch(radius,z,eta_negative,eta_positive,r,z_tj,r_tj,sign,
                       f if sign<0 or not np.isfinite(upper) else None)
        endpoint=lower if sign<0 else upper
        if np.isfinite(endpoint):
            matches=[p for p in other_contacts if abs(p[0]-endpoint)<1e-15]
            if len(matches)!=1:raise ValueError('missing adjacent field TJ endpoint')
            branch['z_m']=np.r_[branch['z_m'],endpoint]
            branch['r_m']=np.r_[branch['r_m'],matches[0][1]]
            branch['grain_label']=np.r_[branch['grain_label'],0]
        fit=_local_fit(branch,z_tj,3*setup['W'])
        integral=_branch_integrals(branch)
        geometry.update({f'theta_{side}_rad':fit['theta_side_equiv_rad'],
            f'kappa1_{side}_per_m':fit['kappa_m_per_m'],
            f'L_f_{side}_m':integral['L_f_m'],
            f'delta_phi_f_{side}_rad':integral['delta_phi_f_rad'],
            f'delta_phi_f_continuous_{side}_rad':_continuous_endpoint_turning(branch,fit['slope_dr_dz'])})
        branches[side]=branch
    result=_side_first_stresses(geometry,_particle_silhouette(branches['positive']))
    # MW/N are particle-shape diagnostics outside this requested two-stress audit.
    result={k:v for k,v in result.items() if 'MW' not in k and '_N' not in k}
    return {**geometry,**result},branches


def root_law(stress, radius, manifest):
    p=manifest['root_barrier_slice']
    barrier=CompleteExpFloorParams(p['G0_eV'],p['Gfloor_eV'],p['a'],p['sigmahat_Pa'],p['n'])
    fit=delta_G_complete_exp_floor_eV(stress,barrier)
    rate=gamma_complete_exp_floor_per_model_time(stress,radius,barrier=barrier,
        clock_scale_per_model_time=manifest['clock_scale'],b_m=manifest['b_event_m'],
        temperature_K=manifest['temperature_K'])
    penalty=manifest['root_formation_penalty_eV']
    rate*=math.exp(-penalty/(1.380649e-23/1.602176634e-19*manifest['temperature_K']))
    rate_s=rate/manifest['seconds_per_model_time']
    return dict(G_fit_eV=fit,G_root_eV=fit+penalty,root_rate_per_s=rate_s,
        root_rate_per_model_time=rate,constant_state_mean_wait_s=manifest['root_threshold_multiplier']/rate_s,
        barrier_reduction_from_zero_eV=p['G0_eV']-fit)


def evaluate_contacts(f,op,manifest):
    g=op.g;setup={**g,'W':op.W,'gamma_s':op.physics.gamma_s};phi=g['ownership']
    mu=op.potential(f).copy();points=[]
    for i,b in enumerate(g['gb']):
        # Restrict tracker to the two adjacent grains; avoid zero-zero third ownership.
        mask=(g['z']<0) if i==0 else (g['z']>0)
        tracker=ContinuousFieldTJTracker(g['z'][mask],g['r_c'],b,6*max(g['dr'],g['dz']))
        p=tracker.locate(f[mask],(f*phi[i])[mask],(f*phi[i+1])[mask])
        points.append((p['z_TJ_m'],p['r_TJ_m']))
    output={}
    for i,name in enumerate(['LEFT','RIGHT']):
        zt,rt=points[i];lower=-np.inf if i==0 else points[0][0];upper=points[1][0] if i==0 else np.inf
        if i==0:
            stress,_=contact_stresses(f,f*phi[i],f*phi[i+1],setup,zt,rt,lower,upper,points)
        else:
            # Both contacts use the live bicrystal's particle-positive frame.
            # At RIGHT the center particle lies toward decreasing global z.
            local_setup={**setup,'z':-g['z'][::-1]}
            stress,_=contact_stresses(f[::-1],(f*phi[2])[::-1],(f*phi[1])[::-1],
                local_setup,-zt,rt,-np.inf,-points[0][0],
                [(-p[0],p[1]) for p in points])
            # Serialize sides in global L/C/R order, while preserving the
            # scalar combination returned by the original production function.
            stress={k.replace('negative','SIDE_TEMP').replace('positive','negative').replace('SIDE_TEMP','positive'):v
                    for k,v in stress.items()}
        weights=_radial_area_weights_to_cutoff(g['r_f'],rt-3*op.W)
        if np.count_nonzero(weights)<3:weights=_radial_area_weights_to_cutoff(g['r_f'],rt-op.W)
        if np.count_nonzero(weights)<2:raise ValueError('GB sampling under-resolved')
        mugb=float(np.sum(weights*_gb_plane_values(mu,g['z'],zt))/np.sum(weights))
        mask=(g['z']>lower)&(g['z']<upper)
        branches=extract_axisymmetric_surface_branches(f[mask],mu[mask],g['r_c'],g['z'][mask],z_tj=zt,r_tj=rt)
        mutj={}
        for branch in branches:
            support=(branch.s_centers_m>=op.W)&(branch.s_centers_m<=3*op.W)
            if np.count_nonzero(support)<2:support=branch.s_centers_m<=3*op.W
            if not np.any(support):raise ValueError('TJ sampling under-resolved')
            mutj[branch.side]=float(np.mean(branch.mu_Pa[support]))
        affinity=mugb-.5*(mutj['negative']+mutj['positive'])
        output[name]={**stress,**root_law(stress['sigma_local_Pa'],rt,manifest),
            'z_TJ_m':zt,'mu_GB_Pa':mugb,'mu_TJ_negative_Pa':mutj['negative'],
            'mu_TJ_positive_Pa':mutj['positive'],'transport_affinity_Pa':affinity,
            'one_b_transport_work_J':affinity*math.pi*rt**2*manifest['b_event_m'],
            'work_is_transport_diagnostic_not_root_barrier':True}
    return output
