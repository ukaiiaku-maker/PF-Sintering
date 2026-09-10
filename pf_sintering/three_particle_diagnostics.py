"""Current-field two-contact diagnostics, with explicit curvature convention.

Outward convex curvature is positive: kappa=-r''/(1+r'^2)^1.5
 +1/(r*sqrt(1+r'^2)); a sphere has kappa=2/R.
Cannon-Carter uses this total curvature and the measured dihedral angle.
The existing PF local geometric stress is reported separately, not replaced.
"""
import numpy as np
from .three_particle_geometry import grain_volumes,topology_status
from .pr_stress_metrology import local_3d_contact_stress


def radius_profile(f,g):
    r=g['r_c'];out=np.full(len(g['z']),np.nan)
    for j,row in enumerate(f):
        edges=np.flatnonzero((row[:-1]>=.5)&(row[1:]<.5))
        if edges.size:
            i=edges[-1];out[j]=r[i]+(row[i]-.5)/(row[i]-row[i+1])*(r[i+1]-r[i])
    return out


def cannon_carter(radius,psi,kappa,gamma_s):
    if not np.isfinite(radius) or radius<=0: raise ValueError('neck radius must be positive')
    stress=2*gamma_s*np.sin(psi/2)/radius-gamma_s*kappa
    return dict(force_N=float(np.pi*radius**2*stress),stress_Pa=float(stress))


def diagnostics(f,op):
    g=op.g;W=op.W;z=g['z'];R=radius_profile(f,g);mu=op.potential(f).copy();vol=grain_volumes(f,g)
    out={'total_volume_m3':float(vol.sum()),'mirror_error':float(np.max(abs(f-f[::-1]))),'energy_J':op.energy(f),'topology_stop':topology_status(f,g)['stop']}
    localization=12/W*f*f*(1-f)**2*g['r_c'][None,:]
    for k,name in enumerate(['left','center','right']):
        weights=localization*g['ownership'][k]
        out[f'V_{name}_m3']=float(vol[k]);out[f'R_{name}_m']=float((3*vol[k]/(4*np.pi))**(1/3))
        out[f'mu_{name}_Pa']=float(np.sum(mu*weights)/np.sum(weights))
        out[f'centroid_{name}_m']=float(np.sum(f*g['ownership'][k]*g['r_c'][None,:]*z[:,None])/np.sum(f*g['ownership'][k]*g['r_c'][None,:]))
    kappa=np.full_like(R,np.nan)
    # Per-grain curvature; never differentiate through a physical GB groove.
    bounds=[-np.inf,*g['gb'],np.inf]
    for a,b in zip(bounds[:-1],bounds[1:]):
        ids=np.flatnonzero(np.isfinite(R)&(z>a)&(z<b))
        if len(ids)<5:continue
        slope=np.gradient(R[ids],z[ids]);second=np.gradient(slope,z[ids])
        kappa[ids]=-second/(1+slope*slope)**1.5+1/(R[ids]*np.sqrt(1+slope*slope))
    for contact,b in zip(['LEFT','RIGHT'],g['gb']):
        rb=float(np.interp(b,z[np.isfinite(R)],R[np.isfinite(R)]))
        sides=[]
        for sign in [-1,1]:
            distance=sign*(z-b)
            ids=np.flatnonzero(np.isfinite(R)&(distance>=.5*W)&(distance<=2*W))
            if len(ids)<5:raise RuntimeError('contact fit under-resolved')
            coeff=np.polyfit((z[ids]-b)/W,R[ids]/W,3)
            slope=float(np.polyval(np.polyder(coeff),0));second=float(np.polyval(np.polyder(coeff,2),0)/W)
            km=-second/(1+slope*slope)**1.5
            sides.append((slope,km,1/(rb*np.sqrt(1+slope*slope))))
        # Interior angle between the two outward branch tangents, in [0, pi].
        a,beta=sides[0][0],sides[1][0]
        psi=np.arccos(np.clip(-(1+a*beta)/np.sqrt((1+a*a)*(1+beta*beta)),-1,1))
        kt=float(np.mean([s[1]+s[2] for s in sides]));cc=cannon_carter(rb,psi,kt,op.physics.gamma_s)
        pf=local_3d_contact_stress(sides[0][1],sides[1][1],rb,psi,op.physics.gamma_s)
        out.update({f'{contact}_{key}':value for key,value in dict(gb_z_m=float(b),tj_z_m=float(b),tj_r_m=rb,neck_r_m=rb,psi_deg=float(np.rad2deg(psi)),kappa_per_m=kt,CC_force_N=cc['force_N'],CC_stress_Pa=cc['stress_Pa'],PF_geometric_stress_Pa=pf['sigma_3D_local_Pa']).items()})
    return out,dict(z_m=z,r_m=R,kappa_per_m=kappa,mu_field_Pa=mu)


def curvature_watch(f,op):
    """Reuse production branch metrics on four disjoint branches at both TJs.

    Historical 3W/10W distances remain watch locations, not active transfer
    masks in Phase A. The legacy signed total curvature is not used here;
    its absolute meridional/gradient metrics are orientation independent.
    """
    import importlib.util
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'scripts/monitor_current_state_transfer_curvature.py'
    spec=importlib.util.spec_from_file_location('_three_particle_curvature_watch',source)
    monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    g=op.g;z=g['z'];R=radius_profile(f,g);records={}
    for contact,b in zip(['LEFT','RIGHT'],g['gb']):
        rb=float(np.interp(b,z[np.isfinite(R)],R[np.isfinite(R)]))
        for sign in [-1,1]:
            valid=np.isfinite(R)&(sign*(z-b)>=0)
            if contact=='LEFT' and sign>0:valid&=z<=0
            if contact=='RIGHT' and sign<0:valid&=z>=0
            key=f'{contact}_{sign:+d}'
            profile=monitor.branch_profile(z[valid],R[valid],z_tj=b,r_tj=rb,W=op.W,side=key)
            records[key]=monitor.extrema_record(profile,op.W)
    return records
