"""Pairwise three-grain adapter to the existing current-state one-b event.

The inactive eta is bitwise unchanged by the source operation. Physical fast
surface evolution may change it through f; ownership relaxation acts only on
the selected pair. No extra rigid displacement or stress correction is added.
"""
import numpy as np
from numba import njit
from .current_state_mass_transfer import bounded_conservative_transfer
from .corrected_interfacial_energy import _axisym_gate_derivative_of_weighted_gradient


def pair_transfer(state,receiver_mask,donor_mask,*,pair,**kwargs):
    f=state[0];eta=np.asarray(state[1:]);a,b=pair;inactive=3-a-b
    if not np.any(eta[inactive]):
        binary,diag=bounded_conservative_transfer((f,eta[a],eta[b]),receiver_mask,donor_mask,**kwargs)
        out=eta.copy();out[a]=binary[1];out[b]=binary[2]
        diag.update(inactive_grain=inactive,inactive_eta_bitwise_preserved=True)
        return (binary[0],*out),diag
    u=eta[a]+eta[b]
    active_fraction=np.divide(u,f,out=np.zeros_like(f),where=np.abs(f)>1e-30)
    receiver=receiver_mask*active_fraction;donor=donor_mask*active_fraction
    # Reuse the exact bounded source equation and volume/capacity accounting.
    binary,diag=bounded_conservative_transfer((f,eta[a],f-eta[a]),receiver,donor,**kwargs)
    fn=binary[0];pn=np.divide(eta[a],u,out=np.full_like(f,.5),where=np.abs(u)>1e-30)
    un=fn-eta[inactive]
    if un.min() < -2e-14:raise RuntimeError('pair source exhausted active solid')
    out=eta.copy();out[a]=un*pn;out[b]=un-out[a]
    # On pure inactive cells the active source is identically zero.
    pure=u==0.;out[a][pure]=eta[a][pure];out[b][pure]=eta[b][pure]
    if np.max(abs(out.sum(axis=0)-fn))>5e-15:raise RuntimeError('three-grain source closure')
    diag.update(inactive_grain=inactive,inactive_eta_bitwise_preserved=np.array_equal(out[inactive],eta[inactive]),
                partition_closure=float(np.max(abs(out.sum(axis=0)-fn))))
    return (fn,*out),diag


def ownership_pair_step_reference(phi,f,op,pair,dt,M_eta):
    """Same obstacle gradient flow as binary, restricted to an active pair."""
    a,b=pair;g=op.g;bounded=np.minimum(1.,np.maximum(0.,f));d=phi[a]-phi[b]
    jr=np.zeros_like(op.Jr);jz=np.zeros_like(op.Jz)
    jr[:,1:-1]=.5*(bounded[:,:-1]+bounded[:,1:])*np.diff(d,axis=1)/g['dr']
    jz[:-1]=.5*(bounded[:-1]+bounded[1:])*np.diff(d,axis=0)/g['dz']
    div=(jr[:,1:]*g['r_f'][None,1:]-jr[:,:-1]*g['r_f'][None,:-1])/(g['r_c'][None,:]*g['dr'])
    div+= (jz-np.vstack([np.zeros_like(jz[:1]),jz[:-1]]))/g['dz']
    derivative=bounded*op.Wc*(phi[b]-phi[a])-op.k_eta*div
    total=phi[a]+phi[b];out=phi.copy()
    # The obstacle constraint applies to ownership, not to the conserved f.
    out[a]=np.minimum(total,np.maximum(0.,phi[a]-dt*.5*M_eta*bounded*derivative))
    out[b]=total-out[a]
    return out


@njit(cache=True)
def _pair_phi_kernel(phi,f,a,b,dt,M_eta,Wc,k_eta,dr,dz,rc,rf):
    out=phi.copy();nz,nr=f.shape
    for j in range(nz):
        for i in range(nr):
            v=min(1.,max(0.,f[j,i]));d=phi[a,j,i]-phi[b,j,i]
            jm=0.;jp=0.;zm=0.;zp=0.
            if i>0:jm=.5*(v+min(1.,max(0.,f[j,i-1])))*(d-(phi[a,j,i-1]-phi[b,j,i-1]))/dr
            if i<nr-1:jp=.5*(v+min(1.,max(0.,f[j,i+1])))*((phi[a,j,i+1]-phi[b,j,i+1])-d)/dr
            if j>0:zm=.5*(v+min(1.,max(0.,f[j-1,i])))*(d-(phi[a,j-1,i]-phi[b,j-1,i]))/dz
            if j<nz-1:zp=.5*(v+min(1.,max(0.,f[j+1,i])))*((phi[a,j+1,i]-phi[b,j+1,i])-d)/dz
            div=(jp*rf[i+1]-jm*rf[i])/(rc[i]*dr)+(zp-zm)/dz
            derivative=v*Wc*(phi[b,j,i]-phi[a,j,i])-k_eta*div
            total=phi[a,j,i]+phi[b,j,i]
            out[a,j,i]=min(total,max(0.,phi[a,j,i]-dt*.5*M_eta*v*derivative))
            out[b,j,i]=total-out[a,j,i]
    return out


def ownership_pair_step(phi,f,op,pair,dt,M_eta):
    g=op.g
    return _pair_phi_kernel(phi,f,*pair,dt,M_eta,op.Wc,op.k_eta,g['dr'],g['dz'],g['r_c'],g['r_f'])


@njit(cache=True)
def _gb_density(phi,Wc,k_eta,dr,dz,rc,rf):
    nz,nr=phi.shape[1:];out=np.empty((nz,nr))
    for j in range(nz):
        for i in range(nr):
            value=Wc*((phi[0,j,i]*phi[1,j,i]+phi[0,j,i]*phi[2,j,i])+phi[1,j,i]*phi[2,j,i])
            for k in range(3):
                v=phi[k,j,i];grad=0.
                if i<nr-1:
                    d=phi[k,j,i+1]-v;grad+=(k_eta/2*rf[i+1]*d*d/(2*dr*dr))/rc[i]
                if i>0:
                    d=v-phi[k,j,i-1];grad+=(k_eta/2*rf[i]*d*d/(2*dr*dr))/rc[i]
                if j<nz-1:
                    d=phi[k,j+1,i]-v;grad+=k_eta/2*d*d/(2*dz*dz)
                if j>0:
                    d=v-phi[k,j-1,i];grad+=k_eta/2*d*d/(2*dz*dz)
                value+=grad
            out[j,i]=value
    return out


def update_ownership(op,phi):
    op.g['ownership']=phi
    g=op.g;op.gb_density=_gb_density(phi,op.Wc,op.k_eta,g['dr'],g['dz'],g['r_c'],g['r_f'])


def save_event_checkpoint(path,state,restart,*,contact,label):
    """Atomic accepted-state record; fields and event progress share one file."""
    import json,os
    from pathlib import Path
    path=Path(path);metadata={k:v for k,v in restart.items() if k not in ['base_fields','union_previous_fields']}
    temporary=path.with_suffix('.writing.npz')
    np.savez_compressed(temporary,fields=np.array(state),base_fields=np.array(restart['base_fields']),
                        restart_json=json.dumps(metadata),contact=contact,label=label)
    os.replace(temporary,path)


def load_event_checkpoint(path):
    import json
    with np.load(path,allow_pickle=False) as d:
        state=tuple(x.copy() for x in d['fields']);restart=json.loads(str(d['restart_json']))
        restart['base_fields']=tuple(x.copy() for x in d['base_fields']);restart['union_previous_fields']=tuple(x.copy() for x in state)
        return state,restart,str(d['contact']),str(d['label'])
