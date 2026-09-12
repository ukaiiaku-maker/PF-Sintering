"""Buffered native PF blocks; all coefficients, stencils and guards retained.

Private relaxation buffers eliminate allocations between native steps. Caller
owns the supplied f/phi arrays; this routine may reuse them as output buffers.
"""
import numpy as np
from numba import njit,prange
from .three_particle_phase_a import chemical_potential
from .axisym_numba_kernel import flux_kernel,div_and_update_kernel
from .three_particle_bounded_mobility import _rescale_faces

@njit(cache=True,parallel=True)
def _pair_phi_kernel_into(phi,f,a,b,dt,M_eta,Wc,k_eta,dr,dz,rc,rf,out):
    nz,nr=f.shape
    for j in prange(nz):
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


@njit(cache=True,parallel=True)
def _gb_density_into(phi,Wc,k_eta,dr,dz,rc,rf,out):
    nz,nr=phi.shape[1:]
    for j in prange(nz):
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


@njit(cache=True)
def native_block(f,phi,gb_density,a,b,dt,M_eta,Wc,k_eta,Wf,kf,dr,dz,rc,rf,W,Ms,steps=10):
    fn=np.empty_like(f);pn=phi.copy();gb=gb_density.copy()
    mu=np.empty_like(f);jr=np.zeros((f.shape[0],f.shape[1]+1));jz=np.zeros_like(f)
    for _ in range(steps):
        chemical_potential(f,gb,Wf,kf,dr,dz,rc,rf,mu)
        flux_kernel(f,mu,dr,dz,W,Ms,1e-6/W,jr,jz)
        _rescale_faces(f,jr,jz)
        div_and_update_kernel(f,jr,jz,rc,rf,dr,dz,dt,fn)
        if not np.isfinite(fn).all() or fn.min() < -1e-8 or fn.max()>1+1e-8:
            raise RuntimeError('fast field bounds')
        _pair_phi_kernel_into(phi,fn,a,b,dt,M_eta,Wc,k_eta,dr,dz,rc,rf,pn)
        _gb_density_into(pn,Wc,k_eta,dr,dz,rc,rf,gb)
        f,fn=fn,f;phi,pn=pn,phi
    return f,phi,gb


@njit(cache=True)
def native_block_reuse(f,phi,gb_density,a,b,dt,M_eta,Wc,k_eta,Wf,kf,
                       dr,dz,rc,rf,W,Ms,fn,pn,gbn,mu,jr,jz,steps=10):
    """Same native block with caller-owned scratch arrays.

    Scratch initialization reproduces the allocating implementation at every
    diagnostic-block boundary.  For an odd step count, the returned state is
    held in the supplied scratch arrays; callers that retain those arrays must
    rotate the prior input arrays into the next scratch set.
    """
    pn[:]=phi
    jr[:]=0.
    jz[:]=0.
    for _ in range(steps):
        chemical_potential(f,gb_density,Wf,kf,dr,dz,rc,rf,mu)
        flux_kernel(f,mu,dr,dz,W,Ms,1e-6/W,jr,jz)
        _rescale_faces(f,jr,jz)
        div_and_update_kernel(f,jr,jz,rc,rf,dr,dz,dt,fn)
        if not np.isfinite(fn).all() or fn.min() < -1e-8 or fn.max()>1+1e-8:
            raise RuntimeError('fast field bounds')
        _pair_phi_kernel_into(phi,fn,a,b,dt,M_eta,Wc,k_eta,dr,dz,rc,rf,pn)
        _gb_density_into(pn,Wc,k_eta,dr,dz,rc,rf,gbn)
        f,fn=fn,f;phi,pn=pn,phi;gb_density,gbn=gbn,gb_density
    return f,phi,gb_density
