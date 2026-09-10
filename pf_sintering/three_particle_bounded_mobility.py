"""Experimental consistent face quadrature preserving degenerate-mobility zeros.

Not a barrier, energy or continuum mobility change. The original arithmetic
face-field quadrature remains the default production/native implementation.
"""
import numpy as np
from numba import njit
from .three_particle_implicit import ImplicitSurfaceDiffusion
from .axisym_numba_kernel import flux_kernel,div_and_update_kernel


def harmonic(a,b):
    return np.divide(2*a*b,a+b,out=np.zeros_like(a+b),where=(a+b)>0)


@njit(cache=True)
def _rescale_faces(f,jr,jz):
    nz,nr=f.shape
    for j in range(nz):
        for i in range(nr):
            a=f[j,i];qa=a*a*(1-a)**2
            if i<nr-1:
                b=f[j,i+1];qb=b*b*(1-b)**2;avg=.5*(a+b);original=avg*avg*(1-avg)**2
                factor=(2*qa*qb/(qa+qb)/original) if qa+qb>0 and original>0 else 0.
                jr[j,i+1]*=factor
            if j<nz-1:
                b=f[j+1,i];qb=b*b*(1-b)**2;avg=.5*(a+b);original=avg*avg*(1-avg)**2
                factor=(2*qa*qb/(qa+qb)/original) if qa+qb>0 and original>0 else 0.
                jz[j,i]*=factor


def bounded_mobility_update(f,mu,op,h):
    g=op.g
    flux_kernel(f,mu,g['dr'],g['dz'],op.W,op.physics.M_s,1e-6/op.W,op.Jr,op.Jz)
    _rescale_faces(f,op.Jr,op.Jz)
    div_and_update_kernel(f,op.Jr,op.Jz,g['r_c'],g['r_f'],g['dr'],g['dz'],h,op.out)
    return op.out.copy()


class HarmonicSurfaceDiffusion(ImplicitSurfaceDiffusion):
    def mobility(self,f):
        x=f.ravel();op=self.op;q=op.physics.M_s*(12/op.W)*x*x*(1-x)**2
        def face(a,b,normal,transverse):
            value=harmonic(q[a],q[b]);gn=normal@x;gt=transverse@x
            mag=np.sqrt(gn*gn+gt*gt+(1e-6/op.W)**2);nn=gn/mag;nt=gt/mag
            return value*(1-nn*nn),-value*nn*nt
        rr,rz=face(self.left,self.right,self.Gr,self.Gzr)
        zz,zr=face(self.low,self.high,self.Gz,self.Grz)
        return (self.Dr@(self.Gr.multiply(rr[:,None])+self.Gzr.multiply(rz[:,None]))+
                self.Dz@(self.Gz.multiply(zz[:,None])+self.Grz.multiply(zr[:,None]))).tocsr()
    def conservative_update(self,f,mu,h):
        return bounded_mobility_update(f,mu,self.op,h)
