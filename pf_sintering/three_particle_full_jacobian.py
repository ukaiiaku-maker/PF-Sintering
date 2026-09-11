"""Opt-in full current-field Jacobian of the unchanged harmonic native flux.

Includes mobility/projector derivatives omitted by the frozen-mobility step.
The accepted increment remains a common-face conservative divergence.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu,gmres,LinearOperator
from .three_particle_bounded_mobility import HarmonicSurfaceDiffusion,harmonic,_rescale_faces
from .axisym_numba_kernel import flux_kernel,div_and_update_kernel

class FullJacobianSurfaceDiffusion(HarmonicSurfaceDiffusion):
    def native_fluxes(self,f):
        op=self.op;g=op.g;mu=op.potential(f).copy()
        flux_kernel(f,mu,g['dr'],g['dz'],op.W,op.physics.M_s,1e-6/op.W,op.Jr,op.Jz)
        _rescale_faces(f,op.Jr,op.Jz)
        return mu,op.Jr.copy(),op.Jz.copy()

    def rhs(self,f):
        _,jr,jz=self.native_fluxes(f);op=self.op;g=op.g
        div_and_update_kernel(np.zeros_like(f),jr,jz,g['r_c'],g['r_f'],g['dr'],g['dz'],1.,op.out)
        return op.out.copy()

    def flux_jacobian(self,f,mu):
        x=f.ravel();u=mu.ravel();scale=self.op.physics.M_s*12/self.op.W
        q=scale*x*x*(1-x)**2;qp=2*scale*x*(1-x)*(1-2*x);epsilon2=(1e-6/self.op.W)**2
        H=self.Hgradient+sparse.diags(self.op.W_f*(1-6*x+6*x*x))
        def face(a,b,Gn,Gt):
            gn=Gn@x;gt=Gt@x;s=gn*gn+gt*gt+epsilon2
            pnn=1-gn*gn/s;pnt=-gn*gt/s;Q=harmonic(q[a],q[b]);un=Gn@u;ut=Gt@u
            den=q[a]+q[b];ra=np.divide(q[a],den,out=np.zeros_like(den),where=den>0);rb=np.divide(q[b],den,out=np.zeros_like(den),where=den>0)
            projection=pnn*un+pnt*ut
            da=2*rb*rb*qp[a]*projection;db=2*ra*ra*qp[b]*projection
            cn=Q*((-2*gn*(gt*gt+epsilon2)/s**2)*un+(gt*(gn*gn-gt*gt-epsilon2)/s**2)*ut)
            ct=Q*((2*gn*gn*gt/s**2)*un+(gn*(gt*gt-gn*gn-epsilon2)/s**2)*ut)
            rows=np.arange(len(a));direct=sparse.coo_matrix((np.r_[da,db],(np.r_[rows,rows],np.r_[a,b])),shape=(len(a),self.n)).tocsr()
            B=Gn.multiply((Q*pnn)[:,None])+Gt.multiply((Q*pnt)[:,None])
            C=direct+Gn.multiply(cn[:,None])+Gt.multiply(ct[:,None])
            return (B@H+C).tocsr()
        return face(self.left,self.right,self.Gr,self.Gzr),face(self.low,self.high,self.Gz,self.Grz)

    def step(self,f,h):
        if not np.isfinite(h) or h<0:raise ValueError('invalid timestep')
        if h==0:return f.copy()
        mu,jr,jz=self.native_fluxes(f);Kr,Kz=self.flux_jacobian(f,mu);J=(self.Dr@Kr+self.Dz@Kz).tocsr()
        lhs=(self.identity-h*J).tocsc();lhs.eliminate_zeros()
        rhs=-h*(self.Dr@jr[:,1:-1].ravel()+self.Dz@jz[:-1].ravel())
        pre=lhs.copy();pre.data[abs(pre.data)<1e-7]=0;pre.eliminate_zeros();lu=splu(pre)
        delta,info=gmres(lhs,rhs,M=LinearOperator(lhs.shape,lu.solve),rtol=(1e-9 if h>.05 else 1e-11),atol=0.,restart=40,maxiter=3)
        residual=float(np.linalg.norm(lhs@delta-rhs)/max(np.linalg.norm(rhs),1e-300))
        if info or residual>1e-8:raise FloatingPointError('full-Jacobian linear convergence')
        jr[:,1:-1]-=(Kr@delta).reshape(jr[:,1:-1].shape);jz[:-1]-=(Kz@delta).reshape(jz[:-1].shape)
        op=self.op;g=op.g;div_and_update_kernel(f,jr,jz,g['r_c'],g['r_f'],g['dr'],g['dz'],h,op.out);result=op.out.copy()
        if not np.isfinite(result).all() or result.min() < -1e-8 or result.max()>1+1e-8:raise FloatingPointError('full-Jacobian field guard')
        return result
