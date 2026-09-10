"""Current-field, conservative linearly implicit native surface diffusion.

Mobility and its tangential projector are evaluated on the current field.
The chemical potential is linearized using its exact fixed-ownership Hessian.
A step solves (I-h A(f) H(f)) d = h A(f) mu(f). The returned field is
updated through the native face flux with mu(f)+H(f)d, so mass telescopes
independently of linear-solve error. No morphology or volume target is used.
Step doubling in the driver controls the omitted nonlinear terms.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spilu, splu, gmres, LinearOperator
from .axisym_numba_kernel import flux_kernel, div_and_update_kernel


class ImplicitSurfaceDiffusion:
    def __init__(self, op, *, reuse_small_step_preconditioner=False):
        self.op = op
        self.reuse_small_step_preconditioner = bool(reuse_small_step_preconditioner)
        self._preconditioner = None
        self._preconditioner_h = 0.
        self._preconditioner_field = None
        g = op.g
        nz, nr = g['f'].shape
        self.shape = (nz, nr)
        self.n = nz * nr
        dr, dz = g['dr'], g['dz']
        ids = np.arange(self.n).reshape(self.shape)
        # Every face has one orientation and one common flux for both cells.
        left, right = ids[:, :-1].ravel(), ids[:, 1:].ravel()
        low, high = ids[:-1].ravel(), ids[1:].ravel()
        def matrix(rows, cols, data, shape):
            return sparse.coo_matrix((np.broadcast_to(data, np.shape(rows)).ravel(),
                (np.asarray(rows).ravel(), np.asarray(cols).ravel())), shape=shape).tocsr()
        kr = np.arange(len(left)); kz = np.arange(len(low))
        self.Gr = matrix([kr, kr], [left, right], [[-1/dr], [1/dr]], (len(left), self.n))
        self.Gz = matrix([kz, kz], [low, high], [[-1/dz], [1/dz]], (len(low), self.n))
        self.Gzr = matrix([kr]*4,
            [np.roll(ids,-1,axis=0)[:,:-1].ravel(), np.roll(ids,-1,axis=0)[:,1:].ravel(),
             np.roll(ids,1,axis=0)[:,:-1].ravel(), np.roll(ids,1,axis=0)[:,1:].ravel()],
            [[1/(4*dz)],[1/(4*dz)],[-1/(4*dz)],[-1/(4*dz)]], (len(left), self.n))
        im = np.maximum(np.arange(nr)-1,0); ip = np.minimum(np.arange(nr)+1,nr-1)
        denom = (ip-im)*dr
        self.Grz = matrix([kz]*4,
            [ids[:-1,ip].ravel(), ids[:-1,im].ravel(), ids[1:,ip].ravel(), ids[1:,im].ravel()],
            np.array([np.tile(.5/denom,nz-1), np.tile(-.5/denom,nz-1)]*2), (len(low), self.n))
        rc, rf = g['r_c'], g['r_f']
        self.Dr = matrix([left,right], [kr,kr],
            [np.tile(rf[1:-1]/(rc[:-1]*dr),nz), -np.tile(rf[1:-1]/(rc[1:]*dr),nz)], (self.n,len(left)))
        self.Dz = matrix([low,high], [kz,kz], [[1/dz],[-1/dz]], (self.n,len(low)))
        self.L = (self.Dr@self.Gr + self.Dz@self.Gz).tocsr()
        self.Hgradient = -op.k_f*self.L
        self.identity = sparse.eye(self.n,format='csc')
        self.left,self.right,self.low,self.high=left,right,low,high

    def mobility(self, f):
        x=f.ravel(); op=self.op
        def face(a,b,normal,transverse):
            ff=.5*(x[a]+x[b]); q=op.physics.M_s*(12/op.W)*ff**2*(1-ff)**2
            gn=normal@x; gt=transverse@x
            mag=np.sqrt(gn*gn+gt*gt+(1e-6/op.W)**2)
            nn=gn/mag; nt=gt/mag
            return q*(1-nn*nn), -q*nn*nt
        rr,rz=face(self.left,self.right,self.Gr,self.Gzr)
        zz,zr=face(self.low,self.high,self.Gz,self.Grz)
        B_r=self.Gr.multiply(rr[:,None])+self.Gzr.multiply(rz[:,None])
        B_z=self.Gz.multiply(zz[:,None])+self.Grz.multiply(zr[:,None])
        A=(self.Dr@B_r+self.Dz@B_z).tocsr(); A.eliminate_zeros()
        return A

    def step(self,f,h):
        if not np.isfinite(h) or h<0: raise ValueError('invalid timestep')
        if h==0:return f.copy()
        op=self.op;g=op.g
        A=self.mobility(f)
        H=self.Hgradient+sparse.diags(op.W_f*(1-6*f.ravel()+6*f.ravel()**2))
        mu=op.potential(f).copy()
        lhs=(self.identity-h*(A@H)).tocsc();lhs.eliminate_zeros()
        rhs=h*(A@mu.ravel())
        # Dropping is ONLY in the preconditioner; the solved operator retains
        # every native coefficient, including arbitrarily small mobilities.
        reuse=((h>.05 or self.reuse_small_step_preconditioner) and self._preconditioner is not None
               and .3<h/self._preconditioner_h<3.
               and np.max(np.abs(f-self._preconditioner_field))<.005)
        for attempt in range(2):
            if reuse and attempt==0:
                lu=self._preconditioner
            else:
                pre=lhs.copy(); pre.data[np.abs(pre.data)<1e-7]=0; pre.eliminate_zeros()
                lu=splu(pre) if h>.05 else spilu(pre,drop_tol=1e-3,fill_factor=15)
                if h>.05 or self.reuse_small_step_preconditioner:
                    self._preconditioner=lu;self._preconditioner_h=h
                    self._preconditioner_field=f.copy()
            d,info=gmres(lhs,rhs,M=LinearOperator(lhs.shape,lu.solve),
                         rtol=(1e-9 if h>.05 else 1e-11),atol=0.,restart=40,maxiter=3)
            if not info:break
        if info: raise FloatingPointError('implicit Krylov convergence')
        relative_residual=np.linalg.norm(lhs@d-rhs)/max(np.linalg.norm(rhs),1e-300)
        if relative_residual>1e-8: raise FloatingPointError('implicit linear residual')
        effective_mu=mu+(H@d).reshape(f.shape)
        result=self.conservative_update(f,effective_mu,h)
        if not np.isfinite(result).all() or result.min() < -1e-8 or result.max()>1+1e-8:
            raise FloatingPointError('implicit bounds; reduce timestep without clipping')
        return result

    def conservative_update(self,f,mu,h):
        op=self.op;g=op.g
        flux_kernel(f,mu,g['dr'],g['dz'],op.W,op.physics.M_s,1e-6/op.W,op.Jr,op.Jz)
        div_and_update_kernel(f,op.Jr,op.Jz,g['r_c'],g['r_f'],g['dr'],g['dz'],h,op.out)
        return op.out.copy()
