"""Fixed-GB three-grain Phase A: constrained initialization and native PF flux.

The multiphase energy extends the frozen binary energy pairwise:
 f*[Wc sum(i<j) phi_i phi_j + k_eta/2 sum_i |grad phi_i|^2].
With only two nonzero ownership fractions it is exactly the production energy.
Ownership is fixed in Phase A. There is no receiver/deposition operator.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit,prange
from .axisym import axisym_laplacian,axisym_volume,axisym_free_energy
from .axisym_numba_kernel import flux_kernel,div_and_update_kernel
from .corrected_interfacial_energy import _axisym_gate_derivative_of_weighted_gradient
from .gb_obstacle_energy import gb_obstacle_coefficients

@dataclass(frozen=True)
class FrozenPhysics:
    gamma_s: float = 1.0
    gamma_gb: float = 0.34729635533386083
    M_s: float = 6e-34  # actual passive build_case(), NOT active-event mobility
    seconds_per_model_time: float = 0.01557994316955921
    temperature_K: float = 1830.15
    atomic_volume_m3: float = 1e-29

@njit(cache=True,parallel=True)
def chemical_potential(f,gb_density,Wf,kf,dr,dz,rc,rf,out):
    nz,nr=f.shape
    for j in prange(nz):
        for i in range(nr):
            v=f[j,i]
            gm=0. if i==0 else (v-f[j,i-1])/dr
            gp=0. if i==nr-1 else (f[j,i+1]-v)/dr
            zm=0. if j==0 else (v-f[j-1,i])/dz
            zp=0. if j==nz-1 else (f[j+1,i]-v)/dz
            lap=(rf[i+1]*gp-rf[i]*gm)/(rc[i]*dr)+(zp-zm)/dz
            out[j,i]=Wf*v*(1-v)*(1-2*v)-kf*lap+gb_density[j,i]
    return out

class PhaseAOperator:
    def __init__(self,g,physics=FrozenPhysics()):
        self.g=g;self.physics=physics;self.W=g['config'].width
        self.W_f=12*physics.gamma_s/self.W;self.k_f=3*physics.gamma_s*self.W
        coef=gb_obstacle_coefficients(physics.gamma_gb,self.W)
        self.k_eta=coef['k_eta'];self.Wc=coef['Wc']
        phi=g['ownership'];self.gb_density=self.Wc*sum(phi[i]*phi[j] for i in range(3) for j in range(i+1,3))
        for p in phi:
            self.gb_density+=_axisym_gate_derivative_of_weighted_gradient(p,self.k_eta/2,g['dr'],g['dz'],g['r_c'],g['r_f'])
        self.mu=np.empty_like(g['f']);self.Jr=np.zeros((len(g['z']),len(g['r_c'])+1));self.Jz=np.zeros_like(self.mu);self.out=np.empty_like(self.mu)
    def potential(self,f):
        g=self.g
        return chemical_potential(f,self.gb_density,self.W_f,self.k_f,g['dr'],g['dz'],g['r_c'],g['r_f'],self.mu)
    def energy(self,f):
        g=self.g
        return axisym_free_energy(f,self,g['dr'],g['dz'],g['r_c'],g['r_f'],bc_z='noflux')+axisym_volume(f*self.gb_density,g['r_c'],g['dr'],g['dz'])
    def step(self,f,dt,surface_enabled=True):
        if dt<0 or not np.isfinite(dt): raise ValueError('dt must be finite and nonnegative')
        if not surface_enabled or dt==0: return f.copy()
        g=self.g
        flux_kernel(f,self.potential(f),g['dr'],g['dz'],self.W,self.physics.M_s,1e-6/self.W,self.Jr,self.Jz)
        div_and_update_kernel(f,self.Jr,self.Jz,g['r_c'],g['r_f'],g['dr'],g['dz'],dt,self.out)
        if not np.all(np.isfinite(self.out)) or self.out.min() < -1e-8 or self.out.max()>1+1e-8:
            raise FloatingPointError('PF step violated bounds: reduce numerical timestep; no clipping permitted')
        return self.out.copy()

@njit(cache=True)
def _constrained_direction(f,mu,phi,z,rc,scale,Wf):
    """Localized gradient projected onto six exact volume/first-moment constraints."""
    nz,nr=f.shape; gram=np.zeros((6,6));rhs=np.zeros(6);basis=np.empty(6)
    for j in range(nz):
        for i in range(nr):
            for k in range(3):
                basis[k]=phi[k,j,i];basis[k+3]=basis[k]*z[j]/scale
            m=max(f[j,i]*(1-f[j,i]),0.)
            w=m*rc[i]/scale
            for k in range(6):
                rhs[k]+=w*basis[k]*mu[j,i]/Wf
                for l in range(6):gram[k,l]+=w*basis[k]*basis[l]
    lam=np.linalg.solve(gram,rhs);out=np.empty_like(f)
    for j in range(nz):
        for i in range(nr):
            shift=0.
            for k in range(3):shift+=phi[k,j,i]*(lam[k]+lam[k+3]*z[j]/scale)
            out[j,i]=-max(f[j,i]*(1-f[j,i]),0.)*(mu[j,i]/Wf-shift)
    return out


def initialize_step(f,op,dtau=0.01):
    """Initialization-only energy descent, not physical time or a ripening model.

    The projected gradient preserves each integral phi_i*f and z*phi_i*f.
    Fixed ownership preserves GB planes. An energy-decreasing bounded line
    search prevents clipping from silently violating the constraints.
    """
    g=op.g
    direction=_constrained_direction(f,op.potential(f),g['ownership'],g['z'],g['r_c'],g['config'].outer_radius,op.W_f)
    energy=op.energy(f); step=dtau
    for _ in range(30):
        trial=f+step*direction
        if trial.min()>=0 and trial.max()<=1:
            next_energy=op.energy(trial)
            if next_energy<=energy+abs(energy)*1e-14:
                return trial,dict(dtau=step,energy_J=next_energy,max_change=float(np.max(np.abs(trial-f))),projected_mu_residual=float(np.max(np.abs(direction))))
        step*=.5
    raise RuntimeError('initialization line search failed; no canonical state written')


def dilute_material_fraction(f,g):
    """Material below f=0.1; monitors unintended diffuse-vapor storage in initialization."""
    total=axisym_volume(f,g['r_c'],g['dr'],g['dz'])
    return axisym_volume(np.where(f<.1,f,0.),g['r_c'],g['dr'],g['dz'])/total


def initialization_locality_gate(f,seed,g):
    """Conservative rejection guard, not a claim of physical vapor solubility.

    Local surface smoothing must not grow a dilute material reservoir. The
    factor-two threshold is an explicit numerical screening tolerance; passing
    it alone does not qualify an initial morphology.
    """
    initial=dilute_material_fraction(seed,g);current=dilute_material_fraction(f,g)
    return dict(pass_locality=current<=2*initial,initial_dilute_fraction=initial,
                current_dilute_fraction=current,maximum_dilute_fraction=2*initial)
