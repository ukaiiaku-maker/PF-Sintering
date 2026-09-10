"""Axisymmetric CMC construction with explicit contact compatibility checks.

K is the SUM of principal curvatures (a sphere: K=2/a). Lengths in the
BVP are normalized by the outer volume-equivalent radius. For a fixed
planar GB, isotropic Young balance and prescribed dihedral psi imply the
two one-sided slopes +/-tan((pi-psi)/2). Independently solved segments are
NOT a connected geometry unless their contact radii coincide.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import solve_bvp

@dataclass
class CenterCMC:
    radius_scale: float
    ratio: float
    half_length: float
    K: float
    solution: object
    contact_radius: float
    max_residual: float
    volume_relative_error: float
    min_radius: float

    def radius(self,z):
        return self.radius_scale*self.solution.sol(np.abs(np.asarray(z))/self.half_length)[0]


def outer_cap(outer_radius=100e-9,psi_deg=160.):
    """Regular one-pole CMC cap; volume is 4*pi*R_equivalent^3/3."""
    if not 0<psi_deg<180 or outer_radius<=0:raise ValueError('invalid cap parameters')
    alpha=math.radians((180-psi_deg)/2);s=math.sin(alpha)
    a=outer_radius*((4/3)/(2/3+s-s**3/3))**(1/3)
    h=a*s;H=a+h
    volume=math.pi*(2*a**3/3+a*a*h-h**3/3)
    # Moment about the GB: integral_0^H z*pi*(a^2-(z-h)^2) dz.
    moment=math.pi*((a*a-h*h)*H*H/2+2*h*H**3/3-H**4/4)
    return dict(sphere_radius_m=a,contact_radius_m=a*math.cos(alpha),pole_distance_m=H,
                sphere_center_distance_m=h,centroid_distance_m=moment/volume,
                K_per_m=2/a,volume_m3=volume,slope=math.tan(alpha),psi_deg=psi_deg)


def solve_center(ratio,half_length,outer_radius=100e-9,psi_deg=160.,contact_radius=None):
    """Fixed volume plus angle; solve L too only when imposing a contact radius.

    With prescribed L: 3 states (r,r',half-volume), one parameter K.
    With prescribed contact radius: L is a second parameter. The latter
    finds the compatible L, rather than overconstraining four boundary data
    plus volume at arbitrary L.
    """
    if not all(np.isfinite(v) and v>0 for v in [ratio,half_length,outer_radius]) or not 0<psi_deg<180:
        raise ValueError('positive finite dimensions and valid angle required')
    if contact_radius is not None and (not np.isfinite(contact_radius) or contact_radius<=0):
        raise ValueError('positive finite contact radius required')
    scale=outer_radius;ell=half_length/scale
    target=2*np.pi*ratio**3/3;slope=-math.tan(math.radians((180-psi_deg)/2))
    fixed_contact=contact_radius is not None
    x=np.linspace(0,1,81)
    mean=math.sqrt(target/(np.pi*ell))
    r0=mean-slope*ell/6
    y=np.array([r0+.5*slope*ell*x*x,slope*x,target*x])
    p=[1/max(mean,1e-3)-slope/ell]
    if fixed_contact:p.append(math.log(ell))
    def ode(x,y,p):
        L=math.exp(float(np.clip(p[1],-15,15))) if fixed_contact else ell
        r=np.maximum(y[0],1e-8);u=np.clip(y[1],-1e3,1e3)
        return np.array([L*u,L*((1+u*u)/r-p[0]*(1+u*u)**1.5),L*np.pi*r*r])
    def boundary(a,b,p):
        residual=[a[1],a[2],b[1]-slope,b[2]-target]
        if fixed_contact:residual.append(b[0]-contact_radius/scale)
        return np.array(residual)
    sol=solve_bvp(ode,boundary,x,y,p=p,tol=2e-8,max_nodes=4000)
    if not sol.success:raise ValueError('CMC BVP failed: '+sol.message)
    if fixed_contact and abs(sol.p[1])>=15:raise ValueError('BVP left its regular length range')
    L=scale*math.exp(sol.p[1]) if fixed_contact else half_length
    xx=np.linspace(0,1,2001); yy=sol.sol(xx)
    if np.min(yy[0])<=0 or np.max(abs(yy[1]))>100:
        raise ValueError('non-regular center graph')
    residual=float(np.max(sol.rms_residuals))
    return CenterCMC(scale,ratio,L,float(sol.p[0]/scale),sol,float(yy[0,-1]*scale),
                     residual,float(yy[2,-1]/target-1),float(np.min(yy[0])*scale))


def compatible_chain(ratio,outer_radius=100e-9,psi_deg=160.):
    cap=outer_cap(outer_radius,psi_deg)
    guess=(4*np.pi*(ratio*outer_radius)**3/3)/(2*np.pi*cap['contact_radius_m']**2)
    center=solve_center(ratio,guess,outer_radius,psi_deg,contact_radius=cap['contact_radius_m'])
    return center,cap


def describe(center,cap,width=10e-9,gamma=1.,omega=1e-29):
    difference=center.K-cap['K_per_m']
    mismatch=center.contact_radius/cap['contact_radius_m']-1
    return dict(ratio=center.ratio,center_half_length_nm=center.half_length*1e9,
        Kc_per_m=center.K,Ko_per_m=cap['K_per_m'],delta_mu_Pa=gamma*difference,
        delta_mu_J_per_atom=gamma*omega*difference,center_contact_radius_nm=center.contact_radius*1e9,
        outer_contact_radius_nm=cap['contact_radius_m']*1e9,contact_relative_mismatch=mismatch,
        connected=abs(mismatch)<1e-6,center_min_radius_nm=center.min_radius*1e9,
        center_span_over_W=2*center.half_length/width,contact_separation_margin_W=2*center.half_length/width-8,
        resolution_pass=2*center.half_length>=8*width and center.min_radius>=3*width,
        desired_ordering=difference>1e-6*max(abs(center.K),abs(cap['K_per_m'])),
        matched_potential=abs(difference)<=1e-6*max(abs(center.K),abs(cap['K_per_m'])),max_bvp_residual=center.max_residual,
        volume_relative_error=center.volume_relative_error,
        outer_centroid_from_GB_nm=cap['centroid_distance_m']*1e9)


def chemical_potential_null(outer_radius=100e-9,psi_deg=160.):
    """Exact spherical-zone null: center and outer cap have identical K=2/a."""
    cap=outer_cap(outer_radius,psi_deg);a=cap['sphere_radius_m']
    L=cap['sphere_center_distance_m']
    volume=2*np.pi*(a*a*L-L**3/3)
    ratio=(volume/cap['volume_m3'])**(1/3)
    center=solve_center(ratio,L,outer_radius,psi_deg,contact_radius=cap['contact_radius_m'])
    return center,cap


def chain_radius(z,center,cap):
    z=np.abs(np.asarray(z,dtype=float));L=center.half_length
    r=center.radius(np.minimum(z,L))
    outer=z>L
    x=z[outer]-L-cap['sphere_center_distance_m']
    r[outer]=np.sqrt(np.maximum(0,cap['sphere_radius_m']**2-x*x))
    return r


def map_to_pf(center,cap,width,spacing,margin_widths=10):
    """Map a connected CMC contour by signed distance; no field mass correction.

    Production coefficients give 2*sqrt(k_f/W_f)=W, so tanh(d/W) is the
    equilibrium planar profile in this code's width convention. A sqrt(2)
    factor would change its physical interface width and is not inserted.
    """
    from scipy.spatial import cKDTree
    from .three_particle_geometry import ThreeParticleConfig
    from .axisym import r_centers_faces
    from .gb_obstacle_energy import obstacle_profile
    if width<=0 or spacing<=0 or spacing>width/4:raise ValueError('under-resolved interface')
    if 2*center.half_length<8*width:raise ValueError('CMC center GBs are under-resolved at this W')
    if abs(center.contact_radius/cap['contact_radius_m']-1)>1e-6:raise ValueError('disconnected contact radii')
    L=center.half_length;tip=L+cap['pole_distance_m'];a=cap['sphere_radius_m']
    # Preserve analytical GB positions exactly as FV faces. Do not move a
    # contact to its nearest grid row: the matched null is sensitive to that.
    dz=L/int(np.ceil(L/spacing))
    nz=2*int(np.ceil((tip+margin_widths*width)/dz))
    nr=int(np.ceil((max(a,float(center.radius([0])[0]))+margin_widths*width)/spacing))
    z=(np.arange(nz)-(nz-1)/2)*dz;r,rf=r_centers_faces(nr,spacing)
    dense=np.linspace(-tip,tip,int(np.ceil(2*tip/(spacing/12)))+1)
    angles=np.linspace(0,np.arccos(-cap['sphere_center_distance_m']/a),5001)
    pole_z=L+cap['sphere_center_distance_m']+a*np.cos(angles)
    zz=np.unique(np.r_[dense,pole_z,-pole_z,-L,L,0.])
    curve=np.column_stack([zz,chain_radius(zz,center,cap)])
    starts=curve[:-1];vectors=np.diff(curve,axis=0);tree=cKDTree(starts+vectors/2)
    distance=np.empty((nz,nr))
    for j in range(0,nz,64):
        zpart=z[j:j+64];pts=np.column_stack([np.repeat(zpart,nr),np.tile(r,len(zpart))])
        _,idx=tree.query(pts,k=6,workers=1)
        aa=starts[idx];v=vectors[idx]
        tt=np.clip(np.sum((pts[:,None,:]-aa)*v,axis=2)/np.maximum(np.sum(v*v,axis=2),1e-300),0,1)
        dd=np.sqrt(np.min(np.sum((pts[:,None,:]-aa-tt[:,:,None]*v)**2,axis=2),axis=1))
        distance[j:j+len(zpart)]=dd.reshape(len(zpart),nr)
    inside=(r[None,:]<chain_radius(z,center,cap)[:,None])&(abs(z[:,None])<tip)
    f=.5*(1+np.tanh(np.where(inside,distance,-distance)/width))
    left=obstacle_profile(z,width/np.pi,-L);right=obstacle_profile(z,width/np.pi,L)
    phi=np.stack([1-left,left-right,right])[:,:,None]*np.ones((1,1,nr))
    c=ThreeParticleConfig(outer_radius=center.radius_scale,center_ratio=center.ratio,width=width,spacing=spacing,psi_deg=cap['psi_deg'],margin_widths=margin_widths)
    centroid=L+cap['centroid_distance_m']
    return dict(config=c,f=f,ownership=phi,eta=phi*f[None,:,:],z=z,r_c=r,r_f=rf,dr=spacing,dz=dz,
                gb=np.array([-L,L]),neck=cap['contact_radius_m'],centers=np.array([-centroid,0.,centroid]),
                radii=center.radius_scale*np.array([1,center.ratio,1]),cmc_center=center,cmc_cap=cap)
