"""Finite-neck axisymmetric three-grain seed; never an equilibrated checkpoint.

A quintic on each side of each TJ matches radius, slope and second derivative
at the spherical-cap join. Only the physical dihedral corner is non-C1.
The signed-distance construction uses densely sampled meridian segments.
Ownership has compact obstacle profiles and closes as L+C+R=1, including
vacuum. GB coordinates are explicit and independent of evolving grain volumes.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial import cKDTree
from .axisym import r_centers_faces, axisym_volume
from .gb_obstacle_energy import obstacle_profile


@dataclass(frozen=True)
class ThreeParticleConfig:
    outer_radius: float = 100e-9
    center_ratio: float = 0.70
    neck_ratio: float = 0.40  # relative to smaller nominal cap radius
    width: float = 10e-9
    spacing: float = 1.25e-9
    psi_deg: float = 160.0
    margin_widths: float = 10.0

    def validate(self):
        vals = asdict(self)
        if not all(np.isfinite(v) and v > 0 for v in vals.values()):
            raise ValueError('all geometry parameters must be positive and finite')
        if not 0 < self.neck_ratio < 0.8 or not 0 < self.psi_deg < 180:
            raise ValueError('invalid neck ratio or dihedral angle')
        if self.spacing > self.width / 4:
            raise ValueError('interface needs at least four cells per width')
        if self.outer_radius * min(1, self.center_ratio) < 6*self.width:
            raise ValueError('smallest nominal radius is below six interface widths')
        if self.margin_widths < 6:
            raise ValueError('vacuum margin must be at least six widths')


def geometry_parameters(c):
    c.validate()
    ro, rc = c.outer_radius, c.outer_radius*c.center_ratio
    rb = c.neck_ratio*min(ro,rc)
    half = math.sqrt(rc*rc-rb*rb)
    separation = half + math.sqrt(ro*ro-rb*rb)
    return dict(radii=np.array([ro,rc,ro]), centers=np.array([-separation,0,separation]),
                gb=np.array([-half,half]), neck=rb)


def meridian(z, c):
    """Seed r(z), with explicit C2 cap joins (all calculations scaled by Ro)."""
    g=geometry_parameters(c); scale=c.outer_radius
    x=np.asarray(z)/scale; gb=g['gb']/scale; centers=g['centers']/scale
    radii=g['radii']/scale; rb=g['neck']/scale
    grain=np.searchsorted(gb,x)
    result=np.sqrt(np.maximum(0,radii[grain]**2-(x-centers[grain])**2))
    slope=math.tan(math.radians((180-c.psi_deg)/2))
    for k,b in enumerate(gb):
        for side,owner in [(-1,k),(1,k+1)]:
            span=0.45*radii[owner]
            end=b+side*span
            d=end-centers[owner]
            r=math.sqrt(radii[owner]**2-d*d)
            deriv=-d/r*side
            second=-radii[owner]**2/r**3
            # Same one-sided meridional curvature at each TJ, independently
            # joined to the cap belonging to that side. No flat collar.
            poly=BPoly.from_derivatives([0,span],[[rb,slope,1/min(radii)], [r,deriv,second]])
            distance=(x-b)*side
            mask=(distance>=0)&(distance<=span)
            result[mask]=poly(distance[mask])
    return result*scale


def build_three_particle(c=ThreeParticleConfig()):
    g=geometry_parameters(c); dx=c.spacing
    half=g['centers'][-1]+c.outer_radius+c.margin_widths*c.width
    nz=2*int(np.ceil(half/dx)); nr=int(np.ceil((max(g['radii'])+c.margin_widths*c.width)/dx))
    z=(np.arange(nz)-(nz-1)/2)*dx; r,rf=r_centers_faces(nr,dx)
    tip=g['centers'][-1]+c.outer_radius
    # Use an angular cap sampling as well as uniform z to resolve the tips.
    dense=np.linspace(-tip,tip,int(np.ceil(2*tip/(dx/10)))+1)
    angle=np.linspace(0,np.pi,2001)
    cap=g['centers'][-1]+c.outer_radius*np.cos(angle)
    dense=np.unique(np.r_[dense,cap[cap>=g['centers'][-1]],-cap[cap>=g['centers'][-1]],g['gb']])
    curve=np.column_stack([dense,meridian(dense,c)])
    starts=curve[:-1]; vectors=np.diff(curve,axis=0)
    mids=starts+vectors/2
    points=np.column_stack([np.repeat(z,nr),np.tile(r,nz)])
    tree=cKDTree(mids)
    _,indices=tree.query(points,k=4,workers=1)
    a=starts[indices]; v=vectors[indices]
    t=np.clip(np.sum((points[:,None,:]-a)*v,axis=2)/np.sum(v*v,axis=2),0,1)
    distance=np.sqrt(np.min(np.sum((points[:,None,:]-a-t[:,:,None]*v)**2,axis=2),axis=1)).reshape(nz,nr)
    inside=(r[None,:]<=meridian(z,c)[:,None]) & (np.abs(z[:,None])<tip)
    f=0.5*(1+np.tanh(np.where(inside,distance,-distance)/c.width))
    a=obstacle_profile(z,c.width/np.pi,g['gb'][0])
    b=obstacle_profile(z,c.width/np.pi,g['gb'][1])
    ownership=np.stack([1-a,a-b,b])[:,:,None]*np.ones((1,1,nr))
    eta=ownership*f[None,:,:]
    return dict(config=c,f=f,ownership=ownership,eta=eta,z=z,r_c=r,r_f=rf,
                dr=dx,dz=dx,**g)


def grain_volumes(f,g):
    return np.array([axisym_volume(f*p,g['r_c'],g['dr'],g['dz']) for p in g['ownership']])


def topology_status(f,g):
    """Conservative stop before either contact or the center is unresolved."""
    width=g['config'].width
    span=float(g['gb'][1]-g['gb'][0])
    center_radius=(3*grain_volumes(f,g)[1]/(4*np.pi))**(1/3)
    center_rows=np.abs(g['z'])<g['gb'][1]-2*width
    radius=np.sqrt(np.maximum(0,2*np.sum(f*g['r_c'][None,:],axis=1)*g['dr']))
    minimum=float(np.min(radius[center_rows])) if np.any(center_rows) else 0.0
    reasons=[]
    if span < 8*width: reasons.append('GB/TJ regions too close')
    if center_radius < 4*width: reasons.append('center equivalent radius under-resolved')
    if minimum < 3*width: reasons.append('center interior approaching unresolved pinch-off')
    return dict(stop=bool(reasons),reasons=reasons,center_span_m=span,center_min_radius_m=minimum)
