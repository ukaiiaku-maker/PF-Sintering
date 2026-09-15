"""Map a screened sharp three-particle profile to the existing PF fields."""
from __future__ import annotations

import math
import json
import numpy as np
from scipy.spatial import cKDTree

from .axisym import r_centers_faces
from .gb_obstacle_energy import obstacle_profile
from .three_particle_geometry import ThreeParticleConfig
from .three_particle_sharp_design import (
    SharpDesign, center_squared_radius_coefficients, evaluate_even_squared_radius,
    evaluate_polynomial, outer_squared_radius_coefficients,
)


def sharp_radius(z, design: SharpDesign):
    """Evaluate the complete symmetric sharp contour at axial positions z."""
    z=np.asarray(z,dtype=float);a=np.abs(z);half=.5*design.Lc_m
    outer_coeff,length=outer_squared_radius_coefficients(design,0.)
    result=np.zeros_like(a)
    center=a<=half
    uc=np.clip(a[center]/half,0,1)
    result[center]=np.sqrt(np.maximum(0,evaluate_even_squared_radius(
        center_squared_radius_coefficients(design),uc)))
    outer=(a>half)&(a<=half+length)
    uo=np.clip((a[outer]-half)/length,0,1)
    result[outer]=np.sqrt(np.maximum(0,evaluate_polynomial(outer_coeff,uo)))
    return result


def map_sharp_to_pf(design: SharpDesign, spacing=0.5e-9, margin_widths=10):
    """Signed-distance mapping with GBs exactly aligned to finite-volume faces."""
    if spacing>design.W_m/4:raise ValueError('interface is under-resolved')
    half=.5*design.Lc_m;_,outer_length=outer_squared_radius_coefficients(design,0.)
    tip=half+outer_length
    dz=half/int(np.ceil(half/spacing))
    nz=2*int(np.ceil((tip+margin_widths*design.W_m)/dz))
    probe=np.linspace(-tip,tip,12001);rmax=float(np.max(sharp_radius(probe,design)))
    nr=int(np.ceil((rmax+margin_widths*design.W_m)/spacing))
    z=(np.arange(nz)-(nz-1)/2)*dz;r,rf=r_centers_faces(nr,spacing)
    dense=np.linspace(-tip,tip,int(np.ceil(2*tip/(spacing/12)))+1)
    zz=np.unique(np.r_[dense,-half,0.,half])
    curve=np.column_stack([zz,sharp_radius(zz,design)])
    starts=curve[:-1];vectors=np.diff(curve,axis=0);tree=cKDTree(starts+vectors/2)
    distance=np.empty((nz,nr))
    for j in range(0,nz,48):
        zp=z[j:j+48];points=np.column_stack([np.repeat(zp,nr),np.tile(r,len(zp))])
        _,idx=tree.query(points,k=6,workers=1);aa=starts[idx];vv=vectors[idx]
        tt=np.clip(np.sum((points[:,None]-aa)*vv,axis=2)/np.maximum(np.sum(vv*vv,axis=2),1e-300),0,1)
        dd=np.sqrt(np.min(np.sum((points[:,None]-aa-tt[:,:,None]*vv)**2,axis=2),axis=1))
        distance[j:j+len(zp)]=dd.reshape(len(zp),nr)
    inside=(r[None,:]<sharp_radius(z,design)[:,None])&(np.abs(z[:,None])<tip)
    f=.5*(1+np.tanh(np.where(inside,distance,-distance)/design.W_m))
    left=obstacle_profile(z,design.W_m/np.pi,-half)
    right=obstacle_profile(z,design.W_m/np.pi,half)
    ownership=np.stack([1-left,left-right,right])[:,:,None]*np.ones((1,1,nr))
    config=ThreeParticleConfig(outer_radius=design.Ro_m,center_ratio=design.Rc_over_Ro,
        neck_ratio=min(.79,design.rTJ0_m/min(design.Ro_m,design.Rc_m)),width=design.W_m,
        spacing=spacing,psi_deg=.5*(design.theta_center_deg+design.theta_outer_deg),
        margin_widths=margin_widths)
    # Centroids are measured from the mapped ownership below; these are sharp
    # metadata only and never drive evolution.
    return dict(config=config,f=f,ownership=ownership,eta=ownership*f[None],z=z,
        r_c=r,r_f=rf,dr=spacing,dz=dz,gb=np.array([-half,half]),neck=design.rTJ0_m,
        centers=np.array([-half-.5*outer_length,0.,half+.5*outer_length]),
        radii=np.array([design.Ro_m,design.Rc_m,design.Ro_m]),sharp_design=design)


def load_mapped_sharp_state(path):
    """Reconstruct geometry and fields exactly from a mapped source archive."""
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        design=SharpDesign(**meta['sharp_design'])
        config=ThreeParticleConfig(outer_radius=design.Ro_m,center_ratio=design.Rc_over_Ro,
            neck_ratio=min(.79,design.rTJ0_m/min(design.Ro_m,design.Rc_m)),width=design.W_m,
            spacing=float(data['dr']),psi_deg=.5*(design.theta_center_deg+design.theta_outer_deg),
            margin_widths=10)
        g=dict(config=config,f=data['f'].copy(),ownership=data['ownership'].copy(),
            z=data['z'].copy(),r_c=data['r_c'].copy(),r_f=data['r_f'].copy(),
            gb=data['gb'].copy(),dr=float(data['dr']),dz=float(data['dz']),
            radii=data['radii'].copy(),centers=data['centers'].copy(),neck=design.rTJ0_m,
            sharp_design=design)
        g['eta']=g['ownership']*g['f'][None]
        fields=tuple(value.copy() for value in data['fields'])
    return g,fields,meta
