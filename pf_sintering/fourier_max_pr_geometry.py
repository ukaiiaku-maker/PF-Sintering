"""Corrected single-mode Fourier fallback at maximum PR growth."""
from __future__ import annotations

import math

import numpy as np

from .axisym import axisym_volume, r_centers_faces


EPS1 = 0.4
EPS2 = 0.0
LAMBDA_OVER_RCYL = 2.0*math.sqrt(2.0)*math.pi
R0_OVER_RCYL = math.sqrt(1.0-0.5*EPS1**2)


def build_fourier_max_pr_geometry(R_cyl=100e-9, W=10e-9,
                                  spacing=1.25e-9, tail_margin_W=8.0):
    """One substrate half-lobe, one particle lobe, and a C3 crest cap.

    The neck-forming ``cos(2*pi*z/lambda)`` mode has
    ``k*R_cyl=1/sqrt(2)``, the classical fastest-growing inviscid PR value.
    With ``eps2=0`` there is no unequal long-mode lobe pair.
    """
    lam = LAMBDA_OVER_RCYL*R_cyl
    z_tj = 0.5*lam
    z_crest = lam
    r_crest = R_cyl*(R0_OVER_RCYL+EPS1)
    r_neck = R_cyl*(R0_OVER_RCYL-EPS1)
    curvature_crest = -R_cyl*EPS1*(2.0*math.pi/lam)**2
    a_cap = math.sqrt(r_crest/abs(curvature_crest))
    z_end = z_crest+a_cap+tail_margin_W*W
    Nz = int(math.ceil(z_end/spacing))
    dz = z_end/Nz
    z = (np.arange(Nz)+0.5)*dz
    r_max = r_crest+6.0*W
    Nr = int(math.ceil(r_max/spacing))
    dr = r_max/Nr
    r_c, r_f = r_centers_faces(Nr, dr)
    Z, radial = np.meshgrid(z, r_c, indexing="ij")
    radius = R_cyl*(
        R0_OVER_RCYL+EPS1*np.cos(2.0*math.pi*z/lam))
    f_profile = 0.5*(1.0-np.tanh((radial-radius[:, None])/W))
    rho = np.sqrt(((Z-z_crest)/a_cap)**2+(radial/r_crest)**2)
    f_cap = 0.5*(1.0-np.tanh((rho-1.0)*r_crest/W))
    f = np.where((z >= z_crest)[:, None], f_cap, f_profile)
    identity_width = 1.5*dz
    particle_fraction = 0.5*(1.0+np.tanh(
        (z-z_tj)/identity_width))[:, None]
    particle = f*particle_fraction
    substrate = f-particle
    result = dict(
        f=f, e1=particle, e2=substrate, z=z, r_c=r_c, r_f=r_f,
        Nz=Nz, Nr=Nr, dz=dz, dr=dr, z1=z_tj, R_z1=r_neck,
        lam=lam, R_cyl=R_cyl, z2=z_crest, a_cap=a_cap,
        r_neck_sharp_m=r_neck, r_crest_sharp_m=r_crest,
        z_min_m=0.0, z_max_m=z_end,
        handoff="single_mode_maximum_PR_fourier_fallback",
        EPS1=EPS1, EPS2=EPS2,
        lambda_over_R_cyl=LAMBDA_OVER_RCYL,
        kR=2.0*math.pi/LAMBDA_OVER_RCYL,
        no_unequal_lobe_coordinate=True)
    result["diffuse_volume_m3"] = float(axisym_volume(f, r_c, dr, dz))
    result["particle_diffuse_volume_m3"] = float(axisym_volume(
        particle, r_c, dr, dz))
    return result
