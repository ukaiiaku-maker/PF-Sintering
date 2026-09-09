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
                                  spacing=1.25e-9, tail_margin_W=8.0,
                                  eps1=EPS1, radial_margin_W=6.0,
                                  close_left_substrate=False,
                                  r0_over_rcyl=None):
    """One substrate half-lobe, one particle lobe, and a C3 crest cap.

    The neck-forming ``cos(2*pi*z/lambda)`` mode has
    ``k*R_cyl=1/sqrt(2)``, the classical fastest-growing inviscid PR value.
    With ``eps2=0`` there is no unequal long-mode lobe pair.
    """
    eps1 = float(eps1)
    if not math.isfinite(eps1) or not 0.0 <= eps1 < math.sqrt(2.0):
        raise ValueError("eps1 must be finite and satisfy 0 <= eps1 < sqrt(2)")
    if r0_over_rcyl is None:
        r0_over_rcyl = math.sqrt(1.0-0.5*eps1**2)
        r0_definition = "analytic periodic-Fourier volume match"
    else:
        r0_over_rcyl = float(r0_over_rcyl)
        if not math.isfinite(r0_over_rcyl) or r0_over_rcyl <= eps1:
            raise ValueError("r0_over_rcyl must be finite and exceed eps1")
        r0_definition = "explicit numerical closed-body volume match"
    lam = LAMBDA_OVER_RCYL*R_cyl
    z_tj = 0.5*lam
    z_crest = lam
    r_crest = R_cyl*(r0_over_rcyl+eps1)
    r_neck = R_cyl*(r0_over_rcyl-eps1)
    curvature_crest = -R_cyl*eps1*(2.0*math.pi/lam)**2
    if curvature_crest == 0.0:
        raise ValueError("eps1=0 does not define the qualified ellipsoidal crest cap")
    a_cap = math.sqrt(r_crest/abs(curvature_crest))
    if tail_margin_W <= 0.0 or radial_margin_W <= 0.0:
        raise ValueError("axial and radial margins must be positive")
    z_raw_min = (-a_cap-tail_margin_W*W
                 if close_left_substrate else 0.0)
    z_raw_max = z_crest+a_cap+tail_margin_W*W
    z_extent = z_raw_max-z_raw_min
    Nz = int(math.ceil(z_extent/spacing))
    dz = z_extent/Nz
    z_raw = z_raw_min+(np.arange(Nz)+0.5)*dz
    # Public coordinates remain non-negative cell centers.  ``z_raw`` retains
    # the Fourier coordinate in which the crests are at 0 and lambda.
    z = z_raw-z_raw_min
    r_max = r_crest+radial_margin_W*W
    Nr = int(math.ceil(r_max/spacing))
    dr = r_max/Nr
    r_c, r_f = r_centers_faces(Nr, dr)
    Z, radial = np.meshgrid(z, r_c, indexing="ij")
    radius = R_cyl*(
        r0_over_rcyl+eps1*np.cos(2.0*math.pi*z_raw/lam))
    f_profile = 0.5*(1.0-np.tanh((radial-radius[:, None])/W))
    Z_raw = Z+z_raw_min
    rho_right = np.sqrt(((Z_raw-z_crest)/a_cap)**2+(radial/r_crest)**2)
    f_cap_right = 0.5*(1.0-np.tanh((rho_right-1.0)*r_crest/W))
    f = np.where((z_raw >= z_crest)[:, None], f_cap_right, f_profile)
    if close_left_substrate:
        rho_left = np.sqrt((Z_raw/a_cap)**2+(radial/r_crest)**2)
        f_cap_left = 0.5*(1.0-np.tanh((rho_left-1.0)*r_crest/W))
        f = np.where((z_raw <= 0.0)[:, None], f_cap_left, f)
    identity_width = 1.5*dz
    particle_fraction = 0.5*(1.0+np.tanh(
        (z_raw-z_tj)/identity_width))[:, None]
    particle = f*particle_fraction
    substrate = f-particle
    result = dict(
        f=f, e1=particle, e2=substrate, z=z, r_c=r_c, r_f=r_f,
        Nz=Nz, Nr=Nr, dz=dz, dr=dr,
        z1=z_tj-z_raw_min, R_z1=r_neck,
        lam=lam, R_cyl=R_cyl, z2=z_crest-z_raw_min, a_cap=a_cap,
        r_neck_sharp_m=r_neck, r_crest_sharp_m=r_crest,
        z_min_m=0.0, z_max_m=z_extent,
        z_raw_min_m=float(z_raw_min), z_raw_max_m=float(z_raw_max),
        tail_margin_W=float(tail_margin_W),
        radial_margin_W=float(radial_margin_W),
        close_left_substrate=bool(close_left_substrate),
        handoff="single_mode_maximum_PR_fourier_fallback",
        EPS1=eps1, EPS2=EPS2, R0_OVER_RCYL=r0_over_rcyl,
        R0_definition=r0_definition,
        lambda_over_R_cyl=LAMBDA_OVER_RCYL,
        kR=2.0*math.pi/LAMBDA_OVER_RCYL,
        no_unequal_lobe_coordinate=True)
    result["diffuse_volume_m3"] = float(axisym_volume(f, r_c, dr, dz))
    result["particle_diffuse_volume_m3"] = float(axisym_volume(
        particle, r_c, dr, dz))
    return result
