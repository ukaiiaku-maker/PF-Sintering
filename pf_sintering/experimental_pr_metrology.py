"""Online metrology for the single experimental particle PR pathway."""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import savgol_filter

from .axisym import axisym_mu_f_gb
from .pr_experimental_geometry import experimental_geometry_record
from .pr_stress_metrology import convex_mean_width


def _outer_contour_and_sample(f, sampled, r_c, level=0.5):
    radius = np.full(f.shape[0], np.nan)
    values = np.full(f.shape[0], np.nan)
    for j, row in enumerate(f):
        crossing = np.where((row[:-1]-level)*(row[1:]-level) < 0.0)[0]
        if not len(crossing):
            continue
        i = int(crossing[-1])
        fraction = (level-row[i])/(row[i+1]-row[i])
        radius[j] = r_c[i]+fraction*(r_c[i+1]-r_c[i])
        values[j] = sampled[j, i]+fraction*(sampled[j, i+1]-sampled[j, i])
    return radius, values


def _particle_silhouette(branch):
    z = np.asarray(branch["z_m"], dtype=float)
    r = np.asarray(branch["r_m"], dtype=float)
    points = np.vstack([
        np.column_stack([z, r]),
        np.column_stack([z[::-1], -r[::-1]])])
    mean_width = convex_mean_width(points)
    normal_width = float(2.0*np.max(r))
    result = dict(
        P_h_particle_m=float(mean_width["convex_hull_perimeter"]),
        w_bar_2D_particle_m=float(mean_width["mean_width_2D"]),
        # The rigid densification coordinate is z, hence the requested
        # particle support *normal* to densification is the radial diameter.
        w_N_particle_m=normal_width,
        particle_axial_length_m=float(np.max(z)-np.min(z)),
        particle_max_radius_m=float(np.max(r)),
        particle_z_at_max_radius_m=float(z[np.argmax(r)]))
    # These primitive names are the experimental-analysis quantities.  The
    # generic two-grain extractor also emits names with these spellings, but
    # those include the cropped substrate; replace them with particle-only
    # values before a scalar record is serialized.
    result.update(
        P_h_m=result["P_h_particle_m"],
        w_bar_2D_m=result["w_bar_2D_particle_m"],
        w_N_m=result["w_N_particle_m"])
    return result


def _side_first_stresses(geometry, particle_shape):
    rn = geometry["r_n_m"]
    gamma = geometry["gamma_J_per_m2"]
    k_mw = 2.0/particle_shape["w_bar_2D_particle_m"]
    k_n = 2.0/particle_shape["w_N_particle_m"]
    families = {name: [] for name in ("local", "integral", "MW", "N")}
    side_output = {}
    for side in ("negative", "positive"):
        theta = geometry[f"theta_{side}_rad"]
        contact = math.sin(0.5*theta)/rn
        line = 2.0*math.sin(0.5*theta)/rn
        k_local = geometry[f"kappa1_{side}_per_m"]
        k_integral = (
            geometry[f"delta_phi_f_{side}_rad"]
            / geometry[f"L_f_{side}_m"])
        for label, curvature in (
                ("local", k_local), ("integral", k_integral),
                ("MW", k_mw), ("N", k_n)):
            sigma = gamma*(-curvature+contact+line)
            families[label].append(sigma)
            side_output[f"sigma_{label}_{side}_Pa"] = float(sigma)
        side_output[f"contact_{side}_per_m"] = float(contact)
        side_output[f"line_3D_{side}_per_m"] = float(line)
    side_output.update(
        sigma_local_Pa=float(np.mean(families["local"])),
        sigma_integral_Pa=float(np.mean(families["integral"])),
        sigma_MW_Pa=float(np.mean(families["MW"])),
        sigma_N_Pa=float(np.mean(families["N"])),
        k_MW_per_m=float(k_mw), k_N_per_m=float(k_n))
    return side_output


def _local_curvatures(z, radius, indices, half_window_m):
    km, kt = [], []
    finite = np.isfinite(radius)
    for index in np.asarray(indices, dtype=int):
        selected = finite & (np.abs(z-z[index]) <= half_window_m)
        if np.count_nonzero(selected) < 7:
            continue
        scale = half_window_m
        coefficient = np.polyfit(
            (z[selected]-z[index])/scale, radius[selected], 3)
        slope = float(coefficient[-2]/scale)
        second = float(2.0*coefficient[-3]/scale**2)
        denominator = math.sqrt(1.0+slope*slope)
        km.append(-second/denominator**3)
        kt.append(1.0/(radius[index]*denominator))
    if not km:
        raise ValueError("capillary support has no resolvable curvature fits")
    return np.asarray(km), np.asarray(kt)


def capillary_neck_particle_drive(state, setup, geometry, branches):
    """Neck-minus-broad-particle chemical potential outside the TJ core."""
    f, particle, substrate = state
    mu = axisym_mu_f_gb(
        f, particle, substrate, setup["p"], setup["Wc"], setup["dr"],
        setup["dz"], setup["r_c"], setup["r_f"], bc_z="noflux")
    radius, mu_contour = _outer_contour_and_sample(f, mu, setup["r_c"])
    z = np.asarray(setup["z"])
    positive = np.where(np.isfinite(radius) & (z >= geometry["z_TJ_m"]))[0]
    if len(positive) < 20:
        raise ValueError("particle contour is under-resolved")
    # Smooth only the sharp-contour diagnostic.  The PF mu samples remain the
    # actual variational field interpolated at the f=0.5 contour.
    smooth = radius.copy()
    count = len(positive)
    window = max(7, int(round(2.0*setup["W"]/setup["dz"])) | 1)
    window = min(window, count-(1-count % 2))
    if window >= 7:
        smooth[positive] = savgol_filter(radius[positive], window, 3,
                                         mode="interp")
    zp, rp = z[positive], smooth[positive]
    ds = np.hypot(np.diff(zp), np.diff(rp))
    arclength = np.r_[0.0, np.cumsum(ds)]
    neck_local = np.where(
        (arclength >= 2.0*setup["W"])
        & (arclength <= 4.0*setup["W"]))[0]
    neck_indices = positive[neck_local]
    # The broad particle reservoir follows the current maximum-radius region,
    # excluding the contact and terminal cap by construction.
    usable = (
        (zp >= geometry["z_TJ_m"]+8.0*setup["W"])
        & (zp <= np.max(zp)-8.0*setup["W"]))
    if not np.any(usable):
        raise ValueError("particle has no far-field capillary support")
    far_local_center = int(np.where(usable)[0][np.argmax(rp[usable])])
    z_far = float(zp[far_local_center])
    far_indices = positive[np.abs(zp-z_far) <= 2.0*setup["W"]]
    if min(len(neck_indices), len(far_indices)) < 5:
        raise ValueError("neck or particle capillary support is too small")
    km_neck, kt_neck = _local_curvatures(
        z, smooth, neck_indices, 1.5*setup["W"])
    km_far, kt_far = _local_curvatures(
        z, smooth, far_indices, 2.0*setup["W"])
    mu_geom_neck = setup["gamma_s"]*float(np.mean(km_neck+kt_neck))
    mu_geom_far = setup["gamma_s"]*float(np.mean(km_far+kt_far))
    mu_pf_neck = float(np.mean(mu_contour[neck_indices]))
    mu_pf_far = float(np.mean(mu_contour[far_indices]))
    return dict(
        mu_neck_geom_Pa=mu_geom_neck,
        mu_particle_geom_Pa=mu_geom_far,
        delta_mu_PR_geom_Pa=mu_geom_neck-mu_geom_far,
        mu_neck_PF_Pa=mu_pf_neck,
        mu_particle_PF_Pa=mu_pf_far,
        delta_mu_PR_PF_Pa=mu_pf_neck-mu_pf_far,
        kappa_m_neck_support_per_m=float(np.mean(km_neck)),
        kappa_theta_neck_support_per_m=float(np.mean(kt_neck)),
        kappa_m_particle_support_per_m=float(np.mean(km_far)),
        kappa_theta_particle_support_per_m=float(np.mean(kt_far)),
        z_particle_support_m=z_far,
        neck_support_start_W=2.0, neck_support_end_W=4.0,
        particle_support_half_width_W=2.0,
        contour_smoothing_window_points=int(window),
        neck_support_points=int(len(neck_indices)),
        particle_support_points=int(len(far_indices)))


def measure_experimental_pr_state(state, setup, evaluator):
    node = evaluator(*state)
    geometry, branches = experimental_geometry_record(
        state, setup, z_tj_m=node["z_TJ_m"], r_tj_m=node["r_TJ_m"])
    particle_shape = _particle_silhouette(branches["positive"])
    stresses = _side_first_stresses(geometry, particle_shape)
    capillary = capillary_neck_particle_drive(
        state, setup, geometry, branches)
    scalar = {**geometry, **particle_shape, **stresses, **capillary}
    return scalar, branches, node
