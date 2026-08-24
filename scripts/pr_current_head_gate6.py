"""Gate 6: constrained sharp-interface Hessians of the two exact candidates.

The finite simulation domains are represented by the stability module's open
(``capped``) frustum chain.  No artificial free-surface end caps are added.
There is one interior GB disk at the particle/substrate contact.  Coordinates
are nondimensionalized by each profile's maximum radius; this leaves the
eigenvalue sign unchanged and avoids finite-difference roundoff in SI metres.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pf_sintering.sharp_interface_stability import (  # noqa: E402
    energy, gradient_fd, hessian_fd, total_volume,
)

OUT = ROOT / "runs" / "pr_current_head_regression"
GAMMA_S = 1.0
GAMMA_GB = 0.34729635533386083
EPS1 = 0.4
EPS2 = 0.0
LAMBDA_OVER_RCYL = 2.0*math.sqrt(2.0)*math.pi
R0_OVER_RCYL = math.sqrt(1.0-0.5*EPS1**2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def experimental_curve():
    source = ROOT / "sinter_results_v83.mat"
    checkpoint = (ROOT / "runs" / "pr_experiment_derived_completion" /
                  "checkpoints" / "experimental_surrogate_initial.npz")
    data = np.load(checkpoint)
    f = np.asarray(data["current_f"])
    z = np.asarray(data["z"])
    radial = np.asarray(data["r_c"])
    radius = np.full(len(z), np.nan)
    for i, row in enumerate(f):
        crossings = np.where((row[:-1] >= 0.5) & (row[1:] < 0.5))[0]
        if len(crossings):
            j = int(crossings[-1])
            fraction = (0.5-row[j])/(row[j+1]-row[j])
            radius[i] = radial[j]+fraction*(radial[j+1]-radial[j])
    valid = np.isfinite(radius)
    first, last = np.where(valid)[0][[0, -1]]
    segment = np.arange(first, last+1)
    good = segment[valid[segment]]
    radius = np.interp(segment, good, radius[good])
    z = z[segment]
    # The PF checkpoint is the exact initialized surrogate actually evolved;
    # append the known sharp zero-radius particle tip for the open chain.
    z_tip = float(data["particle_axial_length_m"])
    if z[-1] < z_tip:
        z = np.r_[z, z_tip]
        radius = np.r_[radius, 0.0]
    return z, radius, float(data["z_TJ_initial_m"]), {
        "source": str(source), "source_sha256": sha256(source),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "generator": "f=0.5 contour of exact initialized experimental surrogate",
        "scale_factor": 1.0,
    }


def fourier_curve():
    rcyl = 1.0
    lam = LAMBDA_OVER_RCYL * rcyl
    z_crest = lam
    r_crest = rcyl * (R0_OVER_RCYL + EPS1)
    curvature_crest = -rcyl * EPS1 * (2.0 * math.pi / lam) ** 2
    a_cap = math.sqrt(r_crest / abs(curvature_crest))
    z = np.linspace(0.0, z_crest + a_cap, 2001)
    radius = rcyl * (R0_OVER_RCYL + EPS1*np.cos(2.0*math.pi*z/lam))
    cap = r_crest*np.sqrt(np.maximum(0.0, 1.0-((z-z_crest)/a_cap)**2))
    radius[z >= z_crest] = cap[z >= z_crest]
    return z, radius, 0.5*lam, {
        "generator": "fourier_max_pr_geometry exact sharp profile plus C3 crest cap",
        "epsilon_1": EPS1, "epsilon_2": EPS2,
        "lambda_over_R_cyl": LAMBDA_OVER_RCYL,
        "R0_over_R_cyl": R0_OVER_RCYL, "a_cap_over_R_cyl": a_cap,
    }


def fixed_tip_constrained_modes(radius, dz, igb, n_modes=4):
    """Constrained Lagrangian Hessian with the axis closure held at R=0."""
    radius = np.asarray(radius, dtype=float)
    free = np.arange(len(radius)-1)  # final point is the physical axis tip
    base = radius.copy()

    def expand(values):
        result = base.copy()
        result[free] = values
        return result

    def f_energy(values):
        return energy(expand(values), dz, GAMMA_S, GAMMA_GB, [igb], "capped")

    def f_volume(values):
        return total_volume(expand(values), dz, "capped")

    values = base[free]
    gf = gradient_fd(f_energy, values)
    gv = gradient_fd(f_volume, values)
    lagrange = float(np.dot(gf, gv)/np.dot(gv, gv))
    hl = hessian_fd(f_energy, values)-lagrange*hessian_fd(f_volume, values)
    ghat = gv/np.linalg.norm(gv)
    projector = np.eye(len(values))-np.outer(ghat, ghat)
    u, singular, _ = np.linalg.svd(projector)
    basis = u[:, singular > 1e-9*singular[0]]
    hp = basis.T@hl@basis
    evals, reduced = np.linalg.eigh(0.5*(hp+hp.T))
    order = np.argsort(evals)
    evals = evals[order][:n_modes]
    modes = np.zeros((len(radius), len(evals)))
    modes[free] = (basis@reduced[:, order[:n_modes]])
    residual = gf-lagrange*gv
    return evals, modes, lagrange, float(np.linalg.norm(gv)), float(
        np.linalg.norm(residual)/(np.linalg.norm(gf)+1e-300))


def classify(name, curve_builder):
    z_raw, r_raw, z_gb, provenance = curve_builder()
    rscale = float(np.max(r_raw))
    z_raw = z_raw/rscale
    r_raw = r_raw/rscale
    z_gb /= rscale
    levels = []
    finest = None
    for n in (65, 97, 129):
        z = np.linspace(float(z_raw[0]), float(z_raw[-1]), n)
        radius = np.interp(z, z_raw, r_raw)
        dz = float(z[1]-z[0])
        igb = int(np.argmin(np.abs(z-z_gb)))
        evals, modes, lagrange, gnorm, residual_relative = (
            fixed_tip_constrained_modes(radius, dz, igb, n_modes=4))
        row = {
            "n": n, "dz_over_Rmax": dz, "gb_index": igb,
            "gb_z_error_over_Rmax": float(z[igb]-z_gb),
            "lowest_eigenvalues": [float(value) for value in evals],
            "lambda_lagrange": float(lagrange),
            "volume_gradient_norm": float(gnorm),
            "tangent_gradient_residual_relative": residual_relative,
        }
        levels.append(row)
        finest = (z, radius, modes, igb)
    signs = [np.sign(row["lowest_eigenvalues"][0]) for row in levels]
    resolved = bool(all(sign == signs[-1] and sign != 0 for sign in signs))
    classification = "unstable" if signs[-1] < 0 else "stable"
    z, radius, modes, igb = finest
    np.savez_compressed(
        OUT / f"gate6_{name}_sharp_profile_and_modes.npz",
        z_over_Rmax=z, R_over_Rmax=radius, modes=modes, gb_index=igb)
    return {
        "candidate": name, "topology": "capped (open lateral chain)",
        "fixed_geometric_dofs": ["terminal axis closure R=0"],
        "gb_count": 1, "gamma_s": GAMMA_S, "gamma_gb": GAMMA_GB,
        "coordinate_scale_m": rscale if name == "experimental_surrogate" else None,
        "provenance": provenance, "resolution_levels": levels,
        "classification": classification,
        "sign_resolution_converged": resolved,
        "interpretation_caveat": (
            "The base profile is an as-generated kinetic initial condition, not a "
            "relaxed constrained stationary point. The reported Lagrangian Hessian "
            "sign is therefore a local curvature diagnostic; the tangent-gradient "
            "residual quantifies the nonstationarity."),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "gate": 6,
        "method": "lowest constrained Lagrangian Hessian eigenvalue",
        "source_hashes": {
            "sharp_interface_stability.py": sha256(
                ROOT / "pf_sintering" / "sharp_interface_stability.py"),
            "experimental_particle_geometry.py": sha256(
                ROOT / "pf_sintering" / "experimental_particle_geometry.py"),
            "fourier_max_pr_geometry.py": sha256(
                ROOT / "pf_sintering" / "fourier_max_pr_geometry.py"),
        },
        "candidates": [
            classify("experimental_surrogate", experimental_curve),
            classify("fourier_candidate", fourier_curve),
        ],
    }
    target = OUT / "gate6_candidate_hessians.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    for candidate in result["candidates"]:
        values = [level["lowest_eigenvalues"][0]
                  for level in candidate["resolution_levels"]]
        print(candidate["candidate"], candidate["classification"], values)
    print(target)


if __name__ == "__main__":
    main()
