import math

import numpy as np
from types import SimpleNamespace

from pf_sintering.gb_obstacle_energy import (
    gb_obstacle_coefficients,
    obstacle_ell,
    obstacle_profile,
)
from pf_sintering.corrected_interfacial_energy import (
    axisym_corrected_interfacial_energy_components,
    normalized_ownership_variational_derivatives_axisym,
)
from pf_sintering.corrected_pr_thermodynamics import (
    _radial_area_weights_to_cutoff,
)


def test_continuous_gb_radial_weights_are_exact_and_jump_free():
    faces = np.linspace(0.0, 5.0, 6)
    for cutoff in (0.2, 0.999999, 1.0, 1.000001, 2.7, 5.0):
        weights = _radial_area_weights_to_cutoff(faces, cutoff)
        assert np.isclose(np.sum(weights), 0.5 * min(cutoff, 5.0) ** 2)
    below = _radial_area_weights_to_cutoff(faces, 1.0 - 1e-8)
    above = _radial_area_weights_to_cutoff(faces, 1.0 + 1e-8)
    assert np.linalg.norm(above - below, ord=1) < 2.1e-8


def test_parallel_corrected_planar_binary_energies():
    gamma_s = 1.0
    gamma_gb = 1.0
    width = 10.0e-9
    W_f = 12.0 * gamma_s / width
    k_f = 3.0 * gamma_s * width
    coeff = gb_obstacle_coefficients(gamma_gb, width)
    dx = width / 20_000.0
    coordinate = np.arange(-12.0*width, 12.0*width+dx, dx)

    f = 0.5 * (1.0 - np.tanh(coordinate / width))
    fp = -0.5 / width / np.cosh(coordinate / width)**2
    surface = np.trapezoid(
        0.5 * W_f * f*f * (1.0-f)**2 + 0.5 * k_f * fp*fp,
        coordinate)

    phi = obstacle_profile(
        coordinate, obstacle_ell(coeff["k_eta"], coeff["Wc"]))
    phip = np.gradient(phi, dx, edge_order=2)
    gb = np.trapezoid(
        coeff["Wc"] * phi * (1.0-phi) + coeff["k_eta"] * phip*phip,
        coordinate)

    assert math.isclose(surface, gamma_s, rel_tol=1.0e-10)
    assert math.isclose(gb, gamma_gb, rel_tol=1.0e-8)
    assert math.isclose(gb / (2.0*surface), 0.5, rel_tol=1.0e-8)


def test_parallel_corrected_homogeneous_and_single_grain_backgrounds_vanish():
    # The corrected ownership excess is Wc*eta1*eta2*f*(2-f) and
    # (k_eta/2)(|grad eta1|^2+|grad eta2|^2-|grad f|^2).
    # Both vanish identically for homogeneous phases and for eta1=f,eta2=0.
    f = np.linspace(0.0, 1.0, 101)
    eta1 = f.copy()
    eta2 = np.zeros_like(f)
    coupling_excess = eta1 * eta2 * f * (2.0-f)
    grad_f = np.gradient(f)
    gradient_excess = (
        np.gradient(eta1)**2 + np.gradient(eta2)**2 - grad_f**2)
    assert np.max(np.abs(coupling_excess)) == 0.0
    assert np.max(np.abs(gradient_excess)) == 0.0


def test_normalized_ownership_derivatives_match_discrete_energy():
    rng = np.random.default_rng(113)
    nz, nr = 9, 8
    dr, dz = 1.2, 0.9
    r_c = (np.arange(nr) + 0.5) * dr
    r_f = np.arange(nr + 1) * dr
    p = SimpleNamespace(W_f=2.3, k_f=1.7, k_eta=0.8)
    Wc = 1.4
    f = 0.2 + 0.6 * rng.random((nz, nr))
    phi = 0.1 + 0.8 * rng.random((nz, nr))
    eta1, eta2 = f * phi, f * (1.0 - phi)
    mu, gphi = normalized_ownership_variational_derivatives_axisym(
        f, eta1, eta2, p, Wc, dr, dz, r_c, r_f)
    cell_measure = 2.0 * math.pi * dr * dz * r_c[None, :]

    def energy(f_now, phi_now):
        return axisym_corrected_interfacial_energy_components(
            f_now, f_now * phi_now, f_now * (1.0 - phi_now),
            p, Wc, dr, dz, r_c, r_f)["F_total_corrected"]

    for field_name, derivative in (("f", mu), ("phi", gphi)):
        direction = rng.normal(size=f.shape)
        epsilon = 1.0e-7
        if field_name == "f":
            finite_difference = (
                energy(f + epsilon * direction, phi)
                - energy(f - epsilon * direction, phi)) / (2.0 * epsilon)
        else:
            finite_difference = (
                energy(f, phi + epsilon * direction)
                - energy(f, phi - epsilon * direction)) / (2.0 * epsilon)
        analytical = float(np.sum(cell_measure * derivative * direction))
        assert math.isclose(
            finite_difference, analytical, rel_tol=2.0e-8, abs_tol=2.0e-8)
