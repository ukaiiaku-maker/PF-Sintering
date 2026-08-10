from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import convolve2d


@dataclass(frozen=True)
class ProjectionReport:
    target_masses: tuple[float, float, float]
    final_masses: tuple[float, float, float]
    residuals: tuple[float, float, float]
    iterations: int


def eta_masses(eta1, eta2, eta3, use_eta3: bool) -> tuple[float, float, float]:
    return (
        float(np.sum(eta1, dtype=np.float64)),
        float(np.sum(eta2, dtype=np.float64)),
        float(np.sum(eta3, dtype=np.float64)) if use_eta3 else 0.0,
    )


def _smooth_affinity(a: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=float) / 9.0
    return convolve2d(np.maximum(a, 0.0), kernel, mode="same", boundary="fill")


def _add_with_capacity(
    field: np.ndarray,
    headroom: np.ndarray,
    weight: np.ndarray,
    amount: float,
    tol: float,
) -> float:
    """Add as much of ``amount`` as possible without exceeding headroom."""
    remaining = float(max(amount, 0.0))
    for _ in range(32):
        if remaining <= tol:
            break
        # Milestone 15B: `tol` is scaled to the OVERALL problem magnitude
        # (rtol*scale_ref, scale_ref~total solid mass), not a per-cell
        # quantity. When admissible headroom is thin (every individual
        # cell's headroom below that aggregate-scaled `tol`, but their SUM
        # is well above it -- occurs once initialize_fields makes
        # eta1+eta2=f exactly, leaving no generous initial slack), masking
        # on `headroom > tol` discarded that entire reachable capacity and
        # this loop could return early with `remaining` still nonzero even
        # though `sum(headroom) >> remaining`. Masking on `headroom > 0`
        # instead keeps every genuinely available cell reachable; a cell
        # only reaches here if `headroom = max(fb-total, 0)` is already
        # nonnegative by construction, so this admits no negative/roundoff
        # capacity that wasn't already implied by `headroom` itself.
        mask = headroom > 0
        if not np.any(mask):
            break
        w = np.where(mask, np.maximum(weight, 0.0), 0.0)
        sw = float(np.sum(w, dtype=np.float64))
        if sw <= tol:
            w = np.where(mask, headroom, 0.0)
            sw = float(np.sum(w, dtype=np.float64))
            if sw <= tol:
                break
        proposal = remaining * w / sw
        add = np.minimum(proposal, headroom)
        added = float(np.sum(add, dtype=np.float64))
        if added <= tol:
            break
        field += add
        headroom -= add
        remaining -= added
    return remaining


def project_eta_mass_preserving(
    f,
    eta1,
    eta2,
    eta3,
    params,
    *,
    target_masses: tuple[float, float, float] | None = None,
    rtol: float = 5e-12,
    max_iterations: int = 32,
    return_report: bool = False,
):
    """Enforce local eta admissibility without changing grain ownership totals.

    Physical contract
    -----------------
    This operator is numerical regularization only.  It enforces

        0 <= eta_i,          sum_i eta_i <= f_bounded

    while preserving the integrated ownership of every active grain.  Any
    ownership removed by a local cap is redistributed into available solid
    capacity, preferentially near the same grain.  Therefore this projection
    cannot become an unintended coarsening/dissolution mechanism.

    Explicit physical operators such as Ostwald transfer must update eta before
    calling this routine and pass their *post-operator* masses as the targets.
    RBM/CH/structural smoothing should pass the pre-operator masses when those
    mechanisms are intended to preserve grain volume.
    """
    fb = np.clip(np.asarray(f, dtype=float), 0.0, 1.0)
    use_eta3 = bool(params.use_eta3)
    raw = [
        np.asarray(eta1, dtype=float),
        np.asarray(eta2, dtype=float),
        np.asarray(eta3, dtype=float) if use_eta3 else np.zeros_like(fb),
    ]

    if target_masses is None:
        target_masses = eta_masses(raw[0], raw[1], raw[2], use_eta3)
    targets = np.array(target_masses, dtype=float)
    if not use_eta3:
        targets[2] = 0.0
    if np.any(~np.isfinite(targets)) or np.any(targets < -1e-12):
        raise RuntimeError(f"invalid eta mass targets: {targets.tolist()}")
    targets = np.maximum(targets, 0.0)

    capacity = float(np.sum(fb, dtype=np.float64))
    scale_ref = max(capacity, float(np.max(targets)), 1.0)
    tol = rtol * scale_ref
    if float(np.sum(targets)) > capacity + tol:
        raise RuntimeError(
            "eta mass-preserving projection is infeasible: "
            f"target sum={float(np.sum(targets)):.16e} exceeds "
            f"solid capacity={capacity:.16e}"
        )

    # Start from the raw structural fields, remove negative values, and enforce
    # the pointwise solid-capacity constraint by proportional local scaling.
    fields = [np.maximum(a, 0.0).copy() for a in raw]
    if not use_eta3:
        fields[2].fill(0.0)

    total = fields[0] + fields[1] + fields[2]
    over = total > fb
    if np.any(over):
        local_scale = fb[over] / np.maximum(total[over], 1e-300)
        for a in fields:
            a[over] *= local_scale

    # If removal of small negative overshoots caused a grain to end above its
    # target mass, reduce it globally.  This operation cannot violate the local
    # capacity constraint and creates headroom for the redistribution stage.
    for i in range(3):
        if i == 2 and not use_eta3:
            continue
        current = float(np.sum(fields[i], dtype=np.float64))
        if current > targets[i] + tol:
            fields[i] *= targets[i] / current

    # Affinity starts from both the raw field and the admissible field.  It is
    # refreshed each iteration, allowing redistribution to expand locally if a
    # clipped interfacial band needs more capacity than its immediate neighbors.
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        currents = np.array(
            [float(np.sum(a, dtype=np.float64)) for a in fields], dtype=float
        )
        deficits = targets - currents
        deficits[2] = 0.0 if not use_eta3 else deficits[2]
        if np.max(np.abs(deficits)) <= tol:
            break

        # Numerical overshoot of a target can be removed without harming local
        # admissibility.  Do that before filling positive deficits.
        for i in range(3):
            if i == 2 and not use_eta3:
                continue
            if deficits[i] < -tol:
                cur = float(np.sum(fields[i], dtype=np.float64))
                fields[i] *= targets[i] / cur

        total = fields[0] + fields[1] + fields[2]
        headroom = np.maximum(fb - total, 0.0)
        need = np.maximum(
            targets
            - np.array([float(np.sum(a, dtype=np.float64)) for a in fields]),
            0.0,
        )
        if not use_eta3:
            need[2] = 0.0
        if float(np.sum(need)) <= tol:
            break
        if float(np.sum(headroom, dtype=np.float64)) + tol < float(np.sum(need)):
            raise RuntimeError(
                "eta projection lost feasible solid capacity during redistribution"
            )

        order = np.argsort(-need)
        for i in order:
            if need[i] <= tol or (i == 2 and not use_eta3):
                continue
            total = fields[0] + fields[1] + fields[2]
            headroom = np.maximum(fb - total, 0.0)
            affinity = _smooth_affinity(raw[i] + fields[i])
            # Bias toward existing solid ownership while keeping a tiny fallback
            # everywhere that solid capacity exists.
            weight = headroom * (affinity + 1e-14 * fb)
            remaining = _add_with_capacity(
                fields[i], headroom, weight, float(need[i]), tol
            )
            if remaining > tol:
                # Fail-open locality is not acceptable: if local affinity cannot
                # hold the preserved mass, use all remaining admissible solid
                # capacity rather than silently discarding ownership.
                total = fields[0] + fields[1] + fields[2]
                headroom = np.maximum(fb - total, 0.0)
                remaining = _add_with_capacity(
                    fields[i], headroom, headroom, remaining, tol
                )
            if remaining > tol:
                raise RuntimeError(
                    f"could not restore eta{i+1} mass; residual={remaining:.6e}"
                )
    else:
        raise RuntimeError("eta mass-preserving projection did not converge")

    final_masses = eta_masses(fields[0], fields[1], fields[2], use_eta3)
    residuals = tuple(float(final_masses[i] - targets[i]) for i in range(3))
    if max(abs(r) for r in residuals) > 10 * tol:
        raise RuntimeError(
            "eta mass-preserving projection failed mass closure: "
            f"residuals={residuals}"
        )

    # Final pointwise fail-closed checks.
    total = fields[0] + fields[1] + fields[2]
    local_tol = 1e-11
    if np.min(fields[0]) < -local_tol or np.min(fields[1]) < -local_tol:
        raise RuntimeError("negative eta created by constrained projection")
    if use_eta3 and np.min(fields[2]) < -local_tol:
        raise RuntimeError("negative eta3 created by constrained projection")
    if np.max(total - fb) > local_tol:
        raise RuntimeError("eta constrained projection exceeded local solid capacity")

    result = (fields[0], fields[1], fields[2])
    if return_report:
        return result, ProjectionReport(
            tuple(float(x) for x in targets),
            tuple(float(x) for x in final_masses),
            residuals,
            iterations,
        )
    return result
