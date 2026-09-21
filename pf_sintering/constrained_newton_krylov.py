"""Reduced-space constrained L-BFGS and Newton--Krylov minimization.

This solves the fixed-ownership stationarity equations on an active diffuse
interface band.  Linear volume/moment constraints are enforced in every
search direction.  Projected L-BFGS provides a robust basin solve; an analytic
Hessian-vector action and damped GMRES provide the final Newton polish.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse.linalg import LinearOperator, eigsh, gmres

from .axisym import axisym_laplacian
from .constrained_densification_relaxation import (
    constraint_basis, constraint_values, fixed_ownership_mu,
    multigrain_energy, restore_constraints,
)
from .three_particle_phase_a import FrozenPhysics


def active_interface_mask(f, *, threshold=1e-2, tail_threshold=1e-3,
                          halo_cells=8):
    """Resolved diffuse interface plus a finite-difference halo.

    The outer cutoff excludes asymptotic phase-field tails from the nonlinear
    degrees of freedom.  Its invariance is an explicit qualification check.
    """
    field = np.asarray(f)
    core = (field > threshold) & (field < 1.0-threshold)
    if halo_cells:
        core = binary_dilation(core, iterations=int(halo_cells))
    resolved = (field > tail_threshold) & (field < 1.0-tail_threshold)
    return core & resolved


class LinearConstraintProjector:
    """Euclidean projector onto ``A direction = 0`` with scaled rows."""
    def __init__(self, basis, weight, active):
        self.active = np.asarray(active, dtype=bool)
        raw = (basis*weight[None, :, :]).reshape(len(basis), -1)[:, self.active.ravel()]
        norms = np.linalg.norm(raw, axis=1)
        if np.any(norms <= 0):
            raise ValueError("constraint row has zero active support")
        self.row_scale = norms
        self.A = raw/norms[:, None]
        gram = self.A@self.A.T
        self.gram_condition = float(np.linalg.cond(gram))
        self.gram_inverse = np.linalg.inv(gram)

    def project(self, vector):
        v = np.asarray(vector, dtype=float)
        return v-self.A.T@(self.gram_inverse@(self.A@v))

    def tangent_residual(self, vector):
        return float(np.max(np.abs(self.project(vector))))


def _feasible_reduced_gradient(raw_gradient, x, A, *, minimum_step=1e-5):
    """Return the equality-projected gradient on the current box free set.

    Diffuse fields contain exponentially small tails.  A plain equality
    projection gives some of those tails outward directions and consequently
    limits the entire field to vanishing line-search steps.  They are active
    box constraints, not useful degrees of freedom.  Identify them from the
    feasible descent direction and repeat the equality projection after each
    removal.  No field value is clipped.
    """
    free = np.ones(x.size, dtype=bool)
    projected = np.zeros_like(raw_gradient)
    for _ in range(20):
        Af = A[:, free]
        gram = Af@Af.T
        gf = raw_gradient[free]
        pg = gf-Af.T@np.linalg.solve(gram, Af@gf)
        direction = -pg
        xf = x[free]
        distance = np.where(direction < 0.0, xf+2e-14,
                            np.where(direction > 0.0, 1.0+2e-14-xf, np.inf))
        fraction = distance/np.maximum(np.abs(direction), 1e-300)
        remove_local = fraction < minimum_step
        if not np.any(remove_local):
            projected[free] = pg
            return projected, free
        free_indices = np.flatnonzero(free)
        free[free_indices[remove_local]] = False
        if np.count_nonzero(free) <= A.shape[0]:
            raise RuntimeError("box active set exhausted constrained degrees of freedom")
    raise RuntimeError("box active set did not stabilize")


def exact_hessian_action(f, direction, g, physics=FrozenPhysics()):
    """Analytic fixed-ownership ``d(mu)/d(f)`` action."""
    W = g["config"].width
    Wf = 12.0*physics.gamma_s/W
    kf = 3.0*physics.gamma_s*W
    return (Wf*(1.0-6.0*f+6.0*f*f)*direction
            - kf*axisym_laplacian(direction, g["dr"], g["dz"],
                                  g["r_c"], g["r_f"], bc_z="noflux"))


def hessian_finite_difference_check(f, ownership, g, direction, *, epsilon=1e-7):
    analytic = exact_hessian_action(f, direction, g)
    plus = fixed_ownership_mu(f+epsilon*direction, ownership, g)
    minus = fixed_ownership_mu(f-epsilon*direction, ownership, g)
    finite = (plus-minus)/(2.0*epsilon)
    scale = max(float(np.max(np.abs(analytic))), 1e-300)
    return dict(relative_Linf=float(np.max(np.abs(analytic-finite)))/scale,
                analytic_Linf=scale, finite_difference_Linf=float(np.max(np.abs(finite))))


def projected_hessian_eigenpairs(
        f, ownership, g, *, active_mask, include_moments=True,
        eigenpair_count=5, tolerance=1e-7, maximum_iterations=None):
    """Lowest eigenpairs of the equality-constrained energy Hessian.

    The operator is expressed in the same dimensionless active-cell variables
    as the KKT solver. Constraint-normal directions receive a positive penalty,
    so the lowest eigenpairs belong to the admissible tangent space instead of
    appearing as artificial zero modes of ``P H P``.
    """
    field = np.asarray(f, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    if active.shape != field.shape:
        raise ValueError("active_mask shape does not match field")
    ids = np.flatnonzero(active)
    if len(ids) <= eigenpair_count:
        raise ValueError("active domain is too small for requested eigenpairs")
    scale = g["config"].outer_radius
    basis = constraint_basis(
        ownership, g["z"], scale, include_moments=include_moments)
    weight = np.broadcast_to(g["r_c"][None, :]/scale, field.shape)
    projector = LinearConstraintProjector(basis, weight, active)
    Wf = 12.0*FrozenPhysics().gamma_s/g["config"].width
    # The dimensionless Hessian spectrum is O(1--10) on the resolved grid.
    # Keeping constraint-normal modes at 1e3 places them safely above the
    # low tangent spectrum without changing any tangent eigenvalue.
    normal_penalty = 1.0e3

    def matvec(vector):
        tangent = projector.project(vector)
        full = np.zeros_like(field)
        full.ravel()[ids] = tangent
        raw = (weight*exact_hessian_action(field, full, g)/Wf).ravel()[ids]
        normal = vector-tangent
        return projector.project(raw)+normal_penalty*normal

    operator = LinearOperator(
        (len(ids), len(ids)), matvec=matvec, dtype=float)
    values, vectors = eigsh(
        operator, k=int(eigenpair_count), which="SA", tol=float(tolerance),
        maxiter=maximum_iterations)
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    tangent_residuals = np.array([
        np.max(np.abs(v-projector.project(v))) for v in vectors.T])
    modes = np.zeros((len(values),)+field.shape, dtype=float)
    for i, vector in enumerate(vectors.T):
        modes[i].ravel()[ids] = projector.project(vector)
    return dict(eigenvalues=values, modes=modes,
                tangent_residuals=tangent_residuals,
                active_cells=len(ids), gram_condition=projector.gram_condition)


@dataclass(frozen=True)
class SolverResult:
    f: np.ndarray
    converged: bool
    reason: str
    history: tuple[dict, ...]
    projected_kkt_residual: float
    normalized_constraint_residual: float
    active_cells: int
    gram_condition: float
    lbfgs_iterations: int
    newton_iterations: int


def _lbfgs_direction(gradient, pairs):
    q = gradient.copy(); alpha = []
    for s, y, rho in reversed(pairs):
        a = rho*np.dot(s, q); alpha.append(a); q -= a*y
    if pairs:
        s, y, _ = pairs[-1]
        gamma = np.dot(s, y)/max(np.dot(y, y), 1e-300)
    else:
        gamma = 1.0
    r = gamma*q
    for (s, y, rho), a in zip(pairs, reversed(alpha)):
        r += s*(a-rho*np.dot(y, r))
    return -r


def solve_constrained_stationary(
        f, ownership, g, targets, *, halo_cells=8,
        lbfgs_max_iterations=1200, newton_max_iterations=30,
        kkt_tolerance=1e-7, constraint_tolerance=2e-12,
        history_size=10, energy_relative_tolerance=1e-13, callback=None,
        active_mask=None, active_threshold=1e-2, tail_threshold=1e-3,
        include_moments=True, lbfgs_box_active_step=1e-5,
        newton_box_active_step=1e-2, gmres_maxiter=8,
        newton_minimum_damping=1e-3):
    """Solve the exact active-band KKT stationarity problem."""
    field, _ = restore_constraints(f, ownership, g, targets,
                                   include_moments=include_moments)
    phi = np.asarray(ownership)
    if active_mask is None:
        active = active_interface_mask(field,threshold=active_threshold,
            tail_threshold=tail_threshold,halo_cells=halo_cells)
    else:
        active=np.asarray(active_mask,dtype=bool).copy()
        if active.shape != field.shape:
            raise ValueError("active_mask shape does not match field")
    ids = np.flatnonzero(active)
    scale = g["config"].outer_radius
    basis = constraint_basis(phi, g["z"], scale,
                             include_moments=include_moments)
    weight = np.broadcast_to(g["r_c"][None, :]/scale, field.shape)
    projector = LinearConstraintProjector(basis, weight, active)
    Wf = 12.0*FrozenPhysics().gamma_s/g["config"].width
    history=[]; pairs=[]; energy=multigrain_energy(field,phi,g); previous_x=None;previous_grad=None;previous_free=None

    def gradient(current, *, minimum_step=1e-5):
        mu=fixed_ownership_mu(current,phi,g)
        raw=(weight*mu/Wf).ravel()[ids]
        x=current.ravel()[ids]
        return _feasible_reduced_gradient(raw,x,projector.A,
                                          minimum_step=minimum_step)

    def constraint_error(current):
        actual=constraint_values(current,phi,g["z"],g["r_c"],scale,
                                 include_moments=include_moments)
        return float(np.max(np.abs(actual-targets))/max(np.max(np.abs(targets)),1e-300))

    def maximum_bound_step(x,d):
        candidates=[1.0]
        positive=d>0;negative=d<0
        if np.any(positive):candidates.append(float(np.min((1.0+2e-14-x[positive])/d[positive])))
        if np.any(negative):candidates.append(float(np.min((-2e-14-x[negative])/d[negative])))
        return max(0.0,min(candidates))

    lbfgs_done=0
    for iteration in range(1,lbfgs_max_iterations+1):
        x=field.ravel()[ids].copy()
        audit_grad,audit_free=gradient(field,minimum_step=1e-2)
        residual=float(np.max(np.abs(audit_grad)))
        grad,free=gradient(field,minimum_step=lbfgs_box_active_step)
        if np.max(np.abs(grad))<=kkt_tolerance and residual>kkt_tolerance:
            grad,free=audit_grad,audit_free
        cerr=constraint_error(field)
        if residual<=kkt_tolerance and cerr<=constraint_tolerance:
            return SolverResult(field,True,"lbfgs_stationary",tuple(history),residual,cerr,
                                len(ids),projector.gram_condition,iteration-1,0)
        if previous_x is not None and np.array_equal(free,previous_free):
            s=x-previous_x;y=grad-previous_grad;sy=float(np.dot(s,y))
            if sy>1e-12*np.linalg.norm(s)*np.linalg.norm(y):
                pairs.append((s,y,1.0/sy));pairs=pairs[-history_size:]
        elif previous_x is not None:
            pairs=[]
        direction=np.zeros_like(grad)
        Af=projector.A[:,free]
        candidate=_lbfgs_direction(grad,pairs)[free]
        direction[free]=candidate-Af.T@np.linalg.solve(Af@Af.T,Af@candidate)
        slope=float(np.dot(grad,direction))
        if not np.isfinite(slope) or slope>=0:
            pairs=[];direction=-grad;slope=-float(np.dot(grad,grad))
        max_step=maximum_bound_step(x,direction)
        if max_step < lbfgs_box_active_step:
            pairs=[];direction=-grad;slope=-float(np.dot(grad,grad))
            max_step=maximum_bound_step(x,direction)
        step=min(1.0,0.995*max_step);accepted=False
        for _ in range(40):
            trial=field.copy();trial.ravel()[ids]=x+step*direction
            trial_energy=multigrain_energy(trial,phi,g)
            if trial_energy<=energy+1e-4*step*slope*Wf*2*math.pi*g["dr"]*g["dz"]*scale:
                accepted=True;break
            step*=0.5
        if not accepted or step<1e-14:
            break
        previous_x=x;previous_grad=grad;previous_free=free.copy();field=trial;old_energy=energy;energy=trial_energy;lbfgs_done=iteration
        if iteration==1 or iteration%10==0:
            record=dict(stage="lbfgs",iteration=iteration,energy_J=energy,
                energy_drop_J=energy-old_energy,projected_kkt_residual=residual,
                normalized_constraint_residual=cerr,step=step)
            history.append(record)
            if callback is not None: callback(field, record)

    # Damped reduced-space Newton--Krylov polish.
    newton_done=0
    accepted_damping=newton_minimum_damping
    for iteration in range(1,newton_max_iterations+1):
        x=field.ravel()[ids].copy()
        audit_grad,audit_free=gradient(field,minimum_step=1e-2)
        residual=float(np.max(np.abs(audit_grad)));cerr=constraint_error(field)
        if residual<=kkt_tolerance and cerr<=constraint_tolerance:
            return SolverResult(field,True,"newton_stationary",tuple(history),residual,cerr,
                                len(ids),projector.gram_condition,lbfgs_done,iteration-1)
        grad,free=gradient(field,minimum_step=newton_box_active_step)
        # A larger box-active threshold is only a globalization device.  Once
        # it has solved that reduced problem, return to the audit free set so
        # convergence always means the established 1e-2 box-KKT condition.
        if np.max(np.abs(grad))<=kkt_tolerance and residual>kkt_tolerance:
            grad,free=audit_grad,audit_free
        damping=max(newton_minimum_damping,0.5*accepted_damping)
        accepted=False
        for _ in range(64):
            Af=projector.A[:,free]
            af_gram_inverse=np.linalg.inv(Af@Af.T)
            def matvec(v):
                tangent=np.zeros_like(v)
                vf=v[free]
                tangent[free]=vf-Af.T@(af_gram_inverse@(Af@vf))
                full=np.zeros_like(field);full.ravel()[ids]=tangent
                hv=exact_hessian_action(field,full,g)
                raw=(weight*hv/Wf).ravel()[ids]
                result=np.zeros_like(v)
                rf=raw[free]
                result[free]=rf-Af.T@(af_gram_inverse@(Af@rf))+damping*tangent[free]
                result[~free]=v[~free]
                return result
            operator=LinearOperator((len(ids),len(ids)),matvec=matvec,dtype=float)
            gradient_diagonal=(3.0*FrozenPhysics().gamma_s*g["config"].width/Wf)*(
                2.0/g["dr"]**2+2.0/g["dz"]**2)
            diagonal=(np.abs(weight.ravel()[ids]*(1-6*x+6*x*x+gradient_diagonal))
                      +damping+1e-3)
            def precondition(v):
                result=np.zeros_like(v)
                vf=v[free]/diagonal[free]
                result[free]=vf-Af.T@(af_gram_inverse@(Af@vf))
                result[~free]=v[~free]
                return result
            pre=LinearOperator(operator.shape,matvec=precondition,dtype=float)
            direction,info=gmres(operator,-grad,M=pre,rtol=1e-6,atol=0.,
                                 restart=40,maxiter=gmres_maxiter)
            direction[~free]=0.0
            df=direction[free]
            direction[free]=df-Af.T@(af_gram_inverse@(Af@df))
            slope=float(np.dot(grad,direction))
            # ``gmres`` reports a positive ``info`` when its requested linear
            # residual was not reached within the Krylov budget.  That does
            # not invalidate an inexact Newton direction.  Retain it when it
            # is finite and downhill; the bound-safe energy line search below
            # remains the nonlinear acceptance gate.  Rejecting every such
            # direction caused repeated explicit-gradient fallbacks and a
            # grid-scale two-cycle near stationary diffuse interfaces.
            if slope>=0 or not np.all(np.isfinite(direction)):
                damping*=2.;continue
            step=min(1.0,0.995*maximum_bound_step(x,direction))
            if step < newton_box_active_step:
                distance=np.where(direction < 0.0,x+2e-14,
                                  np.where(direction > 0.0,1.0+2e-14-x,np.inf))
                fraction=distance/np.maximum(np.abs(direction),1e-300)
                limiting=free & (fraction < newton_box_active_step)
                if np.any(limiting):
                    free[limiting]=False;grad[limiting]=0.0
                    continue
                damping*=2.;continue
            for _ in range(35):
                trial=field.copy();trial.ravel()[ids]=x+step*direction
                trial_energy=multigrain_energy(trial,phi,g)
                if trial_energy<energy:
                    accepted=True;break
                step*=0.5
            if accepted:break
            damping*=2.
        if not accepted:
            # A changing box-active set can exhaust Newton trials even though a
            # feasible descent direction still exists.  Take one projected
            # first-order globalization step, then retry Newton at the next
            # iteration; this is not a stationarity or scientific terminal.
            Af=projector.A[:,free];afgi=np.linalg.inv(Af@Af.T)
            gf=grad[free];pg=gf-Af.T@(afgi@(Af@gf))
            direction=np.zeros_like(grad);direction[free]=-pg
            slope=float(np.dot(grad,direction))
            step=min(1.0,0.995*maximum_bound_step(x,direction))
            for _ in range(50):
                trial=field.copy();trial.ravel()[ids]=x+step*direction
                trial_energy=multigrain_energy(trial,phi,g)
                if trial_energy<energy:
                    accepted=True;damping=float("nan");break
                step*=0.5
            if not accepted:
                return SolverResult(field,False,"globalization_no_feasible_descent",tuple(history),
                                    residual,cerr,len(ids),projector.gram_condition,lbfgs_done,iteration-1)
        if np.isfinite(damping): accepted_damping=damping
        field=trial;old_energy=energy;energy=trial_energy;newton_done=iteration
        record=dict(stage="newton",iteration=iteration,energy_J=energy,
            energy_drop_J=energy-old_energy,projected_kkt_residual=residual,
            normalized_constraint_residual=cerr,step=step,damping=damping,gmres_info=int(info))
        history.append(record)
        if callback is not None: callback(field, record)
    grad,_=gradient(field,minimum_step=1e-2)
    return SolverResult(field,False,"newton_iteration_limit",tuple(history),
        float(np.max(np.abs(grad))),constraint_error(field),len(ids),
        projector.gram_condition,lbfgs_done,newton_done)
