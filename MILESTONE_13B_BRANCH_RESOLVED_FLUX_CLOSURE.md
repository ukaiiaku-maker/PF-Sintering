# Milestone 13B: branch-resolved surface-flux closure

## 0. Starting checkpoint

Branch `codex/coarsening-stress-buildup`, HEAD `af349f8` ("docs+feat: Milestone 13 pure surface-diffusion vs GB-migration campaigns and report"), confirmed clean working tree and **190/190** tests passing before any change. The sinusoidal geometry (R2=80nm, wavelength=480nm, amplitude=24nm, overlap=20nm, W=20nm, surface-mobility-scale=0.3) was never changed. Isotropic surface energy, eta frozen for the primary experiment, sink/hazard/RBM off were all preserved throughout. Full suite at the end of this milestone: **194/194** (190 inherited + 4 new in `tests/test_branch_flux.py`).

New module: `pf_sintering/branch_flux.py`. New scripts: `scripts/m13b_branch_flux_closure.py`, `scripts/m13b_active_tol_sensitivity.py`.

## 1. The disagreement, restated precisely

Milestone 13's two flux diagnostics were never actually measuring the same thing, and the apparent contradiction dissolves once that is made explicit:

- `curvature_extraction.branch_mu_J_profile`'s point samples of `J_tangent` are a **local flux-density value at one point on the contour**, not yet an integrated rate.
- `flux_closure.neck_boundary_face_flux_balance`'s particle/substrate split classified each Cartesian boundary **face** of the neck control volume by **absolute position** (X-deviation from a far-field substrate baseline) -- Section 2's explicit instruction was to stop treating that classification as the physical branch decomposition, and this investigation shows why: the two points where the free surface actually crosses that boundary can both fall on the same side of a position threshold calibrated against a distant baseline, silently misattributing one branch's real flux to the other's bucket. This was never a sign of an inconsistent underlying flux field.

## 2. Branch-cut construction (Section 3) and normal-integration convergence (Section 4)

`branch_flux.branch_cut_flux` identifies each branch by **identity** (reusing `curvature_extraction._walk_branch`'s validated local circle-marching walker, not position), locates the point and local tangent at fixed arclength `s_cut` from the TJ, and integrates `J . t_hat` across a line normal to that tangent, at a sequence of half-widths, to check convergence explicitly.

**Validation on a controlled synthetic case** (straight interface, uniform flux field, `tests/test_branch_flux.py`): the cut integral reproduces the analytic `2*halfwidth*flux_magnitude` to `<1e-9` relative at every half-width tested (0.5W, 1W, 1.5W, 2W, 3W) -- the integration machinery itself is exact; any discrepancy found below on the real geometry is physical, not a bug in this construction.

## 3. The central finding: TJ-core dominance (Sections 5-6)

Building the naive two-branch model (`Q_p` = flux from the particle branch into the TJ, `Q_s` = flux from the TJ out along the substrate branch, `dM/dt ~ Q_p - Q_s`) and testing it against the exact three-way closure:

- **A** (direct finite difference of the disk's own f-mass over one step) and **B** (`branch_flux.cartesian_total_boundary_flux`, the exact Cartesian boundary-face flux for the same disk) agree **to machine precision at every state tested** (confirmed via `bc_ops.flux_divergence`'s own exact discrete divergence theorem -- this was never in question).
- **C** (the naive two-branch-cut estimate, `Q_p - Q_s` at `s_cut=2W`) **does not match A/B** at early times: `C/B` ratio is `-0.14` at t=0, `-0.41` at t=0.015s, `-1.69` at t=0.06s (wrong sign, wrong magnitude).

Per Section 6's explicit instruction ("if C does not close, identify the missing physical branch/channel rather than forcing the two-cut interpretation"), a radius-scan of `B` at the TJ (`0.25W, 0.5W, 0.75W, 1W, 1.5W, 2W, 3W`) was performed:

| radius | `B` (top TJ, t=0.015s) |
|---|---|
| 0.25W | 3.47e-17 |
| 0.5W | 6.94e-17 |
| 1.0W | 7.72e-18 |
| 1.5W | -6.17e-19 |
| 2.0W | 3.13e-17 |
| 3.0W | 3.11e-17 |

**The dominant flux-accumulation signal is already present within 0.25-0.5W of the TJ itself** -- comparable in magnitude to the total at 2-3W, not built up gradually along the branches from further away. Confirmed at both TJs (symmetric to <2%) and at t=0.06s independently (same pattern). The missing channel is not a third 1D branch: it is a genuinely 2D, spatially concentrated **TJ-core** region (within roughly 1W) where the diffuse profiles of the particle branch, substrate branch, and internal GB overlap and interact -- `J_normal` (the flux component transverse to the local branch tangent, which the 1D-channel picture assumes is zero) is measured at **87% of `|J_tangent|` right at the TJ**, decaying to `<1%` by `s~15nm` (`tests` confirm this via `branch_mu_J_profile`'s existing `J_normal` output) -- i.e. the free surface only becomes well-approximated as two separate 1D tangential flux channels a few nm beyond the TJ itself, and the two-cut model's entire premise (a clean point-TJ joining two independent 1D channels) is not accurate within that inner region.

## 4. Closure improves over time -- a genuine physical trend

Tracking `C/B` at `s_cut=2W` across the full frozen-eta trajectory (both TJs agree to <2%):

| t (s) | C/B (top TJ) | C/B (bottom TJ) |
|---|---|---|
| 0 | -0.142 | -0.146 |
| 0.015 | -0.414 | -0.424 |
| 0.03 | -0.275 | -0.286 |
| 0.06 | -1.691 | -1.553 |
| 0.10 | 0.081 | 0.079 |
| 0.15 | 0.538 | 0.540 |
| 0.20 | 0.711 | 0.715 |
| 0.25 | 0.807 | 0.810 |
| 0.30 | 0.921 | 0.931 |

The two-branch-cut model is a poor approximation early in the trajectory (large, sign-flipping error) but becomes an increasingly good one as the neck widens (agreement to ~8% by t=0.3s). This is physically sensible: Milestone 13 found `L_contact_TJ_sub` grows substantially over this window (widening from its initial near-contact value) -- as the neck widens, the local TJ geometry smooths out and the "TJ-core complexity zone" shrinks relative to the fixed `s_cut=2W` scale, so the two independent-1D-channel approximation becomes progressively more valid.

**This improvement is not yet reliable enough to trust at a fixed `s_cut`, however**: at the best-agreeing time (t=0.3s), scanning `s_cut` itself (Section 3's explicit robustness check) shows the closure ratio is still highly sensitive to cut location -- `1.225` at 1.5W, `0.921` at 2.0W, `0.274` at 2.5W, `-4.231` at 3.0W (sign flip again). The apparent convergence at `s_cut=2W` specifically should not be over-interpreted as the two-cut model having become generally valid -- it has not, at any tested cut location, converged robustly across both time AND cut-location simultaneously.

## 5. Q_p(t), Q_s(t), R_flux(t) (Section 7) -- reported, with an explicit reliability caveat

At `s_cut=2W`, `halfwidth=1W` (top TJ):

| t (s) | Q_p | Q_s | R_flux = Q_s/Q_p |
|---|---|---|---|
| 0 | -1.275e-17 | -1.517e-18 | 0.119 |
| 0.015 | -1.163e-17 | +1.320e-18 | -0.113 |
| 0.03 | -1.324e-17 | -8.677e-18 | 0.656 |
| 0.06 | -1.424e-17 | -6.190e-18 | 0.435 |
| 0.10 | -1.056e-17 | -1.135e-17 | 1.074 |
| 0.15 | -7.651e-18 | -1.461e-17 | 1.909 |
| 0.20 | -6.181e-18 | -1.634e-17 | 2.644 |
| 0.25 | -5.496e-18 | -1.733e-17 | 3.153 |
| 0.30 | -5.261e-18 | -1.954e-17 | 3.714 |

Taken at face value against the interpretation guide in the handoff (`R_flux<1`: filling, `>1`: depletion), `R_flux` crosses 1 around t~0.1s and rises to nearly 4 by t=0.3s -- which would suggest the neck transitions to net depletion partway through the trajectory.

**This is contradicted directly by the exact ground truth.** `A` (and `B`, identical by construction) -- the actual, exact mass-balance rate for the same disk -- is **positive at every single sampled time, 0 through 0.3s, with no exception** (`7.90e-17, 3.13e-17, 1.66e-17, 4.76e-18, 9.71e-18, 1.29e-17, 1.43e-17, 1.47e-17, 1.55e-17`, top TJ) -- the neck-adjacent region is gaining mass throughout, never losing it, fully consistent with Milestone 13's own `dM_neck_f/dt>0` finding over the same window. `R_flux`'s apparent crossing of 1 is an artifact of the ratio's sensitivity to compensating errors in the individual `Q_p`/`Q_s` estimates (Section 4's `C=Q_p-Q_s` difference is far better behaved than either term or their ratio alone) -- **`R_flux` must never be read as a standalone morphological indicator; it is reported here exactly as requested, but its crossing of 1 does not correspond to any real transition in the physical neck-filling behavior.**

## 6. Three-way neck-mass closure (Section 6, summary)

A and B agree exactly at every state and every control-volume definition tested in this milestone (an algebraic identity, not a physical claim). C (the two-branch model) does not close reliably at any fixed `s_cut`/time combination tested, though it improves substantially in the net (not the ratio) over the trajectory. **Outcome: B4** -- see Section 9.

## 7. Geometric particle-lobe control volume (Sections 8-9)

Per Section 9's explicit instruction, a particle-side region was defined directly from **the instantaneous TJ pair's own shared neck column** (`X > x_neck`, `x_neck = 0.5*(x_TJ_top + x_TJ_bottom)`) -- in this geometry the two TJs share essentially one X coordinate, so this is simply the half-plane on the particle side of the neck's own vertical cross-section. Its flux closure is unambiguous (a single internal boundary line; `cartesian_total_boundary_flux` applies directly) and closes to **machine precision at every sampled time** (`closure_residual ~1e-25` against values `~1e-16` to `1e-15`, i.e. `<1e-9` relative).

`M_particle_geom(t)`: `2.0367e-14` (t=0) -> dips to `~2.013-2.014e-14` and plateaus there from t=0.06s through t=0.25s -> drops further to `1.9925e-14` by t=0.3s. **Net change over the full window: -2.2%, a genuine decrease, not a plateau or recovery.**

This is a materially different verdict from Milestone 13's two other particle-size proxies on the exact same trajectory: `V2` (eta-weighted) showed net **growth**; `A_particle_geom` (a cruder X-deviation-from-substrate-baseline threshold) showed **non-monotonic** rise-fall-rise behavior. This `x_neck`-anchored, exactly-closing metric is both more directly tied to the physical TJ/GB connection (Section 9's request) and shows the cleanest signal of the three: **the particle lobe, defined this way, does genuinely lose conserved f-mass over this window** -- satisfying Section 8's explicit bar ("do not call it particle shrinkage unless an actual particle-lobe f-mass metric decreases") for the first time in this line of investigation.

## 8. Active-set tolerance sensitivity (Section 10)

`scripts/m13b_active_tol_sensitivity.py`: same State-A contact, full production `M_eta` (eta-active), dx=2.5nm, t=0.02s, at `active_tol = 1e-5, 1e-4, 1e-3` (a 100x span). Results at the common end time:

| `active_tol` | `V2` | `M_neck_f` | rel. diff vs. `1e-5` (`V2`) | rel. diff vs. `1e-5` (`M_neck_f`) |
|---|---|---|---|---|
| 1e-5 | 2.0359089927e-14 | 3.9394278503e-15 | -- | -- |
| 1e-4 | 2.0359852954e-14 | 3.9394286681e-15 | 3.75e-5 | 2.08e-7 |
| 1e-3 | 2.0363658419e-14 | 3.9394364589e-15 | 2.24e-4 | 2.19e-6 |

TJ position agrees to 9 significant figures across all three tolerances. Cumulative `safety_fraction` (the genuine numerical-overshoot diagnostic, Milestone 13 Section 11) stays at true roundoff (`3.6e-11` to `4.1e-11`) at every tolerance -- it does not grow even at the loosest tested value. **Physical observables are insensitive to `active_tol` over two orders of magnitude; it is confirmed to be a purely numerical active-set tolerance, not a hidden physical parameter**, and the existing default (`1e-4`, chosen in Milestone 13 for the same roundoff-negligibility property) is retained without change.

## 9. B1/B2/B3/B4 classification

**B4.** The branch-cut balance does not reproduce the total `dM_neck/dt` reliably: it is qualitatively wrong (wrong sign, large magnitude error) early in the trajectory and only partially converges (to ~92% agreement in the net, at one specific cut location) by the latest time tested, while remaining highly sensitive to the exact cut location even at that best-case time. Per Section 6/11's explicit instruction, the missing channel was identified rather than forcing the two-cut interpretation: **it is the TJ core itself** (within ~1W of the TJ), a genuinely 2D region where the two branches' diffuse profiles overlap and `J_normal` is not negligible (87% of `J_tangent` right at the TJ), not a separate third 1D branch that could simply be added to the model.

This does **not** overturn Milestone 13's central finding that `dM_neck_f/dt>0` throughout 0-0.3s -- that finding rests entirely on the exact A/B calculation, which was never in question and is reconfirmed here at every sampled time. What B4 does overturn is any attempt to attribute that neck-filling behavior to a clean "particle donates, substrate drains" two-channel picture, or to read `R_flux` crossing 1 as evidence of an emerging depletion regime -- neither is supported once the closure is checked rather than assumed.

**A genuine, independently well-supported finding survives this investigation intact**: the `x_neck`-anchored geometric particle-lobe control volume (Section 7 above), whose closure is exact (unlike the TJ two-cut model), shows the particle lobe's own conserved f-mass decreasing by ~2.2% over the same window -- a real, if modest, particle-lobe mass loss, obtained independently of the TJ branch-cut controversy.

## 10. Recommendation for next steps

1. **Do not rely on the two-branch-cut model (`Q_p`, `Q_s`, `R_flux`) for morphological interpretation** at any cut location tested; use the exact `A`/`B` Cartesian control-volume balance (already established as reliable in Milestones 12B/13/13B) for any future neck-mass-balance question.
2. **The `x_neck`-anchored geometric particle-lobe metric is the most trustworthy particle-size proxy identified across Milestones 13/13B** (exact flux closure, directly tied to the physical TJ/GB connection, and the only one of the three tested metrics to show a clean, monotonic-trending net decrease) -- prefer it over `V2` or the X-deviation-threshold `A_particle_geom` in future work.
3. If a branch-resolved (not just total) neck flux accounting is still wanted, the TJ-core region itself needs its own explicit treatment (e.g. a small inner disk excised and accounted separately, with the two branches modeled only outside it) rather than extending `s_cut` further out, which Section 4 above shows does not resolve the issue and can make it worse.
4. Per the handoff's explicit terminal instruction, no geometry change and no sink/hazard/RBM activation follows from this milestone -- this is a diagnostics-correction milestone, and its scientific content is the closure investigation itself, not a basis for the next morphology campaign.

---

### Appendix: raw run artifacts

- `runs/m13b_branch_flux_closure_dx2p5.json` -- full 9-timepoint campaign (TJ-core radius scans, three-way closure at 4 cut locations, `Q_p`/`Q_s`/`R_flux`, particle-lobe mass and closure).
- `runs/m13b_active_tol_sensitivity_dx2p5.json` -- active-set tolerance sweep.

### Stop conditions honored

Geometry (R2, wavelength, amplitude, overlap, W, surface-mobility-scale) was never changed. Isotropic surface energy, eta frozen for the primary experiment, sink/hazard/RBM off were preserved throughout every run in this milestone. `pf_sintering/branch_flux.py` is new diagnostic-only code, not wired into any production default path.
