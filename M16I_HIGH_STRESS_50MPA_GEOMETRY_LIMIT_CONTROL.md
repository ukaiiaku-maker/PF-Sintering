# PAUSED HIGH-STRESS HAZARD CONTROL — CURRENT GEOMETRY BUILDS ONLY ~2 MPa ON THIS HORIZON

Status classification (per M16J Section 2): **strongly decelerating / slowly increasing
stress tail** — NOT a mathematically proven asymptote. The run was paused, not concluded,
because M16J redirects effort toward a geometry search rather than continuing this
trajectory indefinitely.

This document freezes the exact state of the M16I barrier-ladder correction's 50 MPa-target
FINITE run (and its co-running ZERO companion) at the moment both were safely paused to
begin the M16J geometry-driven stress-concentration search. Both trajectories remain valid,
resumable, reportable results in their own right — the crest-capped one-contact geometry
(`R_cyl=100nm`, `psi=160°`) demonstrably does **not** build stress past ~2 MPa within
t=[0,157.5] under a 50 MPa hazard target, which is the primary motivating observation for
M16J's geometry search.

## 1. Provenance

| field | value |
|---|---|
| git branch (at pause time) | `codex/coarsening-stress-buildup` |
| git HEAD (at pause time) | `6a6ac9733efdddfb6d660311ba31472a573aa53e` |
| exact command (finite, 50 MPa) | `../.venv/bin/python scripts/m16i_three_barrier_campaign.py --campaign-root runs/m16i_three_barrier_highstress --regime finite --t-target 250 --live-update-wall-s 60 --live-update-dt 1.0 --n-snapshots 20 --sigma-target-hazard-MPa 50 --v0-mult 1000` |
| exact command (zero, companion) | `../.venv/bin/python scripts/m16i_three_barrier_campaign.py --campaign-root runs/m16i_three_barrier_prod --regime zero --t-target 250 --live-update-wall-s 60 --live-update-dt 1.0 --n-snapshots 20` |
| PID (finite) / (zero) | 98755 / 98753 |
| stop signal | `SIGTERM` (not `SIGKILL`) sent to both, after each process's on-disk checkpoint was confirmed valid and independently reloadable |
| initial-state hash (`array_hash` over f,e1,e2,z) | `f626593f0f06c17e71c5f4c0913dc82802ea4f6740815747c19121b2d3115cc5` — matches the hash recorded for `runs/m16i_three_barrier_prod/initial_state.npz` in the M16I report, confirming both regimes started from the identical canonical geometry |

## 2. Checkpoint integrity (verified by direct reload)

| regime | checkpoint step | checkpoint t | f/e1/e2 shape | all-finite | sink_state present | hp_state present | rng_state present |
|---|---|---|---|---|---|---|---|
| finite (50 MPa) | 3,228,000 | 157.617 | (414, 185) | yes | yes | yes | yes |
| zero | 3,048,000 | 148.848 | (414, 185) | yes | yes | yes | yes |

Both `checkpoints/state_checkpoint.npz` files reload without error via `np.load`, both
`meta.json` files parse and contain the full expected key set
(`a0, Vp0, Vs0, V0, com0, z_gb_prev, sink_state, hp_state, rng_state, barrier_eV,
pre_event_state`). Both are safely resumable later with
`scripts/m16i_three_barrier_campaign.py --campaign-root <root> --regime <finite|zero>
--t-target <T>` if this control trajectory needs to be extended in the future.

## 3. Latest diagnostic sample (finite, 50 MPa) — step 3,225,554, t=157.498

(The last *appended history row*, one diagnostic sample behind the checkpoint step itself,
which is normal — diagnostics are sampled less often than checkpoints.)

| quantity | value |
|---|---|
| `sigma_target_hazard` | 50 MPa |
| `A0` | 5.3497e-19 J = 3.339 eV |
| `V0` | 1.0e-26 m³ = 1000·Ω |
| `T` | 1000 K |
| `gamma_s` | 1.0 (normalized) |
| `gamma_gb` | 0.34730 |
| `W` | 10 nm |
| `dx` | 1.25 nm |
| `R_cyl` (particle geometry) | 100 nm |
| `psi` (dihedral construction angle) | 160° |
| `n_events` | 0 |
| `sigma_s` (Hussein Eq. 1b, 1.5W window) | 2.0569 MPa |
| `sigma_curvature` | 12.654 MPa |
| `sigma_width` | -10.597 MPa |
| `r_neck` (1.5W) | 79.024 nm |
| `X_neck` | 92.929 nm |
| `Q` (=X_neck/(C_GB·r_neck)) | 1.1941 (**positive**, consistent with `sigma_s>0`; see Section 6 audit below) |
| `hazard` (instantaneous rate) | 2.386e-11 s⁻¹ (negligible) |
| `mass_drift` | 1.58e-13 (negligible) |

For reference, the companion ZERO run reached t=148.85 with `sigma_s≈0.004 MPa`,
`a/a0=1.024`, `mass_drift=5.9e-4` (well under the 0.2% gate), still slowly densifying with
the sink continuously active — consistent with its whole history.

## 4. Interpretation

Over t=[0,157.5] — comfortably longer than the baseline (2.5 MPa target) case's first
event at t=22.13 — the 50 MPa-target trajectory shows σ_s rising from ~0.85 MPa (t=0) to
~2.06 MPa (t=157.5), with the *rate* of increase itself continuously decelerating
(confirmed by hourly monitoring: Δσ_s per unit Δt shrank from ~0.0014/t-unit near t=100 to
well under that by t=157). This is the "strongly decelerating / slowly increasing stress
tail" referenced above — not yet proven to be a hard mathematical asymptote (that would
require the extrapolation-with-holdout-validation machinery M16J introduces), but
clearly incompatible with reaching 50 MPa on any practically reachable horizon for *this*
particular geometry (`R_global~100nm`, `r_neck~80nm`, i.e. essentially no separation
between global and local curvature scales).

This is exactly the observation motivating M16J's central hypothesis: real stress
concentration requires `R_particle >> r_neck`, which this geometry does not have by
construction.

## 5. Code state at pause time

`git status --short` at pause time showed only one untracked file beyond the already-
committed M16I work: `scripts/m16i_highstress_finite_figures.py` (the dedicated 50 MPa
evolution-plot generator added after the M16I report was written, per user request). This
was committed as part of code preservation before M16J geometry work began (see the M16J
report's code-preservation section for the exact commit).

## 6. Q-diagnostic audit (M16J Section 30)

The concern raised was that the live dashboard "appears to plot Q with a negative sign."
Audited directly:

- **Formula** (`scripts/m16i_three_barrier_campaign.py`, `diagnostics_row`):
  `row["Q_1p5W"] = X_neck / (C_GB * w["r_neck"])`, guarded by
  `if np.isfinite(C_GB) and C_GB > 0`. `X_neck` and `r_neck` are both geometric magnitudes
  and are positive by construction (`X_neck=2*a_contact`, `r_neck=abs(r_neck)` inside
  `hussein_eq1b_sigma`); `C_GB=sqrt(1-(gamma_gb/(2*gamma_s))^2) ∈ (0,1]` is a square root,
  never negative. **The stored `Q_1p5W` value is therefore always non-negative by
  construction** — confirmed numerically above (`Q=1.1941` at the latest sample; spot-
  checked several earlier rows, all positive).
- **Dashboard rendering** (`pf_sintering/live_dashboard.py`,
  `render_dashboard_figure`): `Q_1p5W` is plotted, unmodified, on a `twinx()` axis
  (`ax_q2`) sharing a subplot with `sigma_curvature`/`sigma_width` (the latter is
  genuinely negative, ~-10 to -11 MPa). No sign flip exists in the plotting code. The
  likely source of the "appears negative" impression is that `ax_q2`'s independent,
  auto-scaled right-hand axis was not given an explicit floor, so the dotted Q line could
  visually sit low in the shared plot area near where the (unrelated, left-axis) negative
  `sigma_width` curve also sits — a **display-clarity artifact, not a data or sign bug**.
- **Fix applied** (data unchanged): `ax_q2` now gets an explicit y-lower-bound of 0 and a
  dashed horizontal reference line at `Q=1` (the `sigma_s=0` boundary), so the panel can no
  longer be visually misread as showing negative Q. See the M16J report's code-
  preservation section for the exact diff and confirmation this only touches rendering.
