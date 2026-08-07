# PF-Sintering

Python implementation of the phase-field sintering model originally developed from the MATLAB v64 code base.

## Current branch

Development is on `python-v64-port`. The current target is the particle-on-substrate model and the short-contact activation-stress campaign.

The implementation currently includes the conserved Cahn-Hilliard solid field `f`, structural order parameters, anisotropic capillarity in the CH evolution, explicit substrate Ostwald transfer, integrated-hazard sink activation, rigid-body densification/advection, dynamic GB excess energy, diagnostics, frame output, and restartable HDF5 checkpoints.

The particle-on-substrate path is the current production target. A three-particle initializer is present, but the complete two-GB kinetics and two-sink bookkeeping still need to be completed before that topology should be treated as production-ready.

## Physics contract

The Python model is not required to reproduce MATLAB numerically. MATLAB is provenance and a diagnostic reference; correctness is defined by the physical decomposition below.

- `f` is the conserved solid-mass field. CH and RBM should conserve its integral to numerical tolerance.
- CH changes solid geometry but does not change integrated grain ownership.
- Structural `eta_i` relaxation is numerical/structural regularization only. It must not produce secular grain dissolution.
- Explicit Ostwald transfer is the intended secular grain-volume redistribution mechanism in a no-event run.
- RBM translates/deforms grain ownership but should not spuriously change the integrated amount of a grain.
- Hazard events alter geometry only through their explicitly defined event mechanics.

To enforce this separation, the current production integrator uses a mass-preserving constrained `eta` projection after CH, structural relaxation, and RBM. The projection enforces `eta_i >= 0` and `sum(eta_i) <= f` while redistributing locally clipped grain ownership into available solid capacity rather than discarding it.

The operator ledger `scripts/diagnose_v2_operator_ledger.py` is the primary regression diagnostic. In a no-event run, the expected result is:

```text
post_ch_constrained      ~ 0 net grain-volume change
ostwald                  ~= prescribed Ostwald rate
ac_raw + constrained     ~ 0 net grain-volume change
rbm_raw + constrained    ~ 0 net grain-volume change
NON-OSTWALD NET          ~ 0
```

## Install

From the shared working root used in development:

```bash
cd /Volumes/Data/Data/PF-sintering
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./PF-Sintering pytest
```

## Small development runs

The default `dev` preset deliberately uses a much smaller particle/grid problem and a short time window. It is useful for implementation tests and is **not** a replacement physical calibration for v64.

```bash
pf-sintering --preset dev --time-ms 0.1 --no-event-prints
```

An explicitly reduced grid can be requested:

```bash
pf-sintering --preset dev \
  --nx 96 --ny 128 \
  --dx-nm 5 \
  --r2-nm 80 \
  --time-ms 0.2 \
  --output-dir runs/small_test \
  --out-tag small_test
```

## Geometry and system-size flags

Geometry and grid dimensions are runtime parameters rather than hard-coded constants. Important controls include:

- `--nx`, `--ny`: explicit number of grid cells. If omitted, the domain is derived from the particle dimensions.
- `--dx-nm`: grid spacing.
- `--r1-nm`, `--r2-nm`, `--r3-nm`: reference particle radii.
- `--interface-cells`: diffuse-interface width measured in grid cells.
- `--domain-margin-r2`: automatic-domain margin measured in units of `R2`.
- `--initial-overlap-nm`: initial geometric overlap.
- `--wall-frac`: substrate-wall position as a fraction of the x-domain.
- `--aspect-ratio`: elongated-particle aspect ratio.
- `--contact-orientation short_plane|long_plane`: chooses the short or long wall-parallel contact plane.
- `--geometry substrate|threeparticle`: topology selector; substrate is the currently qualified development path.

For example, an automatically sized geometry with altered particle dimensions is:

```bash
pf-sintering --preset dev \
  --dx-nm 4 \
  --r1-nm 120 --r2-nm 90 --r3-nm 120 \
  --aspect-ratio 2.5 \
  --contact-orientation short_plane \
  --time-ms 0.2
```

## Full v64 dimensional preset

The v64 dimensional defaults remain available as a physical-scale preset:

```bash
pf-sintering --preset v64 \
  --sigma-target-mpa 75 \
  --aspect-ratio 2 \
  --contact-orientation short_plane \
  --time-ms 2500 \
  --output-dir runs/v64_75MPa \
  --out-tag v64_75MPa
```

Any geometry flag can override an individual v64 preset value.

## Activation-stress sweep

The short-contact sweep has a Python launcher:

```bash
python PF-Sintering/scripts/run_short_activation_sweep.py \
  --preset dev \
  --sigmas-mpa 0.1 75 150 \
  --time-ms 0.2
```

A deliberately small sweep can be launched with:

```bash
python PF-Sintering/scripts/run_short_activation_sweep.py \
  --preset dev \
  --nx 96 --ny 128 --dx-nm 5 --r2-nm 80 \
  --sigmas-mpa 0.1 75 150 \
  --time-ms 0.2
```

## Outputs

Each run writes:

- compressed phase-field frames as `.npz`,
- a diagnostic time series as `.csv`,
- a run summary as `.json`,
- periodic HDF5 restart checkpoints, and
- a restartable final HDF5 state.

`latest_restart.txt` points to the newest restart state.

Fresh runs fail closed if a final output with the same tag already exists unless `--overwrite` is explicitly requested.

## Tests

```bash
cd /Volumes/Data/Data/PF-sintering
python -m pytest ./PF-Sintering/tests -q
```

The current suite checks configurable grid sizing, field initialization, the 9-point Laplacian, corrected low-level kernel wiring, completion of a tiny integration run, and the mass-preserving structural-projection contract.
