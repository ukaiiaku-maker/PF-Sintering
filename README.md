# PF-Sintering

Python implementation of the phase-field sintering model currently developed in MATLAB.

## Current branch

Development is on `python-v64-port`. The first target is the v64 particle-on-substrate model and the short-contact activation-stress campaign.

The implementation currently includes the conserved Cahn-Hilliard solid field `f`, structural order parameters, anisotropic capillarity in the CH evolution, explicit substrate Ostwald transfer, integrated-hazard sink activation, rigid-body densification/advection, dynamic GB excess energy, diagnostics, frame output, and restartable HDF5 checkpoints.

The particle-on-substrate path is the current production target. A three-particle initializer is present, but the complete two-GB kinetics and two-sink bookkeeping still need to be ported before that topology should be treated as production-ready. The MATLAB and Python random streams are intentionally not required to match event-for-event.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Small development runs

The default `dev` preset deliberately uses a much smaller particle/grid problem and a short time window. It is for implementation tests and is **not** a replacement physical calibration for v64.

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

The uploaded MATLAB v64 dimensional defaults are available without making them the development default:

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

The MATLAB short-contact sweep has a Python launcher:

```bash
python scripts/run_short_activation_sweep.py \
  --preset dev \
  --sigmas-mpa 0.1 75 150 \
  --time-ms 0.2
```

A deliberately small sweep can be launched with:

```bash
python scripts/run_short_activation_sweep.py \
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

## Tests

```bash
pytest -q
```

The initial smoke suite checks configurable grid sizing, field initialization, the 9-point Laplacian, and completion of a tiny integration run.
