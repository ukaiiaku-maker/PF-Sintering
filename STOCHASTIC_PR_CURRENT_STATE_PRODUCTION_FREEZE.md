# Stochastic PR Current-State Production Freeze

Freeze date: 2026-09-09

This snapshot preserves the implementation used by the live eight-avalanche
stochastic Plateau-Rayleigh production campaign.  It is a source/provenance
freeze only; creating it did not stop, restart, or mutate the live process.

## Production entry points

- Live continuation: `scripts/pr_large_avalanche_resume13_alpha0p70_enospc.py`
- Current-state event adapter:
  `scripts/pr_large_avalanche_current_state_transfer_production.py`
- Current-state event implementation:
  `pf_sintering/production_mass_transfer_event.py`
- Conservative transfer kernel:
  `pf_sintering/current_state_mass_transfer.py`
- Renewal controller: `scripts/pr_avalanche_renewal_five.py`
- Read-only curvature watcher:
  `scripts/monitor_current_state_transfer_curvature.py`
- Source parent commit before this freeze:
  `493a2358d9226ed97284cd59bd07d7cd75f1932b`
- Earlier verified stochastic-production parent:
  `d2aafbf7df99728acb67af941f2c078126455e1f`

## Frozen physical and stochastic settings

- Temperature: `1830.15 K`
- Closed total-solid-volume boundary; external reservoir disabled
- Event displacement: `b = 2.5e-10 m` (`0.25 nm`)
- Root barrier: exact 1830.15 K EXP-floor slice plus additive pristine-root
  penalty `0.9803507282790153 eV`
- Root threshold multiplier: `1.25`
- Preserved OS seed:
  `299814244860662850753714406905245801947`
- Preserved avalanche-3 root threshold: `4.640741065761592`
- Initial descendant facilitation: `1.500000 eV`
- Descendant amplitude decay: `h[j+1] = 0.70 h[j]`
- Facilitated-source lifetime: `0.009 s`
- Descendant safety cap: `25`
- Grain-boundary diffusivity: `1.37908e-14 m^2/model-time`
- Surface Mullins coefficient: `3.375e-30 m^4/model-time`
- Physical clock mapping: `0.01557994316955921 s/model-time`
- Activation quantity: instantaneous measured-angle local geometric stress;
  integral stress retained as an independent diagnostic
- Completed-event fields remain native; equilibrated copies are diagnostic
  only and never feed back into production

The external barrier export is deliberately not copied into this repository:

- Path:
  `/Volumes/Data/Data/INRL_lambert_onsager/Forward_Model/data/zro2/bicrystal_creep_barrier_export.json`
- SHA-256:
  `fa7c9a2fb30e55596d9cdf47d2b03d530ef073cbe15f6210282285b2fcee7f37`
- 1830.15 K slice:
  `Gfloor=2.9476002588459194 eV`,
  `G0=5.3646516047156325 eV`,
  `sigmahat=1697232928.2808514 Pa`,
  `a=0.9682214719449614`, `n=0.46101864227254197`

## Frozen numerical and geometry settings

- Grid shape: `1320 x 261`
- Diffuse-interface width: `W = 1e-8 m`
- Axial/radial spacing:
  `dz = 1.2498591941394256e-9 m`,
  `dr = 1.2480209319616609e-9 m`
- Fourier reference: `R_cyl = 1e-7 m`,
  `lambda = 8.885765876316732e-7 m`
- Radial face after vacuum-only padding: `3.257334632419935e-7 m`
- Existing coordinates and field values were preserved exactly during padding
- Passive accepted macro interval: `1.0 model-time`
- Passive internal hazard/geometry interval: `0.025 model-time`
- Maximum passive numerical step: `8.086604871225672e-5 model-time`
- Root crossing remains rollback/bisection localized
- Sparse detached thermodynamic diagnostic cadence: `25 model-time`
- Current-state event continuation rebuilds donor/receiver masks from every
  committed native state and conserves transferred volume
- Legacy packet-event Courant value: `C4 = 0.05` when that explicitly selected
  integrator is used; it is not substituted for the current-state transfer
  propagator
- Solid-volume relative tolerance: `1e-7`
- Requested campaign duration: eight complete avalanches

## Runtime environment

- macOS `14.3.1`, arm64
- Python `3.13.2` (conda-forge, Clang 18.1.8)
- NumPy `2.5.1`
- SciPy `1.18.0`
- Numba `0.67.0`
- h5py `3.16.0`
- scikit-image `0.26.0`
- Matplotlib `3.11.1`
- pytest `9.1.1`

## Validation

The focused production regression suite passed at freeze time:

```text
83 passed, 5 subtests passed in 9.36 s
```

Pytest was invoked with `PYTHONPATH=tests/pytest_stubs` because the native
Python 3.13/libedit `readline` extension on this host segfaults during pytest's
stdio workaround.  The no-op stub is test-only; it is never on the production
runtime path.

Coverage included the closed-volume projector, corrected dynamics and energy,
continuous field TJ, current-state conservative transfer, integrator selection,
restart bookkeeping, avalanche source state, Courant threading, radial vacuum
padding, model-time transport/event logic, stress metrology, and movie archive.

## Intentionally excluded from this production commit

- Live and historical `runs/` outputs and checkpoints
- The external barrier-export JSON (identified above by path and SHA-256)
- User-provided `.docx` files, the `Avalanching/` reference folder, and the
  large MATLAB data file
- Historical milestone notes and rendered/post-processing products
- Inactive experimental geometry/TJ operators, including
  `pf_sintering/matched_outer_tj.py`
- Inactive parameter-study, calibration, audit, and superseded recovery scripts
- Unrelated dirty README, M16 multisink/movie, and recovery-driver edits

These exclusions preserve the working tree and historical research material
without implying that they are part of the live production call graph.
