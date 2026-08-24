# Stochastic PR production baseline

This snapshot freezes the verified ten-realization, five-event-per-realization
production implementation.  The exact commit SHA is written into the runtime
campaign manifest through `PRODUCTION_BASELINE_COMMIT` at launch.

## Frozen production settings

- Temperature: 1830.15 K.
- Barrier: exact exported EXP-floor temperature slice; no refit.
- Barrier export SHA-256:
  `fa7c9a2fb30e55596d9cdf47d2b03d530ef073cbe15f6210282285b2fcee7f37`.
- Event Burgers-vector quota: `b_event = 0.25 nm`.
- Exported creep-fit Burgers vector, provenance only: `b_fit = 0.36 nm`.
- Site population: `N_sites = 2*pi*r_TJ/b_event`.
- Physical time map: 0.01557994316955921 s/model time.
- Coarsening reservoir: neck unprotected, `tau_coarsen/tau_surf = 20`.
- Activation stress: local geometric 3-D contact-force stress.
- Event physics: original full-branch, zero-storage TJ node.
- Explicit fourth-order Courant: `C4 = 0.05`, passed explicitly from each
  worker and asserted against the event integrator's returned diagnostic.
- Censoring envelope: `V_p/V_p,cycle >= 0.88` and `r_n >= 35 nm`.
- If hazard and a validity boundary cross within the same 0.25 analysis
  block, their crossing order is estimated and the preserved valid block-start
  state is replayed at 0.005 model-time analysis cadence.  The PF timestep and
  equations are unchanged; this prevents a first-passage event that occurred
  first from being mislabeled as right-censored at the later saved endpoint.
- Ensemble: ten independent, recorded OS-derived seeds; at most five complete
  one-b events per realization.
- `matched_outer_tj.py` is diagnostic only and is not imported or selected by
  the production driver.

## Geometry and grid

- Fourier construction: `R_cyl = 100 nm`, `epsilon_1 = 0.4`,
  `epsilon_2 = 0`, `lambda/R_cyl = sqrt(2)*pi`.
- Diffuse width: `W = 10 nm`.
- Nominal spacing: 1.25 nm.
- Realized grid: 984 x 157; `dz = 1.2492527893242913 nm`,
  `dr = 1.2478766271735947 nm`.
- Model timestep: 4.8828125e-5.

## Verification

The explicit-call smoke check used a preserved valid pre-nucleation state and
completed `q/b = 1e-4` in one accepted step.  The event integrator returned
`explicit_max_fourth_order_courant = 0.05`, which passed exact equality.

The launch manifest records the full `pip freeze --all`, Python executable,
Python version, platform, barrier slice, external export hash, settings, and
seed provenance.  Each worker separately records its realized grid, barrier,
event Burgers vector, and received Courant value.
