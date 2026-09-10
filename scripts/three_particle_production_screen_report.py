"""Assemble viability evidence without promoting a rejected geometry."""
from pathlib import Path
import json,hashlib,math
import numpy as np
D=Path('docs/three_particle/production_screen')

def main():
    screen=json.loads((D/'screen.json').read_text());rows=screen['candidates']
    for row in rows:
        if 'final' not in row:continue
        row['derivatives']={name:{key:(float(row['final'][name][key])-float(value))/row['short_time_s']
            for key,value in row['initial'][name].items() if isinstance(value,(float,int)) and not isinstance(value,bool)} for name in ['LEFT','RIGHT']}
        with np.load(row['checkpoint']['path']) as d:
            f=d['f_final'];z=d['z'];r=d['r_c'];gb=d['gb'];dr=r[1]-r[0];W=4e-9
            core=(z>gb[0]+2*W)&(z<gb[1]-2*W)
            minimum=float(np.min(np.sqrt(2*np.sum(f[core]*r[None,:],axis=1)*dr)))
            row['final_resolution_margins']=dict(center_span_over_W=float((gb[1]-gb[0])/W),
                contact_separation_margin_W=float((gb[1]-gb[0])/W-8),center_interior_radius_over_W=minimum/W,
                pinch_radius_margin_W=minimum/W-3,f_min=float(f.min()),f_max=float(f.max()))
    (D/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    history=json.loads((D/'preserved_unequal_history.json').read_text())
    continuation=json.loads((D/'continuation_0.74_81.850.json').read_text())
    accepted=[r for r in rows if r['status']=='SCREENED'];first=history['initial'];last=history['final'];late=history['late_loading_start'];end=continuation['history'][-1]
    table=['| Rc/Ro | Ro (nm) | Lc/W | Cleanup | LEFT local (MPa) | LEFT root rate (s⁻¹) | Mean wait/contact (s) |',
           '|---:|---:|---:|:---|---:|---:|---:|']
    for r in rows:
        c=r.get('initial',{}).get('LEFT')
        if c:table.append(f"| {r['ratio']:.2f} | {r['outer_radius_nm']:.3f} | {r['cmc']['center_span_over_W']:.3f} | {'pass' if r['status']=='SCREENED' else 'reject'} | {c['sigma_local_Pa']/1e6:.4f} | {c['root_rate_per_s']:.5f} | {c['constant_state_mean_wait_s']:.2f} |")
    text=f'''# Shared bicrystal-physics viability result

**The stationary null is no longer a gate. No event-producing geometry is yet
qualified by this screen; stochastic three-particle production remains disabled.**
The original unequal 0.70 case, physics, field guard and live production are
unchanged. No clipping, fitted correction, barrier tuning or null subtraction.

## What is implemented and verified

Both contacts now call the actual bicrystal stress formulas, retaining all four
one-sided local values and all four continuous-integral values. Each contact
passes the existing within-contact local stress to the unchanged live root
barrier/rate. The actual PF chemical-potential affinity and one-b transport work
are independent outputs. See [exact call-path audit](PHYSICS_AUDIT.md) and the
frozen [live launch manifest](bicrystal_launch_manifest.json).

The absent-third-grain field regression matches both bicrystal stress families
and their one-sided values exactly. Root-rate parity is tested over negative,
zero and positive stresses with the live temperature, barrier and prefactor.
The focused suite passes **37 tests**. Production source files were not edited.

## Resolved geometry screen

{len(rows)} candidates were evaluated at W=4 nm and dr=0.5 nm, using aligned dz,
the same mapped CMC construction, 200 blocked-GB cleanup steps and 1000 released
native steps. **{len(accepted)} pass the existing cleanup/topology checks.**
The remainder are retained as rejected evidence; their rates cannot justify
selecting them. The largest two geometries deliberately test the cost of
resolving a very small center radius ratio.

'''+ '\n'.join(table)+f'''

LEFT/RIGHT full quantities, derivatives of every scalar stress/work/rate and
resolution margins are in [screen.json](screen.json). Symmetric local stresses
and root rates agree between contacts; the original signed full-branch integral
conventions are preserved separately. No new stress scalar ranks these cases.
Stronger curvature contrast does not imply stronger local activation stress;
larger sizes also change the existing circular-TJ site count.

The micro-interval derivatives include substantial mapped-field relaxation.
They are not extrapolated linearly to infer an event time. The ratio 0.80 is a
reversed-curvature control, not an intended smaller-center ripening candidate.

## Preserved 100/70/100 nm actual-field trajectory

The coherent overlap -> extend -> refined -> refined-target checkpoint chain
was remeasured without evolving or changing it. Across {history['samples']}
actual saved states to {last['time_s']:.6f} s, the LEFT production local stress
changes {first['LEFT_sigma_local_Pa']/1e6:.5f} -> {last['LEFT_sigma_local_Pa']/1e6:.5f} MPa,
and its root rate changes {first['LEFT_root_rate_per_s']:.6f} ->
{last['LEFT_root_rate_per_s']:.6f} s⁻¹. The center volume changes
{100*(last['center_volume_m3']/first['center_volume_m3']-1):+.6f}%.

From the first saved state at/after 0.398847 s ({late['time_s']:.6f} s),
the further local-stress rise is
{(last['LEFT_sigma_local_Pa']-late['LEFT_sigma_local_Pa'])/1e6:.6f} MPa.
Thus initial relaxation and later loading are kept distinct.
Sparse trapezoidal quadrature gives P(any root) approximately
{100*history['probability_any_root_sparse_estimate']:.3f}% through the valid trajectory.
This is a probability estimate, not a sampled or exactly localized crossing.

## Compact shortlist continuation

Rc/Ro=0.74 and Ro=81.85 nm pass cleanup with Lc/W about 8.2.
Its initial mean waiting time is 23.6 s per contact. The guarded current-field
continuation ends at **{end['time_s']:.6f} s**, status **{continuation['status']}**,
after {continuation['wall_s']:.1f} wall seconds. Center volume changes
**{100*end['center_relative_change']:+.6f}%**; final LEFT local stress is
**{end['contacts']['LEFT']['sigma_local_Pa']/1e6:.6f} MPa** and rate
**{end['contacts']['LEFT']['root_rate_per_s']:.6f} s⁻¹**.
Integrated hazards are {end['hazard']['LEFT']:.6f} and
{end['hazard']['RIGHT']:.6f}; P(any root) is approximately
**{100*continuation['probability_any_root']:.3f}%**.
A positive center-volume change means this shortlist also fails the required
continued smaller-center ripening direction, despite its initial negative
micro-interval volume rate. That is a viability rejection, not a null test.
The event-free continuation does not reach the preregistered combined-hazard
screen target of 1.25 unless its status explicitly says so.
See [every accepted state](continuation_0.74_81.850.json).

This study does not establish that the proposed shared-physics mechanism is
impossible. It establishes that these tested trajectories have not yet supplied
a validated event-producing geometry under the unchanged numerical limits.
A stochastic implementation and event/avalanche strain histories remain
conditional on that viability evidence. No stationary-null optimization is
needed or planned. A future numerical-bound remedy must be independently
validated rather than replacing the field guard, clipping, tuning kinetics or
choosing a favorable random threshold.

## Outputs and reproducibility

[Study plots](screen_plots.pdf), [preserved unequal history](preserved_unequal_history.csv)
and the JSON records retain LEFT/RIGHT stresses, hazards and center volume.
The history also labels a center two-contact stress diagnostic and an
area-weighted cluster stress diagnostic; neither enters activation.
No event markers, avalanche sizes or activated strain are fabricated for an
events-off study. Re-run the entry points listed below;
all actual field arrays stay in the isolated runs subtree with hashes.
Entry points are `three_particle_production_screen.py`,
`three_particle_production_history.py`,
`three_particle_production_viability_continue.py`,
`three_particle_production_screen_report.py` and
`three_particle_production_screen_figures.py` under `scripts/`.
'''
    second=json.loads((D/'continuation_0.65_119.999.json').read_text())
    second_end=second['history'][-1]
    extra=f'''## Stronger-drive shortlist

Rc/Ro=0.65, Ro=119.999 nm also passes cleanup at about 8.2W thickness.
Its continuation ends at {second_end['time_s']:.6f} s with status
**{second['status']}**, after {second['wall_s']:.1f} wall seconds.
Center volume changes **{100*second_end['center_relative_change']:+.6f}%**;
LEFT local stress is {second_end['contacts']['LEFT']['sigma_local_Pa']/1e6:.6f} MPa
and the root rate is {second_end['contacts']['LEFT']['root_rate_per_s']:.6f} s⁻¹.
The accumulated hazards are {second_end['hazard']['LEFT']:.6f} and
{second_end['hazard']['RIGHT']:.6f}, giving approximate any-root probability
**{100*second['probability_any_root']:.3f}%**.
See [accepted states](continuation_0.65_119.999.json).

Both shortlists use particle-positive bicrystal coordinates at each GB.
The preliminary global-coordinate integral histories are explicitly archived
as rejected diagnostic provenance; they are not used in these results.

'''
    text=text.replace('## Outputs and reproducibility',extra+'## Outputs and reproducibility')
    (D/'REPORT.md').write_text(text)
    manifest=[]
    for p in sorted(Path('runs/three_particle_production_screen').rglob('*.npz')):
        manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (D/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('candidates',len(rows),'accepted',len(accepted),'continuation',continuation['status'])
if __name__=='__main__':main()
