"""Summarize the selected events-off reload; hazards are quadratures, not roots."""
from pathlib import Path
import sys,json
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_geometry import topology_status,grain_volumes
from pf_sintering.three_particle_diagnostics import curvature_watch,radius_profile
from pf_sintering.three_particle_contacts import evaluate_contacts
D=Path('docs/three_particle/production_065');root=Path('runs/three_particle_production_screen/harmonic_continuation_0.65_119.999')
d=json.loads(Path('docs/three_particle/production_screen/harmonic_continuation_0.65_119.999.json').read_text());r=d['history'];t=np.array([x['time_s'] for x in r])
c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g)
# Use immutable sparse snapshots while the live continuation is still writing.
files=sorted(root.glob('state_*.npz'),key=lambda p:float(p.stem.split('_')[1][:-1]));frames=[];audits=[]
for path in [root/'post_transient.npz',*files[::max(1,len(files)//5)],files[-1]]:
    with np.load(path) as data:f=data['f'].copy();ts=float(data['t_model'])*op.physics.seconds_per_model_time
    frames.append((ts,radius_profile(f,g)))
    audits.append(dict(source=str(path),time_s=ts,topology=topology_status(f,g),watch=curvature_watch(f,op,two_contact_center=True)))
report=dict(events_enabled=False,status=d['status'],time_s=float(t[-1]),samples=len(r),
  max_f=max(x['f_max'] for x in r),min_f=min(x['f_min'] for x in r),
  max_abs_mass_relative_error=max(abs(x['mass_relative_error']) for x in r),
  max_mirror_error=max(x['mirror_error'] for x in r),
  maximum_energy_relative_increment=max(b['energy_J']/a['energy_J']-1 for a,b in zip(r,r[1:])),
  center_relative_change=r[-1]['center_relative_change'],
  instantaneous_first_root_timescale_s=1.25/sum(r[-1]['contacts'][c]['root_rate_per_s'] for c in ['LEFT','RIGHT']),
  old_field_guard_unchanged=True,clipping=False,morphology_audits=audits)
(D/'reload_summary.json').write_text(json.dumps(report,indent=2,default=float)+'\n')
fig,axes=plt.subplots(2,3,figsize=(13,7));ax=axes.ravel()
ax[0].plot(t,[100*x['center_relative_change'] for x in r]);ax[0].set_ylabel('Center volume change (%)')
for name in ['LEFT','RIGHT']:
    ax[1].plot(t,[x['contacts'][name]['sigma_local_Pa']/1e6 for x in r],label=name)
    ax[2].plot(t,[x['hazard'][name] for x in r],label=name)
ax[1].set_ylabel('Production local stress (MPa)');ax[1].legend();ax[2].set_ylabel('Integrated root rate (no sampled threshold)')
ax[3].plot(t,[x['energy_J']/r[0]['energy_J']-1 for x in r]);ax[3].set_ylabel('Relative free-energy change')
ax[4].plot(t,[x['f_max']-1 for x in r]);ax[4].set_ylabel('max f - 1');ax[4].ticklabel_format(useOffset=False,axis='y')
for ts,radius in frames:ax[5].plot(g['z']*1e9,radius*1e9,label=f'{ts:.2f} s')
ax[5].set_xlabel('z (nm)');ax[5].set_ylabel('Surface radius (nm)');ax[5].legend(fontsize=7)
for a in ax[:5]:a.set_xlabel('Physical time (s)');a.grid(alpha=.2)
fig.suptitle('Selected 0.65 events-off reload — unchanged kinetics and 1e-8 field guard')
fig.tight_layout();fig.savefig(D/'reload.pdf');fig.savefig(D/'reload.png',dpi=130);plt.close(fig)
print({k:v for k,v in report.items() if k!='morphology_audits'},flush=True)
