"""Read-only comparison of strain definitions on the existing no-event control."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('docs/three_particle/production_065');root=Path('runs/three_particle_production_screen/harmonic_continuation_0.65_119.999')
manifest=json.loads(Path('docs/three_particle/production_screen/bicrystal_launch_manifest.json').read_text());seconds=manifest['seconds_per_model_time']
def measure(path):
 with np.load(path) as data:
  if bool(data['events_enabled']):raise ValueError('requires a no-event control')
  f=data['f'];phi=data['ownership'];z=data['z'];r=data['r_c'];weights=phi*f[None]*r[None,None,:]
  centers=np.sum(weights*z[None,:,None],axis=(1,2))/np.sum(weights,axis=(1,2));line=f[:,0];inside=np.flatnonzero(line>=.5);i,j=inside[0],inside[-1]
  if i==0 or j==len(z)-1:raise ValueError('extent touches the domain boundary')
  low=z[i-1]+(.5-line[i-1])*(z[i]-z[i-1])/(line[i]-line[i-1]);high=z[j]+(.5-line[j])*(z[j+1]-z[j])/(line[j+1]-line[j])
  return dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),time_s=float(data['t_model'])*seconds,
   centroid_span_m=float(centers[2]-centers[0]),near_axis_extent_m=float(high-low),extent_sample_radius_m=float(r[0]),
   center_volume_m3=float(2*np.pi*(r[1]-r[0])*(z[1]-z[0])*weights[1].sum()))
reference=measure(root/'post_transient.npz');rows=[reference]
for path in [*root.glob('state_*.npz'),root/'checkpoint.npz']:
 row=measure(path)
 if row['time_s']>reference['time_s']+1e-12:rows.append(row)
rows.sort(key=lambda row:row['time_s'])
for row in rows:
 row.update(geometric_chain_strain=1-row['centroid_span_m']/reference['centroid_span_m'],near_axis_extent_strain=1-row['near_axis_extent_m']/reference['near_axis_extent_m'],center_relative_volume_change=row['center_volume_m3']/reference['center_volume_m3']-1,production_densification_strain=0.)
fig,axes=plt.subplots(2,2,figsize=(10,7));t=np.array([r['time_s']-reference['time_s'] for r in rows])
for ax,key,label in zip(axes.ravel(),['production_densification_strain','geometric_chain_strain','near_axis_extent_strain','center_relative_volume_change'],['Bicrystal quota-based strain (%)','Geometric centroid strain (%)','Near-axis extent strain (%)','Center volume change (%)']):
 ax.plot(t,[100*r[key] for r in rows]);ax.set_xlabel('Elapsed time after post-transient state (s)');ax.set_ylabel(label);ax.grid(alpha=.2)
fig.suptitle('Passive no-event control: strain definitions are not interchangeable');fig.tight_layout();fig.savefig(D/'passive_strain_accounting.pdf');fig.savefig(D/'passive_strain_accounting.png',dpi=140);plt.close(fig)
report=dict(label='PASSIVE_CONTROL_STRAIN_ACCOUNTING',events_enabled=False,reference=reference,rows=rows,
 production_definition='cumulative accepted event quota times b / initial centroid separation',production_source='scripts/pr_avalanche_renewal_five.py:540',
 geometric_definition='1 - current outer-grain centroid separation / initial separation',extent_definition='1 - current f=0.5 axial extent at the innermost radial cell / initial extent',
 note='The near-axis extent is a separate diagnostic, not a replacement production strain. No fitted baseline subtraction or change to activation is applied.',
 one_forced_b_counted_strain=manifest['b_event_m']/reference['centroid_span_m'])
(D/'passive_strain_accounting.json').write_text(json.dumps(report,indent=2)+'\n');print(dict(records=len(rows),elapsed_s=t[-1],final=rows[-1],one_forced_b_counted_strain=report['one_forced_b_counted_strain']),flush=True)
