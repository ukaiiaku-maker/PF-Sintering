from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pf_sintering.three_particle_geometry import build_three_particle
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_diagnostics import diagnostics,curvature_watch

g=build_three_particle();op=PhaseAOperator(g)
p=Path('runs/three_particle_phase_a/ratio_0.7')
with np.load(p/'initialization_candidate.npz',allow_pickle=False) as data:f=data['f']
d,profiles=diagnostics(f,op);seed,sp=diagnostics(g['f'],op)
fig,axes=plt.subplots(3,1,figsize=(11,9))
z=g['z']*1e9
for sign in [-1,1]:
    axes[0].plot(z,sign*sp['r_m']*1e9,color='gray',label='seed' if sign==1 else None)
    axes[0].plot(z,sign*profiles['r_m']*1e9,color='#277da8',label='candidate' if sign==1 else None)
axes[0].set(aspect='equal',ylabel='r (nm)',title='Initialization candidate — NOT QUALIFIED');axes[0].legend()
axes[1].plot(z,profiles['kappa_per_m']*1e-9);axes[1].set(ylabel='outward curvature (1/nm)',ylim=(-.15,.15))
axes[2].imshow(f.T,origin='lower',aspect='auto',extent=[z[0],z[-1],g['r_c'][0]*1e9,g['r_c'][-1]*1e9],vmin=0,vmax=1)
axes[2].set(ylabel='r (nm)',xlabel='z (nm)')
for ax in axes:
    for b in g['gb']:ax.axvline(b*1e9,color='black',ls=':',lw=.8)
fig.tight_layout();fig.savefig('docs/three_particle/initialization_candidate.png',dpi=150)
watch=curvature_watch(f,op)
Path('docs/three_particle/initialization_watch.json').write_text(json.dumps(watch,indent=2)+'\n')
print(json.dumps(d,indent=2))
