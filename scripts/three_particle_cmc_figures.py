from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,chemical_potential_null,chain_radius,describe

out=Path('docs/three_particle/cmc');data=json.loads((out/'geometry_scan.json').read_text())
null,cap=chemical_potential_null();record=describe(null,cap)
(out/'chemical_potential_null.json').write_text(json.dumps(record,indent=2)+'\n')
fig,ax=plt.subplots(1,2,figsize=(11,4))
r=[x['ratio'] for x in data['compatible_chains']]
ax[0].plot(r,[x['delta_mu_Pa']/1e6 for x in data['compatible_chains']],'o-')
ax[0].axhline(0,color='gray');ax[0].axvline(null.ratio,color='black',ls=':',label=f'null ratio {null.ratio:.5f}')
ax[0].set(xlabel='Center / outer volume-equivalent radius',ylabel='gamma (Kc − Ko) (MPa)',title='Connected, Young-balanced CMC branch');ax[0].legend()
ax[1].plot(r,[2*x['center_half_length_nm'] for x in data['compatible_chains']],'o-')
ax[1].axhline(80,color='red',ls='--',label='8W, W=10 nm');ax[1].axhline(32,color='green',ls=':',label='8W, W=4 nm')
ax[1].set(xlabel='Center / outer volume-equivalent radius',ylabel='GB separation (nm)',title='Independent topology-resolution screen');ax[1].legend()
fig.tight_layout();fig.savefig(out/'cmc_geometry_map.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(10,7))
for ax,(c,o,label) in zip(axes,[(*compatible_chain(.7),'100/70/100 nm by grain volume'),(null,cap,'Chemical-potential-matched null')]):
    tip=c.half_length+o['pole_distance_m'];z=np.linspace(-tip,tip,4001);radius=chain_radius(z,c,o)
    for lo,hi,color in [(-tip,-c.half_length,'#277da8'),(-c.half_length,c.half_length,'#edaa43'),(c.half_length,tip,'#277da8')]:
        m=(z>=lo)&(z<=hi);ax.fill_between(z[m]*1e9,-radius[m]*1e9,radius[m]*1e9,color=color)
    for b in [-c.half_length,c.half_length]:ax.plot([b*1e9]*2,[-c.contact_radius*1e9,c.contact_radius*1e9],color='black',lw=.8)
    ax.set(aspect='equal',xlabel='z (nm)',ylabel='r (nm)',title=f'{label}; ψ=160°; CMC geometry only')
fig.tight_layout();fig.savefig(out/'analytical_morphologies.png',dpi=170)
print(json.dumps(record,indent=2))
