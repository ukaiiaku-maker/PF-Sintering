"""Plot the completed fixed-q audit without labeling capped states converged."""
from pathlib import Path
import sys,json
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from three_particle_forced_event import ContactEvent
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_diagnostics import radius_profile
from monitor_current_state_transfer_curvature import branch_profile
D=Path('docs/three_particle/production_065');data=np.load('runs/three_particle_production_065/fixed_q_relaxation_audit.npz')
assert np.array_equal(data['half_tolerance'],data['quarter_tolerance'])
c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=ContactEvent(g);fig,axes=plt.subplots(1,3,figsize=(13,4));limits=[[] for _ in axes]
for key,label,style in [('initial','Input q/b = 0.5','-'),('nominal','Nominal: 5 blocks','--'),('half_tolerance','Both tighter cases: 2048-block cap','-')]:
 state=tuple(data[key]);m=event.metrics(state,.5);z=g['z'];r=radius_profile(state[0],g);left=m['LEFT_z_TJ_m'];right=m['RIGHT_z_TJ_m'];valid=np.isfinite(r)&(z>=left)&(z<=right)
 p=branch_profile(z[valid],r[valid],z_tj=left,r_tj=m['LEFT_r_n_m'],W=4e-9,side='center');x=p['distance_from_TJ_m']/4e-9
 for index,(ax,y) in enumerate(zip(axes,[p['r_m']*1e9,p['kappa_m_per_m']/1e6,p['dkappa_m_ds_per_m2']/1e14])):
  ax.plot(x,y,style,label=label);limits[index].extend(y[(x>=1)&(x<=7)&np.isfinite(y)].tolist())
for ax,values,ylabel in zip(axes,limits,['Center radius (nm)','Meridional curvature (1/µm)','Curvature gradient (10¹⁴/m²)']):
 lo,hi=min(values),max(values);pad=.1*(hi-lo);ax.set_ylim(lo-pad,hi+pad);ax.set_xlim(1,7);ax.axvline(3,color='gray',ls=':');ax.set_xlabel('Distance from LEFT TJ / W');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
axes[0].legend(fontsize=7);fig.suptitle('Fixed-q diagnostic: no added event quota; tighter criteria did not converge within the cap')
fig.tight_layout();fig.savefig(D/'fixed_q_relaxation_audit.pdf');fig.savefig(D/'fixed_q_relaxation_audit.png',dpi=140);plt.close(fig)
