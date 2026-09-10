"""Render the finite-neck seed without starting any physical evolution."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pf_sintering.three_particle_geometry import build_three_particle,grain_volumes,topology_status

def main():
    g=build_three_particle(); out=Path('docs/three_particle'); out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(11,4))
    for i,color in enumerate(['#277da8','#edaa43','#277da8']):
        for sign in [-1,1]:
            ax.contourf(g['z']*1e9,sign*g['r_c']*1e9,g['eta'][i].T,levels=[.5,1.001],colors=[color])
    for b in g['gb']:
        ax.plot([b*1e9]*2,[-g['neck']*1e9,g['neck']*1e9],color='#323b43',lw=2)
    ax.set(xlabel='z (nm)',ylabel='r (nm)',title='Three-particle finite-neck seed — NOT equilibrated',aspect='equal',ylim=(-130,130),xlim=(-285,285))
    fig.tight_layout();fig.savefig(out/'seed_morphology.png',dpi=160);plt.close(fig)
    v=grain_volumes(g['f'],g)
    record=dict(nominal_radii_nm=(g['radii']*1e9).tolist(),centers_nm=(g['centers']*1e9).tolist(),gb_z_nm=(g['gb']*1e9).tolist(),neck_radius_nm=g['neck']*1e9,volumes_m3=v.tolist(),equivalent_radii_nm=((3*v/(4*np.pi))**(1/3)*1e9).tolist(),shape=list(g['f'].shape),mirror_error=float(np.max(np.abs(g['f']-g['f'][::-1]))),topology=topology_status(g['f'],g),canonical=False)
    (out/'seed_geometry.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
