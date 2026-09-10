"""Remeasure archived screen fields after contact-indexing changes; no PF evolution."""
from pathlib import Path
import sys,json
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_geometry import ThreeParticleConfig
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_contacts import evaluate_contacts
D=Path('docs/three_particle/production_screen')

def main():
    manifest=json.loads((D/'bicrystal_launch_manifest.json').read_text())
    for name in ['screen.json','shortlist_074.json']:
        report=json.loads((D/name).read_text())
        for row in report['candidates']:
            if 'checkpoint' not in row:continue
            with np.load(row['checkpoint']['path']) as d:
                r=d['r_c'];z=d['z'];dr=.5e-9;L=float(d['gb'][1]);dz=L/int(np.ceil(L/dr))
                g=dict(f=d['f_initial'],ownership=d['ownership'],z=z,r_c=r,r_f=np.arange(len(r)+1)*dr,
                    dr=dr,dz=dz,gb=d['gb'],config=ThreeParticleConfig(outer_radius=row['outer_radius_nm']*1e-9,center_ratio=row['ratio'],width=4e-9,spacing=dr))
                op=PhaseAOperator(g)
                row['initial']=evaluate_contacts(d['f_initial'],op,manifest)
                row['final']=evaluate_contacts(d['f_final'],op,manifest)
        report['metrology_frame']='particle-positive at each contact; side output mapped to global grain order'
        (D/name).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
