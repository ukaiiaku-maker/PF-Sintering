"""Choose one angle estimator on analytical mapped CMC fields, then freeze it.

Selection uses ratio=1 at W=10nm; ratio=1.1 is held out until selection.
No evolved state or desired transport direction enters selection.
"""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf,chain_radius


def contour(f,g,level):
    result=np.full(len(g['z']),np.nan)
    for j,row in enumerate(f):
        ids=np.flatnonzero((row[:-1]>=level)&(row[1:]<level))
        if len(ids):
            i=ids[-1];result[j]=g['r_c'][i]+(row[i]-level)/(row[i]-row[i+1])*g['dr']
    return result


def measure(f,g,rule):
    z=g['z'];W=g['config'].width;r=contour(f,g,rule['level']);result=[]
    for b in g['gb']:
        slopes=[];curvatures=[]
        for sign in [-1,1]:
            d=sign*(z-b)/W
            take=np.isfinite(r)&(d>=rule['inner_W'])&(d<=rule['outer_W'])
            if np.count_nonzero(take)<rule['degree']+2:raise ValueError('too few points for frozen angle fit')
            c=np.polyfit((z[take]-b)/W,r[take]/W,rule['degree'])
            slope=float(np.polyval(np.polyder(c),0));second=float(np.polyval(np.polyder(c,2),0)/W)
            slopes.append(slope);curvatures.append(-second/(1+slope*slope)**1.5)
        a,beta=slopes
        angle=np.rad2deg(np.arccos(np.clip(-(1+a*beta)/np.sqrt((1+a*a)*(1+beta*beta)),-1,1)))
        result.append(dict(psi_deg=float(angle),slopes=slopes,meridional_curvatures=curvatures))
    return result


def main():
    out=Path('docs/three_particle/cmc');c,o=compatible_chain(1.)
    g=map_to_pf(c,o,10e-9,1.25e-9);trials=[]
    for level in [.45,.5,.55]:
        for degree in [2,3,4]:
            for inner,outer in [(0,1),(.5,1.5),(.5,2),(1,2),(1,3)]:
                rule=dict(level=level,degree=degree,inner_W=inner,outer_W=outer)
                measured=measure(g['f'],g,rule)
                error=max(abs(r['psi_deg']-160) for r in measured)
                trials.append(dict(rule=rule,max_error_deg=error,angles_deg=[r['psi_deg'] for r in measured]))
    # Exclude the first W containing the diffuse TJ before any evolved-state test.
    # f=.5 is the Gibbs/CMC dividing surface. Other levels diagnose bias;
    # selecting an off-level contour solely to cancel fitting bias is forbidden.
    allowed=[t for t in trials if t['rule']['level']==.5 and t['rule']['inner_W']>=1]
    best=min(allowed,key=lambda t:t['max_error_deg'])
    rule=best['rule'];c2,o2=compatible_chain(1.1);g2=map_to_pf(c2,o2,10e-9,1.25e-9)
    held=measure(g2['f'],g2,rule)
    report=dict(status='FROZEN_ANALYTICAL_CALIBRATION',training_ratio=1.,held_out_ratio=1.1,width_nm=10.,spacing_nm=1.25,rule=rule,training_max_error_deg=best['max_error_deg'],held_out_angles_deg=[r['psi_deg'] for r in held],trials=trials,valid_for='mapped analytical CMC geometry; evolved-state accuracy requires cleanup validation')
    (out/'angle_calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2))
if __name__=='__main__':main()
