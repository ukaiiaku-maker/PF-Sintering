"""Diagnostic stress decomposition and volume-coordinate plateau assessment."""
import numpy as np

PLATEAU=dict(window_loss_fraction=.01,segments=4,max_abs_slope_Pa_per_fraction=5e6,max_stress_range_Pa=5e4,minimum_loss_fraction=.02)

def decompose(contact):
    out={};gamma=contact['gamma_J_per_m2'];radius=contact['r_n_m']
    for side in ('negative','positive'):
        curvature=-gamma*contact[f'kappa1_{side}_per_m']
        tj=3*gamma*np.sin(contact[f'theta_{side}_rad']/2)/radius
        if not np.isclose(curvature+tj,contact[f'sigma_local_{side}_Pa'],rtol=1e-12,atol=1e-6):raise ValueError('one-sided production decomposition mismatch')
        out[side]=dict(curvature_Pa=float(curvature),TJ_Pa=float(tj))
    out['production']={key:float(np.mean([out[side][key] for side in ('negative','positive')])) for key in ('curvature_Pa','TJ_Pa')}
    if not np.isclose(sum(out['production'].values()),contact['sigma_local_Pa'],rtol=1e-12,atol=1e-6):raise ValueError('production mean decomposition mismatch')
    return out

def plateau(rows,settings=PLATEAU):
    x=np.array([r['center_loss_fraction'] for r in rows]);end=float(x[-1]);width=settings['window_loss_fraction']
    if end<settings['minimum_loss_fraction'] or x[0]>end-width:return dict(reached=False,reason='insufficient volume-loss support')
    if np.any(np.diff(x)<=0):return dict(reached=False,reason='volume loss is not strictly increasing')
    grid=np.linspace(end-width,end,settings['segments']+1);results={}
    for name in ('LEFT','RIGHT'):
        stress=np.array([r['contacts'][name]['sigma_local_Pa'] for r in rows]);values=np.interp(grid,x,stress);slopes=np.diff(values)/np.diff(grid)
        local=np.r_[stress[(x>=grid[0])&(x<=grid[-1])],values]
        results[name]=dict(slopes_Pa_per_loss_fraction=slopes.tolist(),stress_range_Pa=float(np.ptp(local)),passes=bool(np.max(abs(slopes))<=settings['max_abs_slope_Pa_per_fraction'] and np.ptp(local)<=settings['max_stress_range_Pa']))
    return dict(reached=all(v['passes'] for v in results.values()),window_start_loss_fraction=float(grid[0]),window_end_loss_fraction=end,contacts=results)


def trial_invariant_reason(field,total_volume,initial_total,*,enforce_reflection=True):
    """Absolute acceptance guards, with reflection optional after handoff."""
    if not np.isfinite(field).all() or field.min() < -1e-8 or field.max()>1+1e-8:return 'field guard'
    if abs(total_volume/initial_total-1)>1e-11:return 'material guard'
    if enforce_reflection and np.max(abs(field-field[::-1]))>1e-8:return 'reflection guard'
    return None
