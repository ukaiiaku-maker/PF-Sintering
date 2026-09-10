"""Bounded native restart overlap at the end of the million-step extension."""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_cmc_ripening import load_initial
from three_particle_implicit_run import record,advance
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_implicit import ImplicitSurfaceDiffusion


def main():
    result={}
    for case in ['unequal','null']:
        g,m=load_initial(case);op=PhaseAOperator(g);it=ImplicitSurfaceDiffusion(op)
        with np.load(f'runs/three_particle_implicit/extend_v1_{case}/checkpoint.npz') as d:
            f=d['f'].copy();t=float(d['t_model'])
        h=10000*m['dt_model'];start=time.perf_counter();a,err=advance(f,h,it,m['rule']);implicit_wall=time.perf_counter()-start
        if err>1:raise RuntimeError('late single macrostep failed controller')
        start=time.perf_counter();n=f.copy()
        for _ in range(10000):n=op.step(n,m['dt_model'])
        native_wall=time.perf_counter()-start
        r0=record(f,op,m,t,0);ra=record(a,op,m,t+h,1);rn=record(n,op,m,t+h,10000)
        errors={k:abs(ra[k]-rn[k]) for k in ['LEFT_kappa_per_m','RIGHT_kappa_per_m','LEFT_neck_r_m','RIGHT_neck_r_m','LEFT_CC_stress_Pa','RIGHT_CC_stress_Pa','LEFT_PF_geometric_stress_Pa','RIGHT_PF_geometric_stress_Pa','delta_mu_projected_Pa']}
        for grain in ['left','center','right']:
            k='V_'+grain+'_m3';errors[k+'_relative']=abs(ra[k]-rn[k])/r0[k]
        errors['field_max']=float(np.max(abs(a-n)))
        good=(errors['field_max']<2e-5 and all(errors['V_'+x+'_m3_relative']<1e-8 for x in ['left','center','right'])
              and max(errors[x+'_CC_stress_Pa'] for x in ['LEFT','RIGHT'])<20000
              and max(errors[x+'_PF_geometric_stress_Pa'] for x in ['LEFT','RIGHT'])<20000
              and max(errors[x+'_kappa_per_m'] for x in ['LEFT','RIGHT'])<20000
              and max(errors[x+'_neck_r_m'] for x in ['LEFT','RIGHT'])<2e-12
              and errors['delta_mu_projected_Pa']<5000)
        result[case]=dict(pass_check=bool(good),t_start_model=t,h_model=h,implicit_wall_s=implicit_wall,native_wall_s=native_wall,error_estimate=err,errors=errors,initial=r0,implicit=ra,native=rn)
        print(case,good,errors,flush=True)
    out=Path('docs/three_particle/implicit/late_native_overlap.json')
    out.write_text(json.dumps(dict(status='PASS' if all(v['pass_check'] for v in result.values()) else 'FAIL',cases=result),indent=2)+'\n')
if __name__=='__main__':main()
