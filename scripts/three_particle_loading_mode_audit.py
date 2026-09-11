"""Finite-difference native-operator response of the observed odd mode."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_bounded_mobility import bounded_mobility_update,HarmonicSurfaceDiffusion
D=Path('docs/three_particle/volume_loading_065');source=D/'late_stiffness_state.npz'
with np.load(source) as d:f=d['f'].copy()
c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);mode=f-f[::-1];mode/=np.max(abs(mode));weight=g['r_c'][None,:];rows=[];responses=[]
def rhs(field):return bounded_mobility_update(field,op.potential(field).copy(),op,1.)-field
for eps in [1e-8,1e-9,1e-10]:
 response=(rhs(f+eps*mode)-rhs(f-eps*mode))/(2*eps);responses.append(response);rayleigh=float(np.sum(mode*response*weight)/np.sum(mode*mode*weight));norm=float(np.sqrt(np.sum(response**2*weight)/np.sum(mode**2*weight)));rows.append(dict(epsilon=eps,native_RHS_rayleigh_per_model_time=rayleigh,response_norm_rate_per_model_time=norm,alignment=rayleigh/norm));print(rows[-1])
it=HarmonicSurfaceDiffusion(op)
from scipy import sparse
H=it.Hgradient+sparse.diags(op.W_f*(1-6*f.ravel()+6*f.ravel()**2));frozen=(it.mobility(f)@(H@mode.ravel())).reshape(f.shape)
report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,frozen_AH_rayleigh_per_model_time=float(np.sum(mode*frozen*weight)/np.sum(mode*mode*weight)),no_accepted_fields_changed=True,no_symmetry_projection=True)
(D/'native_mode_response.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(D/'native_mode_response.npz',mode=mode,responses=np.array(responses),frozen=frozen);print(report)
