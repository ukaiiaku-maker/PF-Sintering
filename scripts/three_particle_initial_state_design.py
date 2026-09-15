#!/usr/bin/env python3
"""Select earlier-stage three-particle initial states from local headroom only.

No trajectory, kinetic projection, phase-field step, or stochastic draw occurs.
"""
from dataclasses import asdict, replace
from pathlib import Path
import csv, json, math, sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pf_sintering.three_particle_contacts import root_law
from pf_sintering.three_particle_sharp_design import (
    SharpDesign, admissibility, center_squared_radius_coefficients,
    contact_state, evaluate_even_squared_radius, half_chain_profile,
    sharp_free_energy,
)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/three_particle/initial_state_design'
MANIFEST=ROOT/'docs/three_particle/production_screen/bicrystal_launch_manifest.json'
W=4e-9; GAMMA_GB=.34729635533386083; H=0.002


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def design_id(d):
    kc=d.k_center0_per_m*1e-6;ko=d.k_outer0_per_m*1e-6
    return (f'Ro{d.Ro_m*1e9:.0f}_q{d.Rc_over_Ro:.2f}_L{d.Lc_m/W:.0f}W_'
            f'r{d.rTJ_over_Rc:.2f}_th{d.theta_center_deg:.0f}_kc{kc:+.0f}_ko{ko:+.0f}')


def physical_copy(d,**changes):
    Ro=changes.get('Ro_m',d.Ro_m);Rc=changes.get('Rc_m',d.Rc_m)
    r=changes.get('rTJ_m',d.rTJ0_m);L=changes.get('Lc_m',d.Lc_m)
    kc=changes.get('kc_per_m',d.k_center0_per_m);ko=changes.get('ko_per_m',d.k_outer0_per_m)
    tc=changes.get('theta_center_deg',d.theta_center_deg);to=changes.get('theta_outer_deg',d.theta_outer_deg)
    return SharpDesign(Ro,Rc/Ro,L,r/Rc,tc,to,kc*Rc,ko*Ro,d.gamma_s,d.W_m)


def perturb(d,name,q):
    if name=='center_outer_exchange':
        Rc=d.Rc_m*math.exp(q)
        Ro3=d.Ro_m**3-.5*(Rc**3-d.Rc_m**3)
        if Ro3<=0:raise ValueError('mass exchange exhausted outer volume')
        return physical_copy(d,Ro_m=Ro3**(1/3),Rc_m=Rc)
    if name=='tj_radius':return physical_copy(d,rTJ_m=d.rTJ0_m*math.exp(q))
    if name=='center_curvature':return physical_copy(d,kc_per_m=d.k_center0_per_m+q/W)
    if name=='outer_curvature':return physical_copy(d,ko_per_m=d.k_outer0_per_m+q/W)
    if name=='center_angle':return physical_copy(d,theta_center_deg=d.theta_center_deg+math.degrees(q))
    if name=='outer_angle':return physical_copy(d,theta_outer_deg=d.theta_outer_deg+math.degrees(q))
    if name=='gb_spacing':return physical_copy(d,Lc_m=d.Lc_m*math.exp(q))
    raise KeyError(name)


def initial_metrics(d,manifest):
    state=contact_state(d,0.);energy=sharp_free_energy(d,0.,GAMMA_GB,401)
    law=root_law(state['sigma_GB_Pa'],state['r_TJ_m'],manifest)
    coeff=center_squared_radius_coefficients(d);u=np.linspace(0,1,401)
    center_min=float(np.sqrt(np.min(evaluate_even_squared_radius(coeff,u))))
    center_max=float(np.sqrt(np.max(evaluate_even_squared_radius(coeff,u))))
    profile=half_chain_profile(d,0.,401)
    outer_max=float(np.max(profile['outer_r_m']))
    return {**energy,**state,**law,'center_min_radius_m':center_min,
        'outer_max_radius_m':outer_max,'center_max_radius_m':center_max,'center_span_over_W':d.Lc_m/W,
        'rTJ_over_W':d.rTJ0_m/W,'center_min_radius_over_W':center_min/W,
        'vacuum_topology_margin_over_W':min(center_min/W-3,d.rTJ0_m/W-6,d.Lc_m/W-8),
        'two_contact_characteristic_wait_s':manifest['root_threshold_multiplier']/(2*law['root_rate_per_s'])}


def local_headroom(d,manifest):
    rows=[]
    for name in ('center_outer_exchange','tj_radius','center_curvature','outer_curvature',
                 'center_angle','outer_angle','gb_spacing'):
        try:
            dm=perturb(d,name,-H);dp=perturb(d,name,H)
            am=admissibility(dm,x_max=.10,samples=201);ap=admissibility(dp,x_max=.10,samples=201)
            if not am['admissible'] or not ap['admissible']:continue
            mm=initial_metrics(dm,manifest);mp=initial_metrics(dp,manifest)
        except (ValueError,np.linalg.LinAlgError):continue
        dE=(mp['free_energy_J']-mm['free_energy_J'])/(2*H)
        ds=(mp['sigma_GB_Pa']-mm['sigma_GB_Pa'])/(2*H)
        dr=(mp['r_TJ_m']-mm['r_TJ_m'])/(2*H)
        dln=(math.log(mp['root_rate_per_s'])-math.log(mm['root_rate_per_s']))/(2*H)
        direction=-1. if dE>0 else 1.
        rows.append(dict(coordinate=name,dE_dq_J=dE,dsigma_dq_Pa=ds,drTJ_dq_m=dr,
            dlnGamma_dq=dln,energy_lowering_direction=direction,
            directional_minus_dE_dq_J=abs(dE),directional_dsigma_dq_Pa=direction*ds,
            directional_drTJ_dq_m=direction*dr,directional_dlnGamma_dq=direction*dln,
            accessible_loading=bool(direction*ds>1e4)))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True);manifest=json.loads(MANIFEST.read_text())
    candidates=[];directions=[];objects={}
    for Ro_nm in (120.,150.,180.):
      for ratio in (.35,.45,.55):
       for L_over_Rc in (1.4,1.8,2.2):
        for rr in (.45,.60,.75):
         for theta in (120.,140.,160.):
          for km in (-10.,0.,10.):
           L=max(12*W,L_over_Rc*ratio*Ro_nm*1e-9)
           d=SharpDesign(Ro_nm*1e-9,ratio,L,rr,theta,theta,
                         km*1e6*ratio*Ro_nm*1e-9,km*1e6*Ro_nm*1e-9)
           adm=admissibility(d,x_max=.10,samples=301)
           if not adm['admissible']:continue
           # Earlier-stage contacts retain at least 20% radial freedom to the outer envelope.
           if d.rTJ0_m/d.Ro_m>0.75:continue
           try:m=initial_metrics(d,manifest);local=local_headroom(d,manifest)
           except (ValueError,np.linalg.LinAlgError):continue
           if m['center_max_radius_m']>1.35*d.Rc_m or m['outer_max_radius_m']>1.20*d.Ro_m:continue
           loading=[r for r in local if r['accessible_loading']]
           if not loading:continue
           name=design_id(d);objects[name]=d
           for row in local:directions.append({'design_id':name,**row})
           best=max(loading,key=lambda r:r['directional_dsigma_dq_Pa'])
           score=(2.0*min(adm['resolution_margin'],2.5)+1.2*len(loading)
                  +min(best['directional_dsigma_dq_Pa']/25e6,4)
                  +2*math.exp(-abs(math.log(m['two_contact_characteristic_wait_s']/10)))
                  +2*math.exp(-abs(theta-160)/20))
           candidates.append(dict(design_id=name,Ro_nm=Ro_nm,Rc_nm=d.Rc_m*1e9,
             Rc_over_Ro=ratio,Lc_over_W=d.Lc_m/W,Lc_over_Rc=d.Lc_m/d.Rc_m,
             rTJ_nm=d.rTJ0_m*1e9,rTJ_over_W=d.rTJ0_m/W,
             rTJ_over_Ro=d.rTJ0_m/d.Ro_m,theta_center_negative_deg=theta,
             theta_center_positive_deg=theta,theta_outer_negative_deg=theta,
             theta_outer_positive_deg=theta,center_curvature_per_um=km,
             outer_curvature_per_um=km,sigma0_MPa=m['sigma_GB_Pa']*1e-6,
             root_rate_per_contact_s=m['root_rate_per_s'],two_contact_characteristic_wait_s=m['two_contact_characteristic_wait_s'],
             surface_area_m2=m['surface_area_m2'],GB_area_m2=m['GB_area_m2'],
             total_interfacial_energy_J=m['free_energy_J'],center_volume_m3=m['Vc_m3'],
             outer_volume_each_m3=m['Vo_m3'],center_min_radius_over_W=m['center_min_radius_over_W'],
             resolution_margin=adm['resolution_margin'],topology_margin_over_W=m['vacuum_topology_margin_over_W'],
             accessible_loading_direction_count=len(loading),best_loading_coordinate=best['coordinate'],
             best_directional_dsigma_dq_MPa=best['directional_dsigma_dq_Pa']*1e-6,
             best_directional_minus_dE_dq_fJ=best['directional_minus_dE_dq_J']*1e15,
             best_directional_dlnGamma_dq=best['directional_dlnGamma_dq'],rank_score=score))
    candidates.sort(key=lambda r:r['rank_score'],reverse=True)
    # Greedy diversity in size, ratio, and curvature; no outcome or random draw is inspected.
    selected=[]
    for row in candidates:
        signature=(row['Ro_nm'],row['Rc_over_Ro'],row['center_curvature_per_um'])
        if any(sum(a!=b for a,b in zip(signature,(q['Ro_nm'],q['Rc_over_Ro'],q['center_curvature_per_um'])))<2 for q in selected):continue
        selected.append(row)
        if len(selected)==5:break
    choice=max(selected,key=lambda r:(r['resolution_margin'],r['accessible_loading_direction_count'],r['rank_score']))
    write_csv(OUT/'all_initial_candidates.csv',candidates);write_csv(OUT/'local_perturbations.csv',directions)
    write_csv(OUT/'selected_initial_candidates.csv',selected)
    payload=dict(label='EARLIER_STAGE_INITIAL_STATE_DESIGN',phase_field_run=False,stochastic_draws=False,
        prescribed_evolution=False,search_count=3*3*3*3*3*3,admissible_headroom_count=len(candidates),
        selected=selected,chosen_design_id=choice['design_id'],chosen_design=asdict(objects[choice['design_id']]),
        perturbation_step_dimensionless=H,perturbation_coordinates={
          'center_outer_exchange':'d ln Rc with exact conservation of 2 Vo + Vc; L, rTJ, angles, physical curvatures fixed',
          'tj_radius':'d ln rTJ at fixed volumes, L, angles, and physical curvatures',
          'center_curvature':'d(k_center W)','outer_curvature':'d(k_outer W)',
          'center_angle':'d theta_center in radians','outer_angle':'d theta_outer in radians',
          'gb_spacing':'d ln Lc at fixed volumes, rTJ, angles, and physical curvatures'},
        selection_rule='resolution, multiple independent energy-lowering/stress-raising directions, moderate unchanged root clock, and geometric diversity; no trajectory prediction')
    (OUT/'selection.json').write_text(json.dumps(payload,indent=2)+'\n')
    report=['# Earlier-stage initial-state design','',
      'This screen selects an initial geometry only. It contains no prescribed trajectory, kinetic projection, PF evolution, or stochastic draw.', '',
      f"{len(candidates)} of {payload['search_count']} cases are resolved through a conservative 10% topology probe and possess at least one independent local direction with dE/dq<0 and d sigma_local/dq>0.",'',
      '|role|design|sigma0 (MPa)|two-contact clock (s)|rTJ/W|resolution margin|loading directions|best coordinate|',
      '|---|---|---:|---:|---:|---:|---:|---|']
    for i,row in enumerate(selected,1):
        report.append(f"|C{i}{' (chosen)' if row is choice else ''}|`{row['design_id']}`|{row['sigma0_MPa']:.2f}|{row['two_contact_characteristic_wait_s']:.2f}|{row['rTJ_over_W']:.1f}|{row['resolution_margin']:.2f}|{row['accessible_loading_direction_count']}|{row['best_loading_coordinate']}|")
    report += ['',f"Chosen initial state: `{choice['design_id']}`. It is selected before mapping and before any stochastic seed is used.",'',
      'The local derivatives demonstrate accessible headroom only. They neither predict nor impose the direction, rate, or magnitude of subsequent evolution. The exact bicrystal stress and root law are used unchanged.']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')

if __name__=='__main__':main()
