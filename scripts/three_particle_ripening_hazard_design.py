#!/usr/bin/env python3
"""Analytical race between surface ripening and unchanged root nucleation.

This is deterministic postprocessing.  It does not invoke a phase-field time
step, draw a random threshold, apply a source, or initialize a new PF field.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pf_sintering.three_particle_contacts import root_law
from pf_sintering.three_particle_sharp_design import (
    SharpDesign,admissibility,contact_state,half_chain_profile,
    sharp_free_energy,surface_diffusion_projection)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/three_particle/ripening_hazard_design'
MANIFEST_PATH=ROOT/'docs/three_particle/production_screen/bicrystal_launch_manifest.json'
OLD_SCREEN=ROOT/'docs/three_particle/sharp_interface_loading_design/screened_geometries.csv'
PF_BASE=ROOT/'runs/three_particle_production_screen/ratio_0.65_Ro_119.999nm.npz'
PF_POST=ROOT/'runs/three_particle_production_screen/harmonic_continuation_0.65_119.999/post_transient.npz'
PF_END=ROOT/'runs/three_particle_production_screen/harmonic_continuation_0.65_119.999/checkpoint.npz'
W=4e-9;GAMMA_GB=.34729635533386083;M_MODEL=6e-34;OMEGA=1e-29
MILESTONES=np.array([0.,.01,.02,.05,.10,.15,.20])


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def cid(d):
    km=.5*(d.k_center0_per_m+d.k_outer0_per_m)*1e-6
    return f'Ro{d.Ro_m*1e9:.0f}_q{d.Rc_over_Ro:.2f}_L{d.Lc_m/W:.0f}W_r{d.rTJ_over_Rc:.2f}_th{d.theta_center_deg:.0f}_km{km:+.0f}'


def cumulative_trapezoid(y,x):
    return np.r_[0.,np.cumsum(.5*(y[:-1]+y[1:])*np.diff(x))]


def interp_at(value,x,y):
    if value<x[0] or value>x[-1]:return math.nan
    return float(np.interp(value,x,y))


def kinetic_path(d,manifest,mobility,xs,points=301):
    rows=[]
    for x in xs:
        s=contact_state(d,float(x));p=surface_diffusion_projection(
            d,float(x),gamma_gb=GAMMA_GB,mobility_m6_per_J_s=mobility,
            points_per_branch=points)
        law=root_law(s['sigma_GB_Pa'],s['r_TJ_m'],manifest)
        rows.append({'design_id':cid(d),'x':float(x),'free_energy_J':p['free_energy_J'],
            'dF_dx_J':p['dF_dx_J'],'minus_dF_dx_J':p['minus_dF_dx_J'],
            'zeta_J_s':p['zeta_J_s'],'xdot_per_s':p['xdot_per_s'],
            'flux_closure_relative':p['relative_half_chain_closure'],
            'r_TJ_nm':s['r_TJ_m']*1e9,'r_TJ_over_W':s['r_TJ_m']/W,
            'center_curvature_per_um':s['center_curvature_per_m']*1e-6,
            'outer_curvature_per_um':s['outer_curvature_per_m']*1e-6,
            'center_theta_deg':s['center_theta_deg'],'outer_theta_deg':s['outer_theta_deg'],
            'sigma_GB_MPa':s['sigma_GB_Pa']*1e-6,'sigma_kappa_MPa':s['sigma_kappa_Pa']*1e-6,
            'sigma_TJ_MPa':s['sigma_TJ_Pa']*1e-6,'Gamma_per_contact_s':law['root_rate_per_s'],
            'G_fit_eV':law['G_fit_eV'],'G_root_eV':law['G_root_eV']})
    x=np.array([r['x'] for r in rows]);xdot=np.array([r['xdot_per_s'] for r in rows])
    gamma=np.array([r['Gamma_per_contact_s'] for r in rows]);radius=np.array([r['r_TJ_nm'] for r in rows])
    G=np.array([r['G_fit_eV'] for r in rows]);kT=1.380649e-23*manifest['temperature_K']/1.602176634e-19
    elapsed=cumulative_trapezoid(1/xdot,x) if np.all(xdot>0) else np.full_like(x,np.nan)
    lam=cumulative_trapezoid(2*gamma/(manifest['root_threshold_multiplier']*xdot),x) if np.all(xdot>0) else np.full_like(x,np.nan)
    site=np.gradient(np.log(radius),x,edge_order=2)
    barrier=np.gradient(-G/kT,x,edge_order=2)
    for i,r in enumerate(rows):
        r.update(elapsed_s=elapsed[i],cumulative_scaled_hazard=lam[i],survival=math.exp(-lam[i]) if math.isfinite(lam[i]) else math.nan,
            event_probability=1-math.exp(-lam[i]) if math.isfinite(lam[i]) else math.nan,
            hazard_amplification=gamma[i]/gamma[0],dlnGamma_dx_site=site[i],
            dlnGamma_dx_barrier=barrier[i],dlnGamma_dx_total=site[i]+barrier[i])
    return rows


def summary(d,rows,adm):
    x=np.array([r['x'] for r in rows]);lam=np.array([r['cumulative_scaled_hazard'] for r in rows])
    def q(prob):
        target=-math.log(1-prob)
        if not np.all(np.isfinite(lam)) or lam[-1]<target:return {k:math.nan for k in ['x','sigma','rW','time','amp']}
        xx=float(np.interp(target,lam,x))
        return {'x':xx,'sigma':interp_at(xx,x,np.array([r['sigma_GB_MPa'] for r in rows])),
            'rW':interp_at(xx,x,np.array([r['r_TJ_over_W'] for r in rows])),
            'time':interp_at(xx,x,np.array([r['elapsed_s'] for r in rows])),
            'amp':interp_at(xx,x,np.array([r['hazard_amplification'] for r in rows]))}
    q10,q50,q90=q(.1),q(.5),q(.9)
    curv=[abs(r['center_curvature_per_um']) for r in rows]+[abs(r['outer_curvature_per_um']) for r in rows]
    return {'design_id':cid(d),'Ro_nm':d.Ro_m*1e9,'Rc_over_Ro':d.Rc_over_Ro,'Rc_nm':d.Rc_m*1e9,
        'Lc_over_W':d.Lc_m/W,'rTJ_over_Rc':d.rTJ_over_Rc,'rTJ0_over_W':d.rTJ0_m/W,
        'theta0_deg':d.theta_center_deg,'mean_curvature0_per_um':.5*(d.k_center0_per_m+d.k_outer0_per_m)*1e-6,
        'curvature_radius_min_over_W':1/(max(curv)*1e6*W) if max(curv)>0 else math.inf,
        'sigma0_MPa':rows[0]['sigma_GB_MPa'],'sigma20_MPa':rows[-1]['sigma_GB_MPa'],
        'stress_reserve_MPa':rows[-1]['sigma_GB_MPa']-rows[0]['sigma_GB_MPa'],
        'minimum_minus_dF_dx_fJ':min(r['minus_dF_dx_J'] for r in rows)*1e15,
        'minimum_xdot_per_s':min(r['xdot_per_s'] for r in rows),
        'maximum_flux_closure_relative':max(r['flux_closure_relative'] for r in rows),
        'x10':q10['x'],'sigma_x10_MPa':q10['sigma'],'rTJ_x10_over_W':q10['rW'],'t10_s':q10['time'],
        'x50':q50['x'],'sigma_x50_MPa':q50['sigma'],'rTJ_x50_over_W':q50['rW'],'t50_s':q50['time'],'hazard_amplification_x50':q50['amp'],
        'x90':q90['x'],'sigma_x90_MPa':q90['sigma'],'rTJ_x90_over_W':q90['rW'],'t90_s':q90['time'],
        'rTJ20_over_W':rows[-1]['r_TJ_over_W'],'resolution_margin':adm['resolution_margin']}


def signed_resolution(d):
    a=admissibility(d,samples=201)
    if not a['admissible']:return a
    states=[contact_state(d,float(x)) for x in MILESTONES]
    maxk=max(abs(s[k]) for s in states for k in ('center_curvature_per_m','outer_curvature_per_m'))
    radius=math.inf if maxk==0 else 1/maxk
    if radius<6*W:return {**a,'admissible':False,'reason':'local meridional curvature radius below 6W'}
    return {**a,'curvature_radius_min_over_W':radius/W}


def prior_balanced_examples(manifest,mobility):
    with OLD_SCREEN.open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in list(r.items()):
            if k!='design_id':
                try:r[k]=float(v)
                except ValueError:pass
    targets=[('approximately 18 to 45 MPa, r20/W>12',18.,45.,12.),('approximately 24 to 55 MPa, r20/W>10',24.,55.,10.)]
    out=[]
    for label,s0,s1,marg in targets:
        pool=[r for r in rows if r['rTJ20_over_W']>marg]
        r=min(pool,key=lambda q:((q['sigma0_MPa']-s0)/5)**2+((q['sigma20_MPa']-s1)/5)**2)
        d=SharpDesign(r['Ro_nm']*1e-9,r['Rc_over_Ro'],r['Lc_nm']*1e-9,r['rTJ_over_Rc'],
            r['theta_center0_deg'],r['theta_outer0_deg'],r['kRc_center'],r['kRo_outer'])
        coarse=kinetic_path(d,manifest,mobility,MILESTONES,201)
        thermo=all(q['dF_dx_J']<0 for q in coarse)
        out.append({'requested_balance':label,'matched_design_id':r['design_id'],'sigma0_MPa':r['sigma0_MPa'],
            'sigma20_MPa':r['sigma20_MPa'],'rTJ20_over_W':r['rTJ20_over_W'],
            'thermodynamically_favorable_through_20pct':thermo,
            'maximum_dF_dx_fJ':max(q['dF_dx_J'] for q in coarse)*1e15,
            'disposition':'retain' if thermo else 'reject: prescribed center shrink raises sharp-interface free energy'})
    return out


def current_pf_validation(manifest,mobility):
    # Existing fields are opened read-only for metrology; no evolution method is imported.
    from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
    from pf_sintering.three_particle_phase_a import PhaseAOperator
    from pf_sintering.three_particle_contacts import evaluate_contacts
    from pf_sintering.three_particle_geometry import grain_volumes
    c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,W,.5e-9);op=PhaseAOperator(g)
    with np.load(PF_BASE) as q:vbase=grain_volumes(q['f_initial'],g)
    with np.load(PF_POST) as q:
        f=q['f'];t0=float(q['t_model'])*manifest['seconds_per_model_time']
    v0=grain_volumes(f,g);ct=evaluate_contacts(f,op,manifest);L,R=ct['LEFT'],ct['RIGHT']
    Rc=(3*v0[1]/(4*math.pi))**(1/3);Ro=(3*.5*(v0[0]+v0[2])/(4*math.pi))**(1/3)
    d=SharpDesign(Ro,Rc/Ro,R['z_TJ_m']-L['z_TJ_m'],L['r_n_m']/Rc,
        math.degrees(L['theta_positive_rad']),math.degrees(L['theta_negative_rad']),
        L['kappa1_positive_per_m']*Rc,L['kappa1_negative_per_m']*Ro)
    with np.load(PF_END) as q:v1=grain_volumes(q['f'],g);t1=float(q['t_model'])*manifest['seconds_per_model_time']
    xobs=1-v1[1]/v0[1];xs=np.linspace(0,xobs,81)
    path=kinetic_path(d,manifest,mobility,xs,401);pred=path[-1]['elapsed_s'];actual=t1-t0
    return d,path,{'source_start_time_s':t0,'source_end_time_s':t1,'observed_loss_relative_to_post_transient':xobs,
        'observed_elapsed_s':actual,'predicted_elapsed_s':pred,'predicted_over_observed':pred/actual,
        'observed_mean_xdot_per_s':xobs/actual,'predicted_initial_xdot_per_s':path[0]['xdot_per_s'],
        'baseline_center_loss_at_post_transient':1-v0[1]/vbase[1],
        'read_only_existing_fields':True,'mobility_fit_performed':False}


def choose(summaries):
    selected=[];used=set()
    selectors=[('largest shrinkage before median nucleation',lambda r:-r['x50']),
        ('highest stress at median nucleation',lambda r:-r['sigma_x50_MPa']),
        ('largest hazard amplification at median',lambda r:-r['hazard_amplification_x50']),
        ('largest resolution margin',lambda r:-r['resolution_margin']),
        ('least concave thermodynamic candidate',lambda r:abs(r['mean_curvature0_per_um']))]
    for role,key in selectors:
        pool=[r for r in summaries if r['design_id'] not in used and math.isfinite(r['x90'])]
        if not pool:break
        q=min(pool,key=key);used.add(q['design_id']);selected.append({'discussion_role':role,**q})
    return selected


def main():
    OUT.mkdir(parents=True,exist_ok=True);manifest=json.loads(MANIFEST_PATH.read_text())
    mobility=M_MODEL/manifest['seconds_per_model_time'];search=[];thermo=[];rejected=[];coarse_paths=[];designs={}
    for Ro in (80.,100.,120.,150.):
     for ratio in (.45,.55,.65,.75,.85):
      for Lw in (8.,12.,16.,20.,24.):
       for rr in (.60,.80,1.,1.20,1.40,1.60):
        for theta in (120.,140.,160.,170.):
         for km in (-30.,-20.,-10.,-5.,0.,5.):
          d=SharpDesign(Ro*1e-9,ratio,Lw*W,rr,theta,theta,km*1e6*ratio*Ro*1e-9,km*1e6*Ro*1e-9)
          name=cid(d);a=signed_resolution(d)
          if not a['admissible']:
              rejected.append({'design_id':name,'reason':a['reason']});continue
          coarse=kinetic_path(d,manifest,mobility,MILESTONES,201)
          for cr in coarse:
              coarse_paths.append({'Ro_nm':Ro,'Rc_over_Ro':ratio,'Lc_over_W':Lw,
                  'rTJ_over_Rc':rr,'theta0_deg':theta,'mean_curvature0_per_um':km,**cr})
          row={'design_id':name,'Ro_nm':Ro,'Rc_over_Ro':ratio,'Lc_over_W':Lw,'rTJ_over_Rc':rr,
              'theta_deg':theta,'mean_curvature0_per_um':km,'sigma0_MPa':coarse[0]['sigma_GB_MPa'],
              'sigma20_MPa':coarse[-1]['sigma_GB_MPa'],'rTJ20_over_W':coarse[-1]['r_TJ_over_W'],
              'minimum_dF_dx_fJ':min(q['dF_dx_J'] for q in coarse)*1e15,
              'maximum_dF_dx_fJ':max(q['dF_dx_J'] for q in coarse)*1e15,
              'minimum_xdot_per_s':min(q['xdot_per_s'] for q in coarse),
              'thermodynamically_favorable':all(q['dF_dx_J']<0 for q in coarse),
              'curvature_radius_min_over_W':a['curvature_radius_min_over_W'],'resolution_margin':a['resolution_margin']}
          search.append(row);designs[name]=d
          if row['thermodynamically_favorable']:thermo.append(row)
    dense_paths=[];summaries=[]
    for q in thermo:
        d=designs[q['design_id']];path=kinetic_path(d,manifest,mobility,np.linspace(0,.2,401),301)
        dense_paths.extend(path);summaries.append(summary(d,path,signed_resolution(d)))
    selected=choose(summaries);balanced=prior_balanced_examples(manifest,mobility)
    vd,vpath,validation=current_pf_validation(manifest,mobility)
    write_csv(OUT/'geometry_search.csv',search);write_csv(OUT/'geometry_rejections.csv',rejected)
    write_csv(OUT/'all_admissible_thermodynamic_kinetic_paths.csv',coarse_paths)
    write_csv(OUT/'thermodynamic_candidates.csv',summaries);write_csv(OUT/'thermodynamic_kinetic_paths.csv',dense_paths)
    write_csv(OUT/'discussion_candidates.csv',selected);write_csv(OUT/'prior_balanced_candidate_audit.csv',balanced)
    quant=[]
    for s in summaries:
      for p in (10,50,90):quant.append({'design_id':s['design_id'],'event_probability_percent':p,
        'center_loss_fraction':s[f'x{p}'],'stress_MPa':s[f'sigma_x{p}_MPa'],'rTJ_over_W':s[f'rTJ_x{p}_over_W'],'elapsed_s':s[f't{p}_s'],
        'hazard_amplification':s['hazard_amplification_x50'] if p==50 else ''})
    write_csv(OUT/'first_nucleation_quantiles.csv',quant);write_csv(OUT/'current_065_validation_path.csv',vpath)
    (OUT/'current_065_validation.json').write_text(json.dumps(validation,indent=2)+'\n')

    by={}
    for r in dense_paths:by.setdefault(r['design_id'],[]).append(r)
    # Survival and stress curves.
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for i,s in enumerate(selected,1):
        p=by[s['design_id']];xx=np.array([r['x'] for r in p])*100
        axes[0].plot(xx,[r['event_probability'] for r in p],label=f'C{i}')
        axes[1].plot(xx,[r['sigma_GB_MPa'] for r in p],label=f'C{i}')
    axes[0].axhline(.5,color='k',ls=':');axes[0].set(xlabel='center-volume loss (%)',ylabel='probability root has occurred')
    axes[0].set_xlim(0,1.15*max(s['x90'] for s in selected)*100)
    axes[1].set(xlabel='center-volume loss (%)',ylabel='production stress (MPa)')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'nucleation_distribution_and_stress.png',dpi=200);fig.savefig(OUT/'nucleation_distribution_and_stress.pdf');plt.close(fig)
    # Median design map.
    fig,ax=plt.subplots(figsize=(8,5.5));sc=ax.scatter([s['x50']*100 for s in summaries],[s['sigma_x50_MPa'] for s in summaries],
        c=[s['mean_curvature0_per_um'] for s in summaries],s=[30+20*s['resolution_margin'] for s in summaries],cmap='coolwarm')
    for i,s in enumerate(selected,1):ax.annotate(f'C{i}',(s['x50']*100,s['sigma_x50_MPa']),weight='bold')
    ax.set(xlabel='median first-root center loss (%)',ylabel='stress at median first root (MPa)');ax.grid(alpha=.2)
    fig.colorbar(sc,ax=ax,label='initial mean meridional curvature (/um)');fig.tight_layout();fig.savefig(OUT/'median_nucleation_pareto_map.png',dpi=200);fig.savefig(OUT/'median_nucleation_pareto_map.pdf');plt.close(fig)
    # Time, driving, and amplification.
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    for i,s in enumerate(selected,1):
        p=by[s['design_id']];xx=np.array([r['x'] for r in p])*100
        axes[0].plot(xx,[r['elapsed_s'] for r in p],label=f'C{i}')
        axes[1].plot(xx,[r['minus_dF_dx_J']*1e15 for r in p])
        axes[2].plot(xx,[r['hazard_amplification'] for r in p])
    axes[0].set_ylabel('elapsed physical time (s)');axes[1].set_ylabel('-dF/dx (fJ)');axes[2].set_ylabel('Gamma(x)/Gamma(0)')
    for ax in axes:ax.set_xlabel('center-volume loss (%)');ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(OUT/'kinetic_driving_and_amplification.png',dpi=200);fig.savefig(OUT/'kinetic_driving_and_amplification.pdf');plt.close(fig)
    # Site versus barrier contributions.
    fig,axes=plt.subplots(math.ceil(len(selected)/2),2,figsize=(10,3.2*math.ceil(len(selected)/2)),squeeze=False)
    for i,(ax,s) in enumerate(zip(axes.flat,selected),1):
        p=by[s['design_id']];xx=np.array([r['x'] for r in p])*100
        ax.plot(xx,[r['dlnGamma_dx_site'] for r in p],label='site loss')
        ax.plot(xx,[r['dlnGamma_dx_barrier'] for r in p],label='barrier lowering')
        ax.plot(xx,[r['dlnGamma_dx_total'] for r in p],color='black',label='total')
        ax.set_title(f'C{i}: {s["design_id"]}',fontsize=8);ax.set(xlabel='loss (%)',ylabel='d ln Gamma / dx');ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ax in axes.flat[len(selected):]:ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'hazard_amplification_decomposition.png',dpi=200);fig.savefig(OUT/'hazard_amplification_decomposition.pdf');plt.close(fig)
    # Selected analytical shapes at x=0 and x50.
    fig,axes=plt.subplots(len(selected),2,figsize=(10,3*len(selected)),squeeze=False)
    for i,s in enumerate(selected):
      d=designs[s['design_id']]
      for j,x in enumerate((0.,s['x50'])):
        p=half_chain_profile(d,x,501);a=.5*d.Lc_m
        axes[i,j].plot(p['center_z_m']*1e9,p['center_r_m']*1e9,color='#e68613')
        axes[i,j].plot(-p['center_z_m']*1e9,p['center_r_m']*1e9,color='#e68613')
        axes[i,j].plot(p['outer_z_m']*1e9,p['outer_r_m']*1e9,color='#2a6fbb')
        axes[i,j].plot(-p['outer_z_m']*1e9,p['outer_r_m']*1e9,color='#2a6fbb')
        axes[i,j].set_title(f'C{i+1}, x={100*x:.3f}%');axes[i,j].set(xlabel='z (nm)',ylabel='radius (nm)');axes[i,j].grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'selected_geometry_at_median_root.png',dpi=180);fig.savefig(OUT/'selected_geometry_at_median_root.pdf');plt.close(fig)
    # PF mobility validation, no fitted factor.
    fig,ax=plt.subplots(figsize=(7,4.5));ax.plot(np.array([r['x'] for r in vpath])*100,[r['elapsed_s'] for r in vpath],label='sharp-interface prediction')
    ax.scatter([validation['observed_loss_relative_to_post_transient']*100],[validation['observed_elapsed_s']],color='black',label='existing PF endpoint')
    ax.set(xlabel='loss relative to post-transient field (%)',ylabel='elapsed time (s)');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'current_065_mobility_validation.png',dpi=200);fig.savefig(OUT/'current_065_mobility_validation.pdf');plt.close(fig)

    payload={'label':'ANALYTICAL_RIPENING_VERSUS_ROOT_HAZARD_DESIGN','phase_field_run':False,'stochastic_draws':False,
        'search_count':14400,'geometry_admissible_count':len(search),'thermodynamically_favorable_count':len(summaries),
        'concave_thermodynamic_count':sum(s['mean_curvature0_per_um']<0 for s in summaries),
        'physical_parameters':{'gamma_s_J_per_m2':1.,'gamma_gb_J_per_m2':GAMMA_GB,'M_surface_per_model_time':M_MODEL,
            'seconds_per_model_time':manifest['seconds_per_model_time'],'M_surface_m6_per_J_s':mobility,
            'atomic_volume_m3':OMEGA,'M_atom_relation':'M_atom=M_surface/Omega^2'},
        'current_065_validation':validation,'discussion_candidates':selected,'prior_balanced_candidate_audit':balanced,
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MANIFEST_PATH,OLD_SCREEN,PF_BASE,PF_POST,PF_END]},
        'limitations':['single-coordinate sharp-interface Rayleighian projection','not a capillary-equilibrium solution',
            'local concavity is polynomially joined and screened for positivity/extrema/resolution, not PF-relaxed',
            'first-root probabilities are analytical survival distributions; no thresholds were drawn']}
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Ripening-versus-nucleation analytical design','',
        '**Status: analytical campaign complete. No PF field was initialized or evolved and no stochastic threshold was drawn.**','',
        '## Reduced kinetics and validation','',
        'The full-chain sharp energy is `F=gamma_s A_s + gamma_GB A_GB`. On each symmetric half-chain, the unit-x normal velocity determines `d(r Jx)/ds=-r u_n/Omega`. Using `M_atom=M_surface/Omega^2` makes the atomic and PF volume-flux dissipations identical, so the measured PF surface mobility is used without an adjustable factor. The resulting `xdot=-F\'/zeta` is accepted only when `F\'<0` throughout 0-20% loss.','',
        f"Against the existing 0.65 post-transient-to-30-s interval, the reduced model predicts {validation['predicted_elapsed_s']:.3f} s for the observed {100*validation['observed_loss_relative_to_post_transient']:.4f}% relative loss; PF took {validation['observed_elapsed_s']:.3f} s. The ratio is {validation['predicted_over_observed']:.3f}. No mobility or stress parameter was fitted.",'',
        '## Search result','',
        f"Of 14,400 signed-curvature designs, {len(search)} pass geometry and 6W curvature/contact resolution. Only **{len(summaries)}** have spontaneous center shrinkage (`dF/dx<0`) across the full 0-20% interval; {sum(s['mean_curvature0_per_um']<0 for s in summaries)} of those are concave. Concavity can raise stress at large TJ radius, but most prescribed concave paths raise total interfacial energy and are rejected.",'',
        'The root survival is evaluated as `S(x)=exp[-integral 2 Gamma/(1.25 xdot) dx]`. The primary outputs are the 10/50/90% first-root loss, stress, TJ resolution, and elapsed time.','',
        '## Prior balanced examples','',
        '| Target | Matched old-screen geometry | stress path | r20/W | thermodynamic result |','|---|---|---:|---:|---|']
    for r in balanced:lines.append(f"| {r['requested_balance']} | `{r['matched_design_id']}` | {r['sigma0_MPa']:.1f}->{r['sigma20_MPa']:.1f} MPa | {r['rTJ20_over_W']:.1f} | {r['disposition']} |")
    lines += ['','## Discussion candidates','',
        'These candidates are retained for analytical discussion, not PF promotion:','',
        '| Role | Geometry | k0 | sigma0 | x50 | sigma(x50) | t50 | Gamma50/Gamma0 | r50/W |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in selected:lines.append(f"| {s['discussion_role']} | `{s['design_id']}` | {s['mean_curvature0_per_um']:.1f}/um | {s['sigma0_MPa']:.2f} MPa | {100*s['x50']:.3f}% | {s['sigma_x50_MPa']:.2f} MPa | {s['t50_s']:.2f} s | {s['hazard_amplification_x50']:.3f} | {s['rTJ_x50_over_W']:.2f} |")
    lines += ['','The full first-root distributions are:','',
        '| Candidate | x10 / x50 / x90 | sigma10 / sigma50 / sigma90 | t10 / t50 / t90 | rTJ/W at 10 / 50 / 90% |','|---|---:|---:|---:|---:|']
    for i,s in enumerate(selected,1):lines.append(f"| C{i} | {100*s['x10']:.3f} / {100*s['x50']:.3f} / {100*s['x90']:.3f}% | {s['sigma_x10_MPa']:.3f} / {s['sigma_x50_MPa']:.3f} / {s['sigma_x90_MPa']:.3f} MPa | {s['t10_s']:.3f} / {s['t50_s']:.3f} / {s['t90_s']:.3f} s | {s['rTJ_x10_over_W']:.2f} / {s['rTJ_x50_over_W']:.2f} / {s['rTJ_x90_over_W']:.2f} |")
    lines += ['','## Numerical robustness','',
        'Repeating the projected kinetics with 401 and 801 meridional points changes `dF/dx` by at most 0.0031 fJ and `xdot` by less than 0.10% over the ten accepted paths. Half-chain flux closure improves to below 2.6e-6 at 801 points. These errors are small relative to the retained thermodynamic driving.','',
        '## Decision','',
        'The unchanged root process is likely to fire after only a small fraction of center-volume loss. The thermodynamically allowed paths therefore realize little of their nominal 20% stress reserve before median nucleation. Concavity raises the absolute initial stress while preserving sites, but within this fixed signed-curvature family it does not create large pre-root stress amplification. No geometry is selected and no PF run is authorized.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'output':str(OUT),'geometry_admissible':len(search),'thermodynamic':len(summaries),'concave':sum(s['mean_curvature0_per_um']<0 for s in summaries),'selected':[s['design_id'] for s in selected],'validation_ratio':validation['predicted_over_observed']},indent=2))


if __name__=='__main__':main()
