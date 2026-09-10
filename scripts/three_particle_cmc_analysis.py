"""Scientific plots/movie and joint qualification from the recorded PF trajectories."""
from pathlib import Path
import sys,json,csv
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.animation import FuncAnimation,PillowWriter
import numpy as np
from pf_sintering.three_particle_phase_a import PhaseAOperator
from three_particle_cmc_ripening import load_initial

OUT=Path('docs/three_particle/cmc/analysis');OUT.mkdir(parents=True,exist_ok=True)
COLORS=['#277da8','#d58b19','#8547a6']

def load_rows(case):
    root=Path('runs/three_particle_cmc/ripening_'+case)
    with (root/'history.csv').open() as stream:rows=[{k:float(v) if v not in ['True','False'] else v=='True' for k,v in r.items()} for r in csv.DictReader(stream)]
    return root,rows,json.loads((root/'report.json').read_text())

def normal_projected_mu(case):
    """Integrate mu df through each radial interface, then average far from GBs.

    For a tanh profile, integral(mu*|df/dn| dn)=gamma*K in the thin-interface
    limit. Neither the value at f=.5 (about 1.5 gamma K) nor the q-weighted
    average (about 1.2 gamma K) is silently labelled the sharp chemical potential.
    """
    root,rows,report=load_rows(case);g,_=load_initial(case);op=PhaseAOperator(g)
    with np.load(root/'frames.npz') as data:frames=data['f']
    result=[];z=g['z'];L=g['gb'][1];W=op.W
    for f in frames:
        mu=op.potential(f);weights=np.maximum(-np.gradient(f,g['dr'],axis=1),0.)
        denominator=np.sum(weights,axis=1)
        projected=np.divide(np.sum(mu*weights,axis=1),denominator,out=np.full(len(z),np.nan),where=denominator>1.)
        # Only complete radial interfaces, with a resolved solid core.
        full=(f[:,0]>.999)&np.isfinite(projected)
        masks=[full&(z<-L-2*W),full&(abs(z)<L-2*W),full&(z>L+2*W)]
        result.append([float(np.mean(projected[m])) for m in masks])
    return np.array(result)

def main():
    datasets={name:load_rows(name) for name in ['unequal','null','equal','unequal_off']}
    mu={name:normal_projected_mu(name) for name in ['unequal','null','equal']}
    root,rows,report=datasets['unequal'];t=np.array([r['t_s'] for r in rows])*1e6
    figures=[]
    def chart(filename,title,series,ylabel):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for label,values,color,style in series:ax.plot(t,values,label=label,color=color,ls=style)
        ax.set(xlabel='Physical time (µs)',ylabel=ylabel,title=title);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/(filename+'.png'),dpi=155);figures.append(fig)
    def data(key):return np.array([r[key] for r in rows])
    chart('grain_volumes','Native PF surface ripening: normalized grain volumes',[(n,data('V_'+n+'_m3')/rows[0]['V_'+n+'_m3'],c,'-') for n,c in zip(['left','center','right'],COLORS)],'V / V₀')
    chart('equivalent_radii','Current volume-equivalent radii',[(n,data('R_'+n+'_m')*1e9,c,'-') for n,c in zip(['left','center','right'],COLORS)],'Radius (nm)')
    for name,suffix,ylabel,scale in [('neck_radii','neck_r_m','Radius (nm)',1e9),('tj_axial_positions','tj_z_m','z (nm)',1e9),('gb_positions','gb_z_m','z (nm)',1e9),('neck_curvature','kappa_per_m','K (1/nm)',1e-9),('cannon_carter_force','CC_force_N','Force (nN)',1e9),('cannon_carter_stress','CC_stress_Pa','Stress (MPa)',1e-6),('pf_geometric_stress','PF_geometric_stress_Pa','Stress (MPa)',1e-6),('dihedral_angles','psi_deg','Angle (degrees)',1)]:
        chart(name,name.replace('_',' ').title(),[(side,data(side+'_'+suffix)*scale,c,style) for side,c,style in [('LEFT',COLORS[0],'-'),('RIGHT',COLORS[2],'--')]],ylabel)
    chart('chemical_potentials','Normal-projected PF chemical potential, excluding 2W around GBs',[(n,mu['unequal'][:,i]/1e6,c,'-') for i,(n,c) in enumerate(zip(['left','center','right'],COLORS))],'∫ μ df (MPa)')
    chart('total_volume_error','Global material conservation',[('relative error',data('total_volume_m3')/rows[0]['total_volume_m3']-1,COLORS[0],'-')],'Vtotal / Vtotal,0 − 1')
    chart('mirror_error','Mirror symmetry before any stochastic event',[('max |f(z,r) − f(−z,r)|',data('mirror_error'),COLORS[0],'-')],'Maximum field difference')
    with np.load(root/'profiles.npz') as p:
        fig,axes=plt.subplots(2,1,figsize=(9,7))
        for i in [0,len(t)//2,len(t)-1]:
            valid=p['r'][i]>3*report['initial_checkpoint']['width_m']
            axes[0].plot(p['z'][valid]*1e9,p['kappa_smooth'][i,valid]*1e-9,label=f'{t[i]:.1f} µs')
            axes[1].plot(p['z'][valid]*1e9,p['mu_surface'][i,valid]/1e6,label=f'{t[i]:.1f} µs')
        axes[0].set(ylabel='K (1/nm)',title='Full free-surface curvature (fixed W-scale smoothing; poles excluded)')
        axes[1].set(ylabel='PF μ at f=0.5 (MPa)',xlabel='z (nm)',title='Local PF field potential; distinct from normal-projected grain potential')
        for ax in axes:
            for b in [-report['initial_checkpoint']['half_length_m'],report['initial_checkpoint']['half_length_m']]:ax.axvline(b*1e9,color='gray',ls=':')
            ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(OUT/'surface_profiles.png',dpi=155);figures.append(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,c in zip(['unequal','null','equal'],COLORS):
        _,rr,dd=datasets[name];tt=np.array([x['t_s'] for x in rr])*1e6
        change=np.array([x['V_center_m3']/rr[0]['V_center_m3']-1 for x in rr])
        axes[0].plot(tt,change*1e6,label=name,color=c)
        axes[1].plot(tt,(mu[name][:,1]-.5*(mu[name][:,0]+mu[name][:,2]))/1e6,label=name,color=c)
    axes[0].set(xlabel='Physical time (µs)',ylabel='Center-volume change (ppm)',title='Primary null vs size/coordination comparisons')
    axes[1].set(xlabel='Physical time (µs)',ylabel='Projected μC − μouter (MPa)',title='No fitted transfer or sign correction')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'controls.png',dpi=155);figures.append(fig)
    with PdfPages(OUT/'phase_a_short_interval_plots.pdf') as pdf:
        for fig in figures:pdf.savefig(fig);plt.close(fig)
    with np.load(root/'frames.npz') as saved:frames=saved['f'];phi=saved['ownership'];z=saved['z'];rad=saved['r'];times=saved['t_s']
    base=np.einsum('kzr,kc->zrc',phi,np.array([[.15,.49,.67],[.92,.65,.24],[.15,.49,.67]]))
    def rgb(f):
        a=base*f[:,:,None]+1-f[:,:,None]
        return np.concatenate([a[:,::-1],a],axis=1).transpose(1,0,2)
    fig,ax=plt.subplots(figsize=(9,5));im=ax.imshow(rgb(frames[0]),origin='lower',extent=[z[0]*1e9,z[-1]*1e9,-rad[-1]*1e9,rad[-1]*1e9],aspect='equal');title=ax.set_title('')
    ax.set(xlabel='z (nm)',ylabel='r (nm)');fig.tight_layout()
    def update(i):
        im.set_data(rgb(frames[i]));title.set_text(f'True-scale PF morphology | t={times[i]*1e6:.1f} µs | ΔVc={(rows[i]["V_center_m3"]/rows[0]["V_center_m3"]-1)*1e6:.3f} ppm');return im,title
    anim=FuncAnimation(fig,update,frames=len(frames),interval=300,blit=False)
    anim.save(OUT/'morphology.gif',writer=PillowWriter(fps=3));plt.close(fig)
    summary={name:dict(volume_changes=d[2]['grain_volume_relative_changes'],total_volume_error=d[2]['total_volume_error'],wall_s=d[2]['wall_s'],t_model=d[2]['t_model'],t_s=d[2]['t_s'],projected_mu_initial_Pa=mu[name][0].tolist() if name in mu else None,projected_mu_final_Pa=mu[name][-1].tolist() if name in mu else None) for name,d in datasets.items()}
    unequal=datasets['unequal'][2];null=datasets['null'][2];off=datasets['unequal_off'][2]
    v0=np.array([unequal['initial']['V_'+n+'_m3'] for n in ['left','center','right']]);dv=v0*np.array(unequal['grain_volume_relative_changes'])
    gates=dict(unequal_center_loses=bool(dv[1]<0),both_outers_gain=bool(dv[0]>0 and dv[2]>0),mass_closes=abs(unequal['total_volume_error'])<1e-10,mirror=unequal['final']['mirror_error']<1e-10,matched_null_negligible=abs(null['grain_volume_relative_changes'][1])<1e-3*abs(unequal['grain_volume_relative_changes'][1]),transport_off_stationary=all(x==0 for x in off['grain_volume_relative_changes']),restart_roundtrip=all(d[2]['restart_roundtrip_exact'] for d in datasets.values()))
    result=dict(status='PASS_CMC_INITIALIZATION_AND_SHORT_INTERVAL_RIPENING' if all(gates.values()) else 'SHORT_INTERVAL_GATE_FAILED',full_phase_a_qualified=False,phase_b_authorized=False,gates=gates,cases=summary,unequal_absolute_grain_volume_changes_m3=dv.tolist(),outer_gain_vs_half_center_loss=(dv[[0,2]]/(-.5*dv[1])).tolist(),scope='Short deterministic interval. Sustained ripening-driven stress loading and long-time performance are not qualified.')
    (OUT/'joint_qualification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
