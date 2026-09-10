"""Publication-ready diagnostics of the shared-physics viability screen."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
D=Path('docs/three_particle/production_screen')

def main():
    rows=json.loads((D/'screen.json').read_text())['candidates']
    with (D/'preserved_unequal_history.csv').open() as f:
        history=[{k:float(v) for k,v in r.items() if v not in ['True','False']} for r in csv.DictReader(f)]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    with PdfPages(D/'screen_plots.pdf') as pdf:
        fig,ax=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
        for row in rows:
            if 'initial' not in row:continue
            c=row['initial']['LEFT'];ok=row['status']=='SCREENED';marker='o' if ok else 'x';color='tab:blue' if ok else 'tab:red'
            x=row['ratio'];args=dict(marker=marker,color=color)
            for axis,y in zip(ax.flat,[row['outer_radius_nm'],row['cmc']['center_span_over_W'],c['sigma_local_Pa']/1e6,c['constant_state_mean_wait_s']]):axis.scatter(x,y,**args)
        for axis,label in zip(ax.flat,['Outer radius (nm)','Center thickness / W','LEFT production local stress (MPa)','Per-contact constant-state mean wait (s)']):
            axis.set(xlabel='Center / outer radius ratio',ylabel=label);axis.grid(alpha=.2)
        ax[0,1].axhline(8,color='k',ls='--',lw=1);ax[1,1].set_yscale('log')
        fig.suptitle('Mapped CMC + identical cleanup; blue: accepted, red: rejected\nStress and waiting time use the live bicrystal law')
        pdf.savefig(fig);fig.savefig(D/'screen_geometry.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
        t=np.array([r['time_s'] for r in history])
        for name,ls in [('LEFT','-'),('RIGHT','--')]:
            ax[0,0].plot(t,[r[name+'_sigma_local_Pa']/1e6 for r in history],ls,label=name)
            ax[0,1].plot(t,[r[name+'_sigma_integral_continuous_Pa']/1e6 for r in history],ls,label=name)
            ax[1,0].plot(t,[r[name+'_hazard_sparse_quadrature'] for r in history],ls,label=name)
        ax[1,1].plot(t,[(r['center_volume_m3']/history[0]['center_volume_m3']-1)*100 for r in history],color='tab:purple')
        for axis,label in zip(ax.flat,['Production local stress (MPa)','Production continuous-integral stress (MPa)','Integrated root rate H (sparse estimate)','Center volume change (%)']):
            axis.set(xlabel='Physical time (s)',ylabel=label);axis.grid(alpha=.2)
        ax[0,0].legend();fig.suptitle('Preserved unequal 0.70 trajectory: new metrology only\nNo events or strain were synthesized; LEFT/RIGHT local curves overlap')
        pdf.savefig(fig);fig.savefig(D/'preserved_unequal.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for row in rows:
            if 'derivatives' not in row:continue
            for name,marker in [('LEFT','o'),('RIGHT','x')]:
                ax[0].scatter(row['ratio'],row['derivatives'][name]['sigma_local_Pa']/1e6,marker=marker)
                ax[1].scatter(row['ratio'],row['derivatives'][name]['root_rate_per_s'],marker=marker)
        for axis,label in zip(ax,['Local stress derivative (MPa/s)','Root-rate derivative (1/s²)']):
            axis.set(xlabel='Center / outer radius ratio',ylabel=label,yscale='symlog');axis.grid(alpha=.2)
        fig.suptitle('1000 native steps immediately after cleanup\nEarly relaxation slopes must not be extrapolated to a root event')
        pdf.savefig(fig);fig.savefig(D/'short_native_slopes.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
        for path,color in [(D/'continuation_0.65_119.999.json','tab:blue'),(D/'continuation_0.74_81.850.json','tab:orange')]:
            data=json.loads(path.read_text());h=data['history'];t=[r['time_s'] for r in h];label=f"ratio {data['ratio']:.2f}"
            ax[0,0].plot(t,[r['contacts']['LEFT']['sigma_local_Pa']/1e6 for r in h],color=color,label=label)
            ax[0,1].plot(t,[r['contacts']['LEFT']['sigma_integral_continuous_Pa']/1e6 for r in h],color=color,label=label)
            ax[1,0].plot(t,[r['hazard']['LEFT']+r['hazard']['RIGHT'] for r in h],color=color)
            ax[1,1].plot(t,[100*r['center_relative_change'] for r in h],color=color)
        for axis,label in zip(ax.flat,['Contact local stress (MPa)','Contact continuous-integral stress (MPa)','Combined hazard H_LEFT + H_RIGHT','Center volume change (%)']):
            axis.set(xlabel='Physical time (s)',ylabel=label);axis.grid(alpha=.2)
        ax[0,0].legend();fig.suptitle('Guarded shortlist continuations; corrected bicrystal coordinate indexing\nLEFT and RIGHT coincide; no stochastic events enabled')
        pdf.savefig(fig);fig.savefig(D/'shortlist_continuations.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
