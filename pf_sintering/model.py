from __future__ import annotations

import csv, json, math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from scipy.signal import convolve2d
from skimage.measure import find_contours

Geometry = Literal["substrate", "threeparticle"]
Contact = Literal["short_plane", "long_plane"]

@dataclass
class ModelConfig:
    preset: str = "dev"
    geometry: Geometry = "substrate"
    nx: int | None = None; ny: int | None = None
    dx: float | None = None; r1: float | None = None; r2: float | None = None; r3: float | None = None
    interface_cells: float = 4.0; aspect_ratio: float = 2.0; contact_orientation: Contact = "short_plane"
    substrate_wall_frac: float | None = None; initial_overlap: float | None = None; domain_margin_r2: float = 3.0
    theta_mis_deg: float = 30.0; sigma_target: float = 75e6; temperature: float = 1000.0
    t_total: float | None = None; dt_override: float | None = None; cfl: float = 0.04; seed: int = 42
    save_interval_steps: int | None = None; diag_every_steps: int | None = None; checkpoint_interval_steps: int | None = None
    hazard_every: int = 1; status_prints: bool = True; event_prints: bool = True; checkpoint: bool = True
    output_dir: Path = Path("runs/dev"); out_tag: str = "dev"; restart_file: Path | None = None
    live_plot: bool = False
    use_aniso_surface: bool = True; psi_measure: bool = True

@dataclass
class Params:
    Nx:int; Ny:int; dx:float; R1:float; R2:float; R3:float; Rx:float; Ry:float; geometry:str
    aspect_ratio:float; contact_orientation:str; substrate_wall_frac:float; initial_overlap:float
    T:float; theta_mis_deg:float; sigma_target:float; interface_width:float
    kB:float=1.380649e-23; Rgas:float=8.314; gamma_s:float=1.0; gamma_gb:float=1.0; Omega:float=1e-29; b:float=2.5e-10
    D_gb:float=0.; GS1:float=0.; GS2:float=0.; k_f:float=0.; W_f:float=0.; k_eta:float=0.; W_cpl_f:float=0.; M_f:float=0.; M_eta:float=0.
    tau_target:float=1.; tau_ripening:float=20.; dt:float=1e-5; t_total:float=2e-3; Nt:int=1; CFL:float=.04
    save_interval:int=1; diag_every:int=1; checkpoint_interval:int=1; hazard_every:int=1
    A0:float=0.; V0:float=0.; sigma_ref:float=100e6; tau_ex0:float=0.; phi_site_max:float=1.; q0:float=-.95; h00:float=-.22
    tau_xi:float=0.; lambda_xi:float=0.; m_xi:float=2.; chi_g:float=8.; delta_gb:float=1e-9; tau_g_load:float=2.; tau_g_drain:float=.05
    gamma_gb_ref:float=1.; gamma_gb_floor:float=.05; gamma_gb_ceil:float=1.95; G_shear:float=150e9; nu_poisson:float=.23
    V_solid_initial:float=0.; V1_initial:float=0.; V2_initial:float=0.; V3_initial:float=0.; V2_gone_frac:float=.05
    use_eta3:bool=False; use_aniso_surface:bool=True; aniso_delta:float=.15; theta_grain:np.ndarray=field(default_factory=lambda:np.zeros(3))
    lut_psi:np.ndarray=field(default_factory=lambda:np.zeros(0)); lut_a:np.ndarray=field(default_factory=lambda:np.zeros(0)); lut_ap:np.ndarray=field(default_factory=lambda:np.zeros(0))

@dataclass
class Sink:
    n_d:int=0; phi:float=0.; active:bool=False; hazard:float=0.; threshold:float=0.; cumulative_disp:float=0.; cumulative_strain:float=0.
    current_disp:float=0.; nucleations:int=0; climbs:int=0; q:float=-.95; qbar:float=0.; g_ex:float=0.; r_nuc:float=0.; tau_sink:float=math.inf

@dataclass
class Stress:
    sigma:float=0.; x_neck:float=math.nan; kappa:float=0.; gb_col:float=math.nan; exists:bool=False; GS:float=0.; gamma_gb_eff:float=math.nan


def _preset(name):
    if name=="v64": return dict(dx=3e-9,r1=500e-9,r2=400e-9,r3=500e-9,t_total=2.5)
    if name=="dev": return dict(dx=5e-9,r1=100e-9,r2=80e-9,r3=100e-9,t_total=2e-3)
    raise ValueError("preset must be dev or v64")

def _gb_energy(theta):
    tc=15.; th=max(theta,1e-6); rs=(th/tc)*(1-math.log(th/tc)) if th<tc else 1.
    return max(rs*(1-.35*math.exp(-((theta-45.)/6.)**2)),.05)

def build_params(c:ModelConfig)->Params:
    b=_preset(c.preset); dx=c.dx or b["dx"]; r1=c.r1 or b["r1"]; r2=c.r2 or b["r2"]; r3=c.r3 or b["r3"]; tt=c.t_total if c.t_total is not None else b["t_total"]
    ar=c.aspect_ratio
    if ar<=0: raise ValueError("aspect_ratio must be >0")
    rx,ry=(r2*math.sqrt(ar),r2/math.sqrt(ar)) if c.contact_orientation=="short_plane" else (r2/math.sqrt(ar),r2*math.sqrt(ar))
    margin=round(c.domain_margin_r2*r2/dx); wall_room=round(.30*(2*rx+margin*dx)/dx)
    nx=c.nx or max(64,2*round((wall_room+round(2*rx/dx)+margin)/2)); ny=c.ny or max(64,2*round((round(2*ry/dx)+margin)/2))
    wall=c.substrate_wall_frac if c.substrate_wall_frac is not None else max(.18,wall_room/nx+.02); ov=c.initial_overlap or 5*dx; W=c.interface_cells*dx
    p=Params(nx,ny,dx,r1,r2,r3,rx,ry,c.geometry,ar,c.contact_orientation,wall,ov,c.temperature,c.theta_mis_deg,c.sigma_target,W)
    p.GS1=r1+r2; p.GS2=r2+r3; p.gamma_gb=_gb_energy(c.theta_mis_deg); p.gamma_gb_ref=p.gamma_gb
    p.D_gb=1e-3*math.exp(-1.5e5/(p.Rgas*p.T)); p.k_f=3*p.gamma_s*W; p.W_f=12*p.gamma_s/W; p.k_eta=3*p.gamma_gb*W; p.W_cpl_f=36*p.gamma_gb/W
    p.M_f=(20e-9)**4/(p.tau_target*p.k_f); p.M_eta=p.M_f/p.dx**2*.01; p.CFL=c.cfl; p.dt=min(p.CFL*p.dx**4/(p.M_f*p.k_f),1e-5)
    p.use_eta3=c.geometry=="threeparticle"; p.use_aniso_surface=c.use_aniso_surface; p.theta_grain=np.array([0.,math.radians(c.theta_mis_deg),0.])
    if p.use_aniso_surface:
        psi=np.linspace(0,math.pi/2,4096); d=p.aniso_delta; a=1-d*np.cos(4*psi); ap=4*d*np.sin(4*psi)
        if d>1/15:
            lo,hi=1e-4,math.pi/4-1e-4
            def h(x): return math.tan(x-math.pi/4)+(4*d*math.sin(4*x))/(1-d*math.cos(4*x))
            for _ in range(80):
                m=(lo+hi)/2
                if h(lo)*h(m)<=0: hi=m
                else: lo=m
            pm=(lo+hi)/2; A=(1-d*math.cos(4*pm))/math.cos(pm-math.pi/4); g=(psi>pm)&(psi<math.pi/2-pm); a[g]=A*np.cos(psi[g]-math.pi/4); ap[g]=-A*np.sin(psi[g]-math.pi/4)
        p.lut_psi,p.lut_a,p.lut_ap=psi,a,ap; fac=float(np.max(a*a+np.abs(a*ap))); p.dt/=max(1.,fac)
    if c.dt_override is not None: p.dt=c.dt_override
    p.t_total=tt; p.Nt=max(1,math.ceil(tt/p.dt)); p.hazard_every=max(1,c.hazard_every); p.save_interval=c.save_interval_steps or max(1,round(p.Nt/100)); p.diag_every=c.diag_every_steps or max(1,round(p.Nt/250)); p.checkpoint_interval=c.checkpoint_interval_steps or max(1,round(p.Nt/5))
    p.V0=100*p.Omega; gs=(p.GS1+p.GS2)/2; xd=.5*gs/2; tau=(xd*xd*p.kB*p.T)/(p.sigma_ref*p.Omega*p.D_gb); gamma0=1e12*(p.b**3/gs**3); p.A0=p.kB*p.T*math.log(gamma0*(tau+.25*tau))+p.sigma_target*p.V0; p.tau_ex0=.25*tau; p.tau_xi=.25*tau; p.lambda_xi=.2*p.A0
    return p

def lap9(a,dx):
    E=np.roll(a,1,1);W=np.roll(a,-1,1);N=np.vstack([a[:1],a[:-1]]);S=np.vstack([a[1:],a[-1:]])
    return (4*(N+S+E+W)+np.roll(N,1,1)+np.roll(N,-1,1)+np.roll(S,1,1)+np.roll(S,-1,1)-20*a)/(6*dx*dx)

def grad(a,dx):
    gx=(np.roll(a,-1,1)-np.roll(a,1,1))/(2*dx); gy=np.zeros_like(a); gy[1:-1]=(a[2:]-a[:-2])/(2*dx); gy[0]=(a[1]-a[0])/dx; gy[-1]=(a[-1]-a[-2])/dx; return gx,gy

def div(vx,vy,dx):
    d=(np.roll(vx,-1,1)-np.roll(vx,1,1))/(2*dx); y=np.zeros_like(vy); y[1:-1]=(vy[2:]-vy[:-2])/(2*dx); y[0]=(vy[1]-vy[0])/dx; y[-1]=(vy[-1]-vy[-2])/dx; return d+y

def initialize_fields(p):
    x=(np.arange(1,p.Nx+1)-p.Nx/2)*p.dx; y=(np.arange(1,p.Ny+1)-p.Ny/2)*p.dx; X,Y=np.meshgrid(x,y); W=p.interface_width
    if p.geometry=="substrate":
        wall=(p.substrate_wall_frac-.5)*p.Nx*p.dx; e1=.5*(1-np.tanh((X-wall)/W)); cx=wall+p.Rx-p.initial_overlap; rr=np.sqrt(((X-cx)/p.Rx)**2+(Y/p.Ry)**2); e2=.5*(1-np.tanh((rr-1)*min(p.Rx,p.Ry)/W)); t1=.5*(1-np.tanh((X-wall)/W));t2=.5*(1+np.tanh((X-wall)/W));return np.maximum(e1,e2),e1*t1,e2*t2,np.zeros_like(X)
    c1=-(p.R1+p.R2-p.initial_overlap); c2=0.; c3=p.R2+p.R3-p.initial_overlap; e=[]
    for c,r in ((c1,p.R1),(c2,p.R2),(c3,p.R3)): e.append(.5*(1-np.tanh((np.hypot(X-c,Y)-r)/W)))
    g1=(c1+c2)/2;g2=(c2+c3)/2;t1=.5*(1-np.tanh((X-g1)/W));t2=.5*(1+np.tanh((X-g1)/W))*.5*(1-np.tanh((X-g2)/W));t3=.5*(1+np.tanh((X-g2)/W));return np.maximum.reduce(e),e[0]*t1,e[1]*t2,e[2]*t3

def effective_gamma(s,p): return min(max(p.gamma_gb_ref+max(0,s.g_ex),p.gamma_gb_floor),p.gamma_gb_ceil)
def reproject(f,*etas):
    fb=np.clip(f,0,1); out=[np.clip(e,0,fb) for e in etas]; sm=sum(out); m=sm>fb+1e-12
    if np.any(m):
        sc=fb[m]/sm[m]
        for e in out:e[m]*=sc
    return out

def evolve_f(f,e1,e2,e3,s1,s2,p):
    fb=np.clip(f,0,1); es=[np.clip(e,0,fb) for e in (e1,e2,e3)]; eta2=sum(e*e for e in es); pair=np.maximum(0,es[0]*es[1]); gl=np.full_like(f,p.gamma_gb_ref);m=pair>1e-20;gl[m]=effective_gamma(s1,p); Wc=36*gl/p.interface_width
    mu0=p.W_f*f*(1-f)*(1-2*f)-Wc*eta2*(1-fb)
    if p.use_aniso_surface:
        gx,gy=grad(f,p.dx); sm=sum(es)+1e-30;th0=(es[0]*p.theta_grain[0]+es[1]*p.theta_grain[1]+es[2]*p.theta_grain[2])/sm;th=np.arctan2(gy,gx);ps=np.mod(th-th0,math.pi/2);a=np.interp(ps,p.lut_psi,p.lut_a);ap=np.interp(ps,p.lut_psi,p.lut_ap);flat=gx*gx+gy*gy<(0.01/p.interface_width)**2;a[flat]=1;ap[flat]=0;mu=mu0-div(p.k_f*(a*a*gx-a*ap*gy),p.k_f*(a*a*gy+a*ap*gx),p.dx)
    else: mu=mu0-p.k_f*lap9(f,p.dx)
    M=np.minimum(p.M_f*(16*f*f*(1-f)**2)**2,p.M_f);Mx=.5*(M+np.roll(M,-1,1));Jx=-Mx*(np.roll(mu,-1,1)-mu)/p.dx;Jy=np.zeros_like(f);My=.5*(M[:-1]+M[1:]);Jy[:-1]=-My*(mu[1:]-mu[:-1])/p.dx;Jyd=np.zeros_like(f);Jyd[1:]=Jy[:-1];return f-p.dt*((Jx-np.roll(Jx,1,1))+(Jy-Jyd))/p.dx

def evolve_eta(e1,e2,e3,p):
    e1=e1+p.M_eta*p.dt*p.k_eta*lap9(e1,p.dx);e2=e2+p.M_eta*p.dt*p.k_eta*lap9(e2,p.dx);e3=e3+p.M_eta*p.dt*p.k_eta*lap9(e3,p.dx) if p.use_eta3 else np.zeros_like(e3);return e1,e2,e3

def overlap_col(line):
    a=np.maximum(0,np.asarray(line));m=float(a.max());
    if m<=0:return 0,math.nan
    i=int(np.argmax(a));lo=max(0,i-8);hi=min(a.size,i+9);w=a[lo:hi]**2;c=np.arange(lo,hi)+1;return m,float((c*w).sum()/(w.sum()+1e-30))
def ostwald_substrate(f,e1,e2,e3,p):
    V=float(e2.sum());fb=np.clip(f,0,1);surf=16*fb*fb*(1-fb)**2;ker=np.ones((3,3))/9;cr=p.Ny//2;_,col=overlap_col(e1[cr]*e2[cr]);col=p.substrate_wall_frac*p.Nx if not math.isfinite(col) else col;CC,RR=np.meshgrid(np.arange(1,p.Nx+1),np.arange(1,p.Ny+1));incl=1-np.exp(-.5*(((CC-col)/max(5,round(2*p.interface_width/p.dx)))**2+((RR-(cr+1))/max(5,round(3*p.interface_width/p.dx)))**2));src=convolve2d(surf*e2*incl,ker,mode="same",boundary="symm");snk=convolve2d(surf*e1*incl,ker,mode="same",boundary="symm");a,b=float(src.sum()),float(snk.sum())
    if a<1e-15 or b<1e-15:return f,e1,e2,e3
    tr=min(V*p.dt/p.tau_ripening,.002*V);rem=np.minimum(src/a*tr,.9*e2);act=float(rem.sum());add=snk/(b+1e-30)*act;cap=np.maximum(0,np.minimum(1-f,1-e1));md=float(np.minimum(add,cap).sum());sf=min(1,md/(act+1e-30));rem*=sf;add*=sf;e2-=rem;f-=rem;add=np.minimum(add,np.maximum(0,np.minimum(1-f,1-e1)));e1+=add;f+=add;return f,e1,e2,e3

def contact_width(e1,e2,p):
    w=4*e1*e2; ws=float(w.sum());W=w.sum(1)*p.dx;wm=float(W.max())
    if ws<=0 or wm<=0:return math.nan,math.nan
    x=.5*float(W.sum()*p.dx/wm);cc=np.arange(1,p.Nx+1)[None,:];cx=(float((w*cc).sum()/ws)-p.Nx/2)*p.dx;return x,cx

def curvature(f,col,p):
    pts=[]
    for rc in find_contours(f,.5):pts.append(np.c_[(rc[:,1]+1)*p.dx,(rc[:,0]+1)*p.dx])
    if not pts:return 0.
    P=np.vstack(pts);P=P[np.abs(P[:,0]-col*p.dx)<max(4*p.interface_width,8*p.dx)]
    if len(P)<12:return 0.
    ks=[];ym=P[:,1].mean()
    for Q in (P[P[:,1]>ym],P[P[:,1]<=ym]):
        if len(Q)<6:continue
        x,y=Q[:,0],Q[:,1];x0,y0=x.mean(),y.mean();sc=max(np.max(np.abs(x-x0)),np.max(np.abs(y-y0)))
        if sc<=0:continue
        u,v=(x-x0)/sc,(y-y0)/sc;A=np.c_[2*u,2*v,np.ones_like(u)];q=np.linalg.lstsq(A,u*u+v*v,rcond=None)[0];R2=q[2]+q[0]**2+q[1]**2
        if R2>0:ks.append(1/(math.sqrt(R2)*sc))
    return min(float(np.mean(ks)) if ks else 0.,1/p.interface_width)
def compute_stress(f,e1,e2,e3,s,p):
    cr=p.Ny//2;ov,col=overlap_col(e1[cr]*e2[cr]);
    if ov<1e-3:return Stress(GS=p.GS1),True,"GB absent"
    if p.geometry=="substrate":x,cx=contact_width(e1,e2,p)
    else:
        nc=int(np.clip(round(col)-1,0,p.Nx-1));solid=np.flatnonzero(f[:,nc]>.5);x=.5*(solid[-1]-solid[0])*p.dx if len(solid)>1 else math.nan;cx=math.nan
    if not math.isfinite(x) or x<=0:return Stress(GS=p.GS1),True,"neck unresolved"
    g=effective_gamma(s,p);psi=2*math.acos(np.clip(g/(2*p.gamma_s),-0.999,0.999));k=curvature(f,col,p);st=Stress(2*p.gamma_s*math.sin(psi/2)/x+p.gamma_s*k,x,k,col,True,p.GS1,g)
    if p.geometry=="substrate" and math.isfinite(cx) and cx<(p.substrate_wall_frac-.5)*p.Nx*p.dx-2*p.interface_width:return st,True,"particle burrowed into substrate"
    return st,False,""
def hazard_step(s,st,p,dt,rng):
    if s.threshold<=0:s.threshold=float(rng.exponential())
    sigma=max(0,st.sigma);floor=p.sigma_target/1000
    if st.exists and sigma>=floor:s.r_nuc=1e12*(p.b**3/st.GS**3)*math.exp(-max(0,p.A0-sigma*p.V0)/(p.kB*p.T));xd=.5*st.GS/2;tau=(xd*xd*p.kB*p.T)/(sigma*p.Omega*p.D_gb);s.tau_sink=tau+p.tau_ex0
    else:s.r_nuc=0;s.tau_sink=math.inf
    on=False
    if not s.active:
        s.hazard+=s.r_nuc*dt
        if s.hazard>=s.threshold:s.active=True;s.n_d=1;s.phi=1;s.nucleations+=1;s.hazard=0;s.threshold=float(rng.exponential());on=True
    else:s.phi=1
    gt=p.chi_g*sigma*p.delta_gb*(0 if s.active else 1);tg=p.tau_g_drain if s.active else p.tau_g_load;s.g_ex=max(0,s.g_ex+dt*(gt-s.g_ex)/tg)
    return on
def center(e,p):
    x=(np.arange(1,p.Nx+1)-p.Nx/2)*p.dx;return float((e*x[None,:]).sum()/(e.sum()+1e-30))
def rbm(f,e1,e2,e3,s,p):
    if not s.active or not math.isfinite(s.tau_sink):return f,e1,e2,e3,False
    den=e1+e2+(e3 if p.use_eta3 else 0)+1e-30;vx=(-p.b/s.tau_sink)*(e2/den);vmax=float(np.max(np.abs(vx)))/p.dx
    if vmax<1e-40:return f,e1,e2,e3,False
    n=max(1,math.ceil(p.dt*vmax/.4));ds=p.dt/n;b=center(e2,p)
    for _ in range(n):
        vr=.5*(vx+np.roll(vx,-1,1));fr=np.maximum(vr,0)*f+np.minimum(vr,0)*np.roll(f,-1,1);f=f-ds*(fr-np.roll(fr,1,1))/p.dx
        for e in (e1,e2,e3):
            fw=(np.roll(e,-1,1)-e)/p.dx;bw=(e-np.roll(e,1,1))/p.dx;e-=ds*vx*np.where(vx>=0,bw,fw);np.clip(e,0,1,out=e)
        ex=np.maximum(0,f-1);f=np.clip(f,0,1);surf=16*f*f*(1-f)**2;ss=surf.sum();
        if ex.sum()>0 and ss>0:f+=surf/ss*ex.sum()
    d=max(0,b-center(e2,p));s.current_disp+=d;s.cumulative_disp+=d;s.cumulative_strain=s.cumulative_disp/p.GS1
    if s.current_disp>=p.b:s.active=False;s.n_d=0;s.phi=0;s.current_disp=0;s.climbs+=1;s.hazard=0;return f,e1,e2,e3,True
    return f,e1,e2,e3,False

class SinteringModel:
    def __init__(self,c:ModelConfig):
        self.c=c;self.p=build_params(c);self.rng=np.random.default_rng(c.seed);self.c.output_dir=Path(c.output_dir);self.c.output_dir.mkdir(parents=True,exist_ok=True);self.diag=[]
        if c.restart_file:self.load(Path(c.restart_file))
        else:
            self.f,self.e1,self.e2,self.e3=initialize_fields(self.p);self.step0=1;self.s=Sink(threshold=float(self.rng.exponential()));self._set_initial_volumes()
        self.st=Stress(GS=self.p.GS1)
    def _set_initial_volumes(self):
        p=self.p;p.V_solid_initial=float(self.f.sum()*p.dx**2);p.V1_initial=float(self.e1.sum()*p.dx**2);p.V2_initial=float(self.e2.sum()*p.dx**2);p.V3_initial=float(self.e3.sum()*p.dx**2)
    def save(self,path,step):
        with h5py.File(path,"w") as h:
            for n,a in (("f",self.f),("eta1",self.e1),("eta2",self.e2),("eta3",self.e3)):h.create_dataset(n,data=a,compression="gzip",compression_opts=1)
            h.attrs["step"]=step;h.attrs["params"]=json.dumps(_json(self.p));h.attrs["sink"]=json.dumps(asdict(self.s));h.attrs["rng"]=json.dumps(self.rng.bit_generator.state)
        (self.c.output_dir/"latest_restart.txt").write_text(str(Path(path).resolve())+"\n")
    def load(self,path):
        with h5py.File(path,"r") as h:
            self.f=h["f"][:];self.e1=h["eta1"][:];self.e2=h["eta2"][:];self.e3=h["eta3"][:];self.step0=int(h.attrs["step"])+1;old=json.loads(h.attrs["params"])
            for k in ("Nx","Ny","dx","R1","R2","R3","aspect_ratio","contact_orientation","geometry"):
                if old[k]!=getattr(self.p,k):raise ValueError(f"restart mismatch: {k}")
            for k in ("V_solid_initial","V1_initial","V2_initial","V3_initial"):setattr(self.p,k,old[k])
            self.s=Sink(**json.loads(h.attrs["sink"]));self.rng.bit_generator.state=json.loads(h.attrs["rng"])
    def run(self):
        p=self.p;last=self.step0-1;stop=False;reason=""
        if self.c.status_prints:print(f"grid={p.Nx}x{p.Ny} dx={p.dx*1e9:.2f} nm R2={p.R2*1e9:.1f} nm Rx/Ry={p.Rx*1e9:.1f}/{p.Ry*1e9:.1f} nm\ndt={p.dt:.3e}s Nt={p.Nt} target={p.sigma_target/1e6:g} MPa")
        for t in range(self.step0,p.Nt+1):
            last=t;self.f=evolve_f(self.f,self.e1,self.e2,self.e3,self.s,Sink(),p);self.e1,self.e2,self.e3=reproject(self.f,self.e1,self.e2,self.e3)
            if p.geometry=="substrate":self.f,self.e1,self.e2,self.e3=ostwald_substrate(self.f,self.e1,self.e2,self.e3,p)
            self.e1,self.e2,self.e3=evolve_eta(self.e1,self.e2,self.e3,p);self.e1,self.e2,self.e3=reproject(self.f,self.e1,self.e2,self.e3)
            if t==1 or (t-1)%p.hazard_every==0:
                self.st,stop,reason=compute_stress(self.f,self.e1,self.e2,self.e3,self.s,p)
                if not stop:
                    on=hazard_step(self.s,self.st,p,p.dt if t==1 else p.dt*p.hazard_every,self.rng)
                    if on and self.c.event_prints:print(f"t={t*p.dt*1e3:.3f} ms activation sigma={self.st.sigma/1e6:.1f} MPa")
            if self.s.active:self.f,self.e1,self.e2,self.e3,done=rbm(self.f,self.e1,self.e2,self.e3,self.s,p)
            if t==1 or t%p.diag_every==0:self.diag.append(dict(step=t,time_s=t*p.dt,sigma_Pa=self.st.sigma,x_neck_m=self.st.x_neck,hazard=self.s.hazard,r_nuc=self.s.r_nuc,strain=self.s.cumulative_strain,V2_ratio=float(self.e2.sum()*p.dx**2/p.V2_initial),Vsolid_ratio=float(self.f.sum()*p.dx**2/p.V_solid_initial)))
            if t==1 or t%p.save_interval==0:
                np.savez_compressed(self.c.output_dir/f"sintering_{self.c.out_tag}_frame{t:09d}.npz",f=self.f.astype("f4"),eta1=self.e1.astype("f4"),eta2=self.e2.astype("f4"),eta3=self.e3.astype("f4"),step=t,dt=p.dt)
                if self.c.status_prints:print(f"t={t*p.dt*1e3:.3f} ms sigma={self.st.sigma/1e6:.1f} MPa V2/V20={self.e2.sum()*p.dx**2/p.V2_initial:.4f} Vs/Vs0={self.f.sum()*p.dx**2/p.V_solid_initial:.8f}")
            if self.c.checkpoint and t%p.checkpoint_interval==0:self.save(self.c.output_dir/f"sintering_{self.c.out_tag}_ckpt_step{t:09d}.h5",t)
            if stop:break
        if self.diag:
            with (self.c.output_dir/f"sintering_{self.c.out_tag}_diagnostics.csv").open("w",newline="") as fh:w=csv.DictWriter(fh,fieldnames=self.diag[0]);w.writeheader();w.writerows(self.diag)
        final=self.c.output_dir/f"sintering_{self.c.out_tag}_final.h5";self.save(final,last);out=dict(final_step=last,final_time_s=last*p.dt,stop_requested=stop,stop_reason=reason,nucleations_gb1=self.s.nucleations,climbs_gb1=self.s.climbs,strain_gb1=self.s.cumulative_strain,Vsolid_ratio=float(self.f.sum()*p.dx**2/p.V_solid_initial),final_file=str(final));(self.c.output_dir/f"sintering_{self.c.out_tag}_summary.json").write_text(json.dumps(out,indent=2));return out

def _json(p):
    d=asdict(p)
    for k,v in list(d.items()):
        if isinstance(v,np.ndarray):d[k]=v.tolist()
        elif isinstance(v,np.generic):d[k]=v.item()
    return d

laplacian_9pt = lap9
