from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import safe, torch, json, numpy as np, cv2
from pathlib import Path
R=Path(str(PROJECT_ROOT))
RUN=R/"outputs/desk02/feature-splatting/2026-09-22_202843"
def load():
    p=torch.load(RUN/"nerfstudio_models/step-000029999.ckpt",map_location="cuda",weights_only=True)["pipeline"]
    return {k.split("gauss_params.")[1]:v for k,v in p.items() if "gauss_params." in k},p
def cams():
    """Follow nerfstudio: train/eval split by fraction (linspace), poses via dataparser transform."""
    t=json.load(open(R/"data/processed/desk02/transforms.json"))
    d=json.load(open(RUN/"dataparser_transforms.json"))
    fr=sorted(t["frames"],key=lambda f:f["file_path"])
    n=len(fr);ntr=int(np.ceil(n*.9))
    tr=np.linspace(0,n-1,ntr,dtype=int);ev=np.setdiff1d(np.arange(n),tr)
    # saved transform = T_orient @ applied_transform; poses in transforms.json already have it applied
    T=np.vstack([np.array(d["transform"]),[0,0,0,1]])@np.linalg.inv(np.vstack([np.array(t["applied_transform"]),[0,0,0,1]]));s=d["scale"]
    out=[]
    for i,f in enumerate(fr):
        c2w=T @ np.array(f["transform_matrix"]);c2w[:3,3]*=s
        out.append(dict(name=f["file_path"],c2w=c2w,split="train" if i in set(tr.tolist()) else "eval"))
    K=np.array([[t["fl_x"],0,t["cx"]],[0,t["fl_y"],t["cy"]],[0,0,1.]])
    dist=np.array([t["k1"],t["k2"],t["p1"],t["p2"]])
    return out,K,dist
def quat2R(q):
    q=q/q.norm(dim=-1,keepdim=True);w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3)
@torch.no_grad()
def render(g,c2w,K,W,H,extra=None,bg=0.,sig=3.0,maxfrag=120_000_000):
    """Fragment-based alpha compositing of projected Gaussians. Returns rgb,depth,alpha,(extra)."""
    dev=g["means"].device
    c2w=torch.as_tensor(c2w,device=dev,dtype=torch.float32).clone()
    c2w[:3,1:3]*=-1   # OpenGL->OpenCV
    Rc=c2w[:3,:3].T;t=-Rc @ c2w[:3,3]
    x=g["means"]@Rc.T+t;keep=x[:,2]>.01
    Kt=torch.as_tensor(K,device=dev,dtype=torch.float32)
    z=x[keep,2];u=x[keep,0]/z*Kt[0,0]+Kt[0,2];v=x[keep,1]/z*Kt[1,1]+Kt[1,2]
    RR=quat2R(g["quats"][keep]);S=torch.exp(g["scales"][keep])
    C=(Rc @ RR)*S[:,None,:];C=C @ C.transpose(1,2)
    fx,fy=Kt[0,0],Kt[1,1]
    # Jacobian of perspective projection
    xc=x[keep];J=torch.zeros(len(z),2,3,device=dev)
    J[:,0,0]=fx/z;J[:,0,2]=-fx*xc[:,0]/z**2;J[:,1,1]=fy/z;J[:,1,2]=-fy*xc[:,1]/z**2
    C2=J @ C @ J.transpose(1,2);a,b,d=C2[:,0,0]+.09,C2[:,0,1],C2[:,1,1]+.09
    det=(a*d-b*b).clamp_min(1e-9);ia,ib,id_=d/det,-b/det,a/det
    ru=(sig*a.sqrt()).clamp(max=200);rv=(sig*d.sqrt()).clamp(max=200)
    vis=(u+ru>0)&(u-ru<W)&(v+rv>0)&(v-rv<H)
    idx=vis.nonzero()[:,0]
    op=torch.sigmoid(g["opacities"][keep][:,0]);col=torch.sigmoid(g["features_dc"][keep])
    x0=(u-ru).floor().clamp(0,W-1).long();x1=(u+ru).ceil().clamp(0,W-1).long()
    y0=(v-rv).floor().clamp(0,H-1).long();y1=(v+rv).ceil().clamp(0,H-1).long()
    nu=(x1-x0+1);nv=(y1-y0+1);cnt=(nu*nv)[idx]
    PP=[];AA=[];ZZ=[];GG=[]
    pos=0;cum=cnt.cumsum(0)
    while pos<len(idx):
        base=cum[pos-1] if pos else 0
        end=int(torch.searchsorted(cum,base+40_000_000).item());end=max(end,pos+1);end=min(end,len(idx))
        sel=idx[pos:end];c_=cnt[pos:end]
        ga=torch.repeat_interleave(sel,c_)
        off=torch.arange(c_.sum(),device=dev)-torch.repeat_interleave(torch.cat([torch.zeros(1,device=dev,dtype=torch.long),c_.cumsum(0)[:-1]]),c_)
        px=x0[ga]+off%nu[ga];py=y0[ga]+off.div(nu[ga],rounding_mode="floor")
        du=px+.5-u[ga];dv=py+.5-v[ga]
        q=ia[ga]*du*du+2*ib[ga]*du*dv+id_[ga]*dv*dv
        al=op[ga]*torch.exp(-.5*q);m=(q<sig*sig)&(al>1e-3)
        PP.append((py*W+px)[m]);AA.append(al[m].float());ZZ.append(z[ga][m]);GG.append(ga[m])
        pos=end
    P=torch.cat(PP);A=torch.cat(AA);Z=torch.cat(ZZ);G=torch.cat(GG);del PP,AA,ZZ,GG
    o=torch.argsort(Z);o=o[torch.argsort(P[o],stable=True)]
    P,A,Z,G=P[o],A[o],Z[o],G[o]
    st=torch.cat([torch.ones(1,device=dev,dtype=torch.bool),P[1:]!=P[:-1]])
    l=torch.log1p(-A.clamp(max=.99999)).double();cs=l.cumsum(0)
    base=cs-l;gid=st.long().cumsum(0)-1;fst=base[st][gid]
    tr=torch.exp(base-fst).float();w=tr*A
    out={}
    N=W*H
    def acc(val):
        r=torch.zeros(N,val.shape[-1] if val.ndim>1 else 1,device=dev,dtype=torch.float32)
        r.index_add_(0,P,w[:,None]*(val if val.ndim>1 else val[:,None]));return r
    a_=acc(torch.ones_like(A))
    rgb=acc(col[G])+(1-a_)*bg
    dep=acc(Z)/a_.clamp_min(1e-6)
    res=dict(rgb=rgb.reshape(H,W,3),depth=dep.reshape(H,W),alpha=a_.reshape(H,W),nfrag=len(P))
    if extra is not None:
        res["extra"]=acc(extra[keep][G]).reshape(H,W,-1)
    res["weights"]=(P,G,w,keep.nonzero()[:,0])
    return res
def undist_img(f,K,dist,W,H):
    im=cv2.imread(str(R/"data/processed/desk02"/f))
    im=cv2.undistort(im,K,np.r_[dist,0.]);im=cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA)[:,:,::-1]
    return im.astype(np.float32)/255
