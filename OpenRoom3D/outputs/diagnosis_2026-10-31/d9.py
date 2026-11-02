from rlib import *
from sklearn.cluster import DBSCAN
g,_=load();cs,K,dist=cams();dev="cuda"
W,H=640,360;Ks=K.copy();Ks[:2]*=.5
d=np.load("scores.npz");s=torch.tensor(d["laptop"],device=dev);h=torch.tensor(d["h"],device=dev)
pl=np.load("plane.npz");n=torch.tensor(pl["n"],device=dev,dtype=torch.float32);d0=float(pl["d"])
m=g["means"];S=torch.exp(g["scales"]);sm,_=S.sort(1);ani=sm[:,2]/sm[:,0].clamp_min(1e-9)
# worst training frames
psn=[]
for c in cs:
    r=render(g,c["c2w"],Ks,W,H,bg=1.);gt=undist_img(c["name"],K,dist,W,H)
    psn.append(-10*np.log10(((r["rgb"].clamp(0,1).cpu().numpy()-gt)**2).mean()))
psn=np.array(psn);o=psn.argsort()[:6];print("worst frames",[(cs[i]["name"][-9:-4],cs[i]["split"],round(float(psn[i]),1)) for i in o])
# laptop 3D region from high-confidence core
idx=(s>=.8).nonzero()[:,0];xy=m[idx].cpu().numpy();lab=DBSCAN(eps=.015,min_samples=8).fit_predict(xy)
b=np.bincount(lab[lab>=0]).argmax();core=idx[torch.tensor(lab==b,device=dev)];print("s>=.8",len(idx),"largest cluster",len(core))
lo=torch.quantile(m[core],.005,dim=0)-.02;hi=torch.quantile(m[core],.995,dim=0)+.02
ext=(hi-lo).tolist();print("laptop aabb extent units",[round(v,3) for v in ext])
inbox=((m>lo)&(m<hi)).all(1)
for th in(.5,.6,.7,.8):
    q=s>=th;print(f"th {th}: selected {int(q.sum())}, inside laptop box {int((q&inbox).sum())} ({(q&inbox).sum().item()/q.sum().item()*100:.1f}%)")
print("box Gaussians",int(inbox.sum()),"with s>=.5",int((inbox&(s>=.5)).sum()))
# rendered-weight attribution on laptop pixels
tot=bel=tbl=need=0.;H_b=[];H_t=[]
for c in [c for c in cs if c["split"]=="eval"]:
    r=render(g,c["c2w"],Ks,W,H,bg=1.,extra=s[:,None].float())
    P,G,w,kp=r["weights"];gi=kp[G]
    e=(r["extra"][...,0]/r["alpha"].clamp_min(1e-3)).reshape(-1)
    # project box membership to pixels: pixel counts as laptop if s high and most weight is in box
    wb=torch.zeros(W*H,device=dev).index_add_(0,P,w*inbox[gi].float());lap=(e>.6)&(wb>.5)
    f=lap[P];tot+=w[f].sum().item();bel+=(w[f]*(h[gi[f]]<-.008).float()).sum().item()
    need+=(w[f]*(ani[gi[f]]>100).float()).sum().item()
    u,v=torch.meshgrid(torch.arange(W,device=dev),torch.arange(H,device=dev),indexing="xy")
    dc=torch.stack([(u+.5-Ks[0,2])/Ks[0,0],(v+.5-Ks[1,2])/Ks[1,1],torch.ones_like(u)],-1).float()
    c2=torch.as_tensor(c["c2w"],device=dev,dtype=torch.float32).clone();c2[:3,1:3]*=-1
    X=(dc*r["depth"][...,None])@c2[:3,:3].T+c2[:3,3];hh=(X@n-d0).reshape(-1)
    H_b.append(hh[lap&(hh<.04)])
    rgb=r["rgb"].reshape(-1,3);X_=X.reshape(-1,3)
    near=(X_-X_.new_tensor(lo.tolist())).norm(dim=1)>-1
    tan=(rgb[:,0]>rgb[:,2]+.12)&(rgb[:,0]>.6)&(hh.abs()<.06)&~lap
    H_t.append(hh[tan])
print(f"laptop pixels: weight from Gaussians >0.008 below table {bel/tot*100:.1f}%, from needles(aniso>100) {need/tot*100:.1f}%")
a=torch.cat(H_t)[:2000000];b_=torch.cat(H_b)[:2000000]
print("table-coloured surface height q05/25/50/75/95",[round(v,4) for v in torch.quantile(a,torch.tensor([.05,.25,.5,.75,.95],device=dev)).tolist()])
print("laptop base surface height q05/25/50/75/95",[round(v,4) for v in torch.quantile(b_,torch.tensor([.05,.25,.5,.75,.95],device=dev)).tolist()])
# Gaussians straddling table plane
sig=torch.sqrt(((quat2R(g["quats"])*S[:,None,:])**2).sum(2)[:,:]@torch.ones(3,device=dev)*0+((n[None,None,:]@(quat2R(g["quats"])*S[:,None,:])).squeeze(1)**2).sum(1))
bx=inbox&(s>=.5);print("in-box selected: straddle table (|h|<2*normal sigma)",int((bx&(h.abs()<2*sig)).sum()),"/",int(bx.sum()),"; below table",int((bx&(h<-.008)).sum()))
print("needles in box",round((ani[inbox]>100).float().mean().item()*100,1),"%; eval metric sharpness; gaussians in box",int(inbox.sum()))
