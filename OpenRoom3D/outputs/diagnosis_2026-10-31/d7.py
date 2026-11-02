from rlib import *
g,_=load();cs,K,dist=cams()
W,H=640,360;Ks=K.copy();Ks[:2]*=.5
res={}
rows=[]
for sp in("eval","train"):
    v=[]
    for c in [c for c in cs if c["split"]==sp]:
        r=render(g,c["c2w"],Ks,W,H,bg=1.);gt=undist_img(c["name"],K,dist,W,H)
        im=r["rgb"].clamp(0,1).cpu().numpy();v.append(-10*np.log10(((im-gt)**2).mean()))
        if sp=="eval" and c["name"][-9:-4] in("00010","00040","00090"):
            d=r["depth"].cpu().numpy();lo,hi=np.quantile(d,[.02,.98]);dv=cv2.applyColorMap((np.clip((d-lo)/(hi-lo),0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)[:,:,::-1]
            rows.append(np.hstack([gt,im,dv/255.]))
    print(sp,"PSNR mean",round(float(np.mean(v)),2),"min",round(float(np.min(v)),2),"max",round(float(np.max(v)),2),"n",len(v))
cv2.imwrite("eval_gt_render_depth.jpg",(np.vstack(rows)[:,:,::-1]*255).astype(np.uint8))
# table plane from sparse-free dense sampling: RANSAC on opaque tan Gaussians
m=g["means"];o=torch.sigmoid(g["opacities"])[:,0];c=torch.sigmoid(g["features_dc"])
print("rgb sample ok")
tan=(c[:,0]>c[:,2]+.1)&(c[:,0]>.5)&(o>.5)&(m.norm(dim=1)<1.0)
p=m[tan];print("tan candidates",len(p))
best=(0,None)
torch.manual_seed(0)
for _ in range(4000):
    i=torch.randint(len(p),(3,));n=torch.linalg.cross(p[i[1]]-p[i[0]],p[i[2]]-p[i[0]]);nn=n.norm()
    if nn<1e-6: continue
    n=n/nn;e=(p-p[i[0]])@n;k=(e.abs()<.004).sum().item()
    if k>best[0]: best=(k,(n.clone(),(p[i[0]]@n).item()))
k,(n,d0)=best
e=(p@n-d0);inl=p[e.abs()<.004]
cen=inl.mean(0);U,S,V=torch.pca_lowrank(inl-cen,q=3);n=V[:,2];d0=(cen@n).item()
if (n*torch.tensor([0,0,1.],device=n.device)).sum()<0:n=-n;d0=-d0
print("plane n",n.tolist(),"d",round(d0,4),"inliers",len(inl),"pca stds",(S/len(inl)**.5).tolist())
dist_all=m@n-d0
np.savez("plane.npz",n=n.cpu().numpy(),d=d0)
print("signed dist quantiles of tan set",[round(v,4) for v in torch.quantile(e[:400000],torch.tensor([.05,.25,.5,.75,.95],device=e.device)).tolist()])
# per-pixel: rendered depth residual on the table
resid=[]
for cc in [c for c in cs if c["split"]=="eval"]:
    r=render(g,cc["c2w"],Ks,W,H,bg=1.)
    u,v=torch.meshgrid(torch.arange(W,device=n.device),torch.arange(H,device=n.device),indexing="xy")
    dirc=torch.stack([(u+.5-Ks[0,2])/Ks[0,0],(v+.5-Ks[1,2])/Ks[1,1],torch.ones_like(u)],-1).float()
    c2w=torch.as_tensor(cc["c2w"],device=n.device,dtype=torch.float32).clone();c2w[:3,1:3]*=-1
    X=(dirc*r["depth"][...,None])@c2w[:3,:3].T+c2w[:3,3]
    h=(X@n-d0);rgb=r["rgb"];mk=(rgb[...,0]>rgb[...,2]+.1)&(rgb[...,0]>.55)&(h.abs()<.05)
    resid.append(h[mk])
h=torch.cat(resid);print("rendered-surface height above fitted table (tan pixels): std",round(h.std().item(),5),"q05,q50,q95",[round(v,5) for v in torch.quantile(h[:1000000],torch.tensor([.05,.5,.95],device=h.device)).tolist()])
print("frac Gaussians below plane >0.01 near-scene",((dist_all<-.01)&(m.norm(dim=1)<1)).float().mean().item())
