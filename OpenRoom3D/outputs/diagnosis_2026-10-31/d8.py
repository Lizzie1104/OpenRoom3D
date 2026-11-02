from pathlib import Path
from rlib import *
import open_clip, torch.nn.functional as F
g,p=load();cs,K,dist=cams()
dev="cuda"
clip,_,_=open_clip.create_model_and_transforms("ViT-L-14-336",pretrained=str(Path(__file__).resolve().parents[3] / "training-assets/models/clip/open_clip_model.safetensors"),device=dev);tok=open_clip.get_tokenizer("ViT-L-14-336");clip.eval()
@torch.no_grad()
def txt(l):
    e=clip.encode_text(tok(l).to(dev)).float();return e/e.norm(dim=-1,keepdim=True)
Wh=p["_model.feature_mlp.hidden_conv.weight"].view(64,-1);Bh=p["_model.feature_mlp.hidden_conv.bias"]
Wc=p["_model.feature_mlp.feature_branch_dict.samclip.weight"].view(768,-1);Bc=p["_model.feature_mlp.feature_branch_dict.samclip.bias"]
@torch.no_grad()
def dec(f):
    return F.normalize(F.linear(F.relu(F.linear(f,Wh,Bh)),Wc,Bc),dim=-1)
fe=torch.cat([dec(x) for x in g["distill_features"].split(65536)])
torch.save(fe.half().cpu(),"gauss_clip.pt")
scene=["laptop","table","floor","chair","bottle","computer mouse","notebook","wall","backpack","cable","glass door","keyboard"]
S=txt(scene);obj=txt(["object"])
pl=np.load("plane.npz");n=torch.tensor(pl["n"],device=dev,dtype=torch.float32);d0=float(pl["d"])
m=g["means"];h=m@n-d0;o=torch.sigmoid(g["opacities"])[:,0]
def proj(q):
    return ((fe@txt([q]).T)/.05).sub(((fe@obj.T)/.05)).sigmoid()[:,0]   # 2-way softmax == project score
closed=(fe@S.T/.05).softmax(1)
for q in ["laptop","keyboard","bottle","computer mouse","cup","banana","elephant"]:
    s=proj(q);sel=s>=.5
    line=f"{q:15s} raw>=.5 {int(sel.sum()):6d} ({sel.float().mean()*100:4.1f}%) mean {s.mean():.3f}"
    if q in scene:
        c=closed[:,scene.index(q)];cs_=c>=.5;line+=f" | closed>=.5 {int(cs_.sum()):6d}  overlap {int((sel&cs_).sum())}"
    print(line)
print("similarity spread: max cos to any scene word, q10/50/90",[round(v,3) for v in torch.quantile((fe@S.T).max(1).values,torch.tensor([.1,.5,.9],device=dev)).tolist()])
print("laptop cos q50 on selected vs others")
W,H=640,360;Ks=K.copy();Ks[:2]*=.5
vs=[c for c in cs if c["name"][-9:-4] in("00020","00045","00095")]
rows=[]
for c in vs:
    gt=undist_img(c["name"],K,dist,W,H);row=[gt]
    for sc in [proj("laptop")[:,None],closed[:,[0]],proj("cup")[:,None]]:
        r=render(g,c["c2w"],Ks,W,H,extra=sc.float(),bg=1.)
        e=(r["extra"][...,0]/r["alpha"].clamp_min(1e-3)).clamp(0,1).cpu().numpy()
        hm=cv2.applyColorMap((e*255).astype(np.uint8),cv2.COLORMAP_JET)[:,:,::-1]/255.
        row.append(.45*gt+.55*hm)
    rows.append(np.hstack(row))
cv2.imwrite("semantic_views.jpg",(np.vstack(rows)[:,:,::-1]*255).astype(np.uint8))
# contamination near the table: selected 'laptop' Gaussians whose centre is at table height and away from main body
s=proj("laptop");sel=(s>=.5)
print("laptop raw: |h|<.005 at table",int((sel&(h.abs()<.005)).sum()),"below table(<-.005)",int((sel&(h<-.005)).sum()),"above .005",int((sel&(h>.005)).sum()))
np.savez("scores.npz",laptop=s.cpu().numpy(),closed_laptop=closed[:,0].cpu().numpy(),h=h.cpu().numpy())
