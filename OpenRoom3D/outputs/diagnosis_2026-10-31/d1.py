from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import safe
import torch, numpy as np, json
from pathlib import Path
R=Path(str(PROJECT_ROOT))
try:
    c=torch.load(R/"outputs/desk02/feature-splatting/2026-09-22_202843/nerfstudio_models/step-000029999.ckpt",map_location="cuda",weights_only=True)
    print("weights_only load ok")
except Exception as e:
    print("WEIGHTS_ONLY FAILED",str(e)[:300]); raise SystemExit
print(c.keys(),c.get("step"))
p=c["pipeline"]
for k,v in p.items():
    if hasattr(v,"shape") and ("gauss" in k or "feature_mlp" in k): print(k,tuple(v.shape))
g={k.split("gauss_params.")[1]:v for k,v in p.items() if "gauss_params." in k}
x=g["means"];s=torch.exp(g["scales"]);o=torch.sigmoid(g["opacities"]).squeeze(-1)
N=len(x);print("N",N)
q=lambda t:[round(v,5) for v in torch.quantile(t.float()[torch.randperm(len(t))[:500000]],torch.tensor([.01,.1,.5,.9,.99],device=t.device)).tolist()]
print("xyz min",x.min(0).values.tolist(),"max",x.max(0).values.tolist())
print("xyz abs q90/99",q(x.abs().max(1).values))
print("opacity q",q(o),"frac<0.1",(o<.1).float().mean().item(),"frac<0.5",(o<.5).float().mean().item())
sm,_=s.sort(1)
print("scale max q",q(sm[:,2]));print("scale min q",q(sm[:,0]))
r=sm[:,2]/sm[:,0].clamp_min(1e-9);print("aniso max/min q",q(r),"frac>10",(r>10).float().mean().item(),"frac>100",(r>100).float().mean().item())
print("flat (min/mid<0.2)",(sm[:,0]/sm[:,1]<.2).float().mean().item())
f=g["distill_features"];print("feat std per dim",[round(v,3) for v in f.std(0).tolist()])
