from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import safe, torch
c=torch.load(PROJECT_ROOT / "data/processed/desk02/feature_splatting_samclip_features.pt",map_location="cpu",weights_only=True,mmap=True)
print({k:(v if k=='args' else type(v)) for k,v in c.items()})
fn=c['image_fnames'];print(len(fn),fn[0].name,fn[89].name,fn[90].name,fn[-1].name)
for k,v in c['feature_dict'].items():
    z=(v==0).all(1).float().mean((1,2));print(k,tuple(v.shape),v.dtype,'zero frac mean',round(z.mean().item(),4),'max',round(z.max().item(),4),'all-zero imgs',(z>.999).sum().item())
