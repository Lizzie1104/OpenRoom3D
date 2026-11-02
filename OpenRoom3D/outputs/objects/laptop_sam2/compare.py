from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys; R=(str(PROJECT_ROOT) + "/"); sys.path[:0]=[R,R+"scripts"]
import numpy as np, torch, cv2
from openroom3d.gaussian_scene import *
from select_by_masks import largest_cluster
RUN=R+"outputs/desk02/feature-splatting/2026-09-22_202843"
g=load_gaussians(RUN);cs,K,dist,_=load_cameras(R+"data/processed/desk02",RUN);Ks=K.copy();Ks[:2]*=.5;W,H=640,360
N=len(g["means"]);sel=torch.zeros(N,dtype=torch.bool,device="cuda");sel[torch.as_tensor(np.load(R+"outputs/objects/laptop_sam2/selection/obj_laptop.npz")["gaussian_index"],device="cuda")]=True
s=np.load(R+"outputs/diagnosis_2026-10-31/scores.npz")["laptop"];x=g["means"].cpu().numpy();i=np.flatnonzero(s>=.5);b=np.zeros(N,bool);b[i[largest_cluster(x[i])]]=True;base=torch.as_tensor(b,device="cuda")
clr=torch.zeros(N,dtype=torch.bool,device="cuda");clr[torch.as_tensor(np.load(R+"outputs/objects/laptop_sam2/inferred_table/inferred_background.npz")["scene_cleared_index"],device="cuda")]=True
rows=[]
for k in [29,39,89]:
    c=cs[k];print(k,c["split"]);f=lambda m:render(g,c["c2w"],Ks,W,H,active=m)["rgb"].clamp(0,1).cpu().numpy()
    gt=cv2.resize(cv2.undistort(cv2.imread(R+"data/processed/desk02/"+c["name"]),K,np.r_[dist,0.]),(W,H),interpolation=cv2.INTER_AREA)[:,:,::-1]/255.
    rows.append(np.hstack([gt,f(base),f(sel),f(~clr)]))
im=(np.vstack(rows)[:,:,::-1].clip(0,1)*255).astype(np.uint8);bar=np.full((44,im.shape[1],3),255,np.uint8)
for i,t in enumerate(["Original photo","BEFORE: text + DBSCAN","AFTER: SAM2 + hull","Scene after removal (no fill)"]): cv2.putText(bar,t,(i*W+14,30),cv2.FONT_HERSHEY_SIMPLEX,.75,(30,30,30),2,cv2.LINE_AA)
cv2.imwrite(R+"outputs/objects/laptop_sam2/selection/heldout_compare_labeled.jpg",np.vstack([bar,im]),[cv2.IMWRITE_JPEG_QUALITY,90])
