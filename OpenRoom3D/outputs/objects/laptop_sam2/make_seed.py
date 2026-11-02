from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys; sys.path.insert(0,str(PROJECT_ROOT))
import numpy as np
from openroom3d.gaussian_scene import load_gaussians
from sklearn.cluster import DBSCAN
R=(str(PROJECT_ROOT) + "/")
g=load_gaussians(R+"outputs/desk02/feature-splatting/2026-09-22_202843")
s=np.load(R+"outputs/diagnosis_2026-10-31/scores.npz")["laptop"];x=g["means"].cpu().numpy()
i=np.flatnonzero(s>=.8);l=DBSCAN(eps=.015,min_samples=8,n_jobs=-1).fit_predict(x[i]);k=np.bincount(l[l>=0]).argmax();j=i[l==k]
np.savez(R+"outputs/objects/laptop_sam2/seed.npz",xyz=x[j],gaussian_index=j);print("seed",len(j))
