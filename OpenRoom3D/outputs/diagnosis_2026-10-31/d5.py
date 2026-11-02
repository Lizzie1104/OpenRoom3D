from rlib import *
from scipy.spatial import cKDTree
g,_=load();cs,K,dist=cams()
d=json.load(open(RUN/"dataparser_transforms.json"));T=np.vstack([np.array(d["transform"]),[0,0,0,1]]);s=d["scale"]
print("det R",np.linalg.det(T[:3,:3]),"RRt-I",np.abs(T[:3,:3]@T[:3,:3].T-np.eye(3)).max())
C=np.array([c["c2w"][:3,3] for c in cs]);print("cam center mean",C.mean(0).round(3),"max abs",np.abs(C).max().round(3))
L=open(R/"data/processed/desk02/sparse_pc.ply").read().splitlines();i=L.index("end_header")
P=np.array([l.split()[:3] for l in L[i+1:]],float)
Q=(P@T[:3,:3].T+T[:3,3])*s
m=g["means"].cpu().numpy();t=cKDTree(m)
print("A dataparser (T then scale) NN dist median",np.median(t.query(Q)[0]).round(4))
Q2=(P@T[:3,:3].T*s+T[:3,3]);print("B scale then T",np.median(t.query(Q2)[0]).round(4))
Q3=P*s;print("C scale only",np.median(t.query(Q3)[0]).round(4))
print("cam-to-sparse dist: cam Q med",np.median(np.linalg.norm(Q,axis=1)).round(3),"means med",np.median(np.linalg.norm(m,axis=1)).round(3))
# look direction sanity: do cameras look at scene centre?
for c in cs[::25]:
    v=-c["c2w"][:3,2];o=c["c2w"][:3,3];print(c["name"][-9:-4],"cam->centre cos",round(float(v@(np.median(Q,0)-o)/np.linalg.norm(np.median(Q,0)-o)),3))
