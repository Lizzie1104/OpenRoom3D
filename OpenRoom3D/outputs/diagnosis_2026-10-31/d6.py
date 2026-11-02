from rlib import *
from scipy.spatial import cKDTree
g,_=load();cs,K,dist=cams()
t=json.load(open(R/"data/processed/desk02/transforms.json"))
fr=sorted(t["frames"],key=lambda f:f["file_path"])
P=np.array([f["transform_matrix"] for f in fr],float)
d=json.load(open(RUN/"dataparser_transforms.json"));Ts=np.vstack([np.array(d["transform"]),[0,0,0,1]]);s=d["scale"]
def rot(a,b):
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b);v=np.cross(a,b);c=a@b;S=np.linalg.norm(v)
    if c<-.9999: raise SystemExit
    Vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]);return np.eye(3)+Vx+Vx@Vx*((1-c)/(S*S+1e-12))
tr=P[:,:3,3].mean(0);up=P[:,:3,1].mean(0);Rr=rot(up,np.array([0,0,1.]))
Tn=np.eye(4);Tn[:3,:3]=Rr;Tn[:3,3]=Rr@-tr
O=Tn@P;sn=1/np.abs(O[:,:3,3]).max()
print("recomputed scale",round(sn,5),"saved",round(s,5));print("T recomputed\n",Tn[:3].round(3),"\nT saved\n",Ts[:3].round(3))
print("raw t max abs",np.abs(P[:,:3,3]).max().round(3),"1/saved",round(1/s,3))
L=open(R/"data/processed/desk02/sparse_pc.ply").read().splitlines();i=L.index("end_header")
X=np.array([l.split()[:3] for l in L[i+1:]],float)
m=g["means"].cpu().numpy();tree=cKDTree(m)
print("recomputed align NN",np.median(tree.query((X@Tn[:3,:3].T+Tn[:3,3])*sn)[0]).round(4))
W,H=640,360;Ks=K.copy();Ks[:2]*=.5
ev=[i for i,c in enumerate(cs) if c["split"]=="eval"]
def score(name,fn):
    v=[]
    for i in ev:
        r=render(g,fn(P[i]),Ks,W,H,bg=1.);gt=undist_img(fr[i]["file_path"],K,dist,W,H)
        v.append(-10*np.log10(((r["rgb"].clamp(0,1).cpu().numpy()-gt)**2).mean()))
    print(name,"eval PSNR",round(float(np.mean(v)),2))
def f(Tm,sc):
    def h(p):
        q=Tm@p;q=q.copy();q[:3,3]*=sc;return q
    return h
score("saved T,s",f(Ts,s));score("recomputed T,s",f(Tn,sn));score("identity,saved s",f(np.eye(4),s))
