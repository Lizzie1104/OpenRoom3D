import cv2, numpy as np, glob, json
from PIL import Image
fs=sorted(glob.glob("../../data/processed/desk02/images/*.png"))
L=[];B=[]
for f in fs:
    g=cv2.imread(f,0);L.append(cv2.Laplacian(g,cv2.CV_64F).var());B.append(g.mean())
L=np.array(L);B=np.array(B)
print("n",len(fs),"shape",cv2.imread(fs[0]).shape)
print("laplacian var q10/50/90",np.quantile(L,[.1,.5,.9]).round(1),"min",L.min().round(1),"worst frames",[fs[i][-9:-4] for i in L.argsort()[:8]])
print("brightness min/med/max",B.min().round(1),np.median(B).round(1),B.max().round(1))
t=json.load(open("../../data/processed/desk02/transforms.json"))
P=np.array([f['transform_matrix'] for f in t['frames']]);c=P[:,:3,3];st=np.linalg.norm(np.diff(c,axis=0),axis=1)
print("cam center range",c.min(0).round(2),c.max(0).round(2),"step med/max",np.median(st).round(3),st.max().round(3),"max step at",st.argmax())
idx=np.linspace(0,len(fs)-1,16).astype(int)
S=Image.new("RGB",(4*400,4*225))
for i,k in enumerate(idx):
    S.paste(Image.open(fs[k]).convert("RGB").resize((400,225)),((i%4)*400,(i//4)*225))
S.save("contact.jpg",quality=85)
