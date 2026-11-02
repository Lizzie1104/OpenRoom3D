from rlib import *
g,_=load();cs,K,dist=cams();W,H=640,360;Ks=K.copy();Ks[:2]*=.5
c=[c for c in cs if c["name"][-9:-4]=="00057"][0];r=render(g,c["c2w"],Ks,W,H,bg=1.);gt=undist_img(c["name"],K,dist,W,H);im=r["rgb"].clamp(0,1).cpu().numpy()
print("global gain fit",float((gt*im).sum()/(im*im).sum()),"best shift px",cv2.phaseCorrelate(cv2.cvtColor(gt,cv2.COLOR_RGB2GRAY),cv2.cvtColor(im.astype(np.float32),cv2.COLOR_RGB2GRAY))[0])
cv2.imwrite("frame57.jpg",(np.hstack([gt,im,np.abs(gt-im)*3]).clip(0,1)[:,:,::-1]*255).astype(np.uint8))
