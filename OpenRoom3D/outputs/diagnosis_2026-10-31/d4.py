from rlib import *
import time
g,_=load();cs,K,dist=cams()
W,H=640,360;Ks=K.copy();Ks[:2]*=.5
print("train/eval",sum(c["split"]=="train" for c in cs),sum(c["split"]=="eval" for c in cs),[c["name"][-9:-4] for c in cs if c["split"]=="eval"])
rows=[]
for sp in ("eval","train"):
    v=[]
    for c in [c for c in cs if c["split"]==sp][:: (1 if sp=="eval" else 9)]:
        t=time.time();r=render(g,c["c2w"],Ks,W,H,bg=1.);gt=undist_img(c["name"],K,dist,W,H)
        im=r["rgb"].clamp(0,1).cpu().numpy();mse=((im-gt)**2).mean();p=-10*np.log10(mse);v.append(p)
        if sp=="eval" and len(rows)<3:
            d=r["depth"].cpu().numpy();lo,hi=np.quantile(d,[.02,.98]);dv=cv2.applyColorMap((np.clip((d-lo)/(hi-lo),0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)[:,:,::-1]
            rows.append(np.hstack([gt,im,dv/255.]))
        print(sp,c["name"][-9:-4],round(float(p),2),"frag",r["nfrag"]//10**6,"M",round(time.time()-t,2),"s",flush=True)
    print(sp,"mean PSNR",round(float(np.mean(v)),2),"n",len(v))
cv2.imwrite("eval_gt_render_depth.jpg",(np.vstack(rows)[:,:,::-1]*255).astype(np.uint8))
print("gpu peak GiB",round(torch.cuda.max_memory_allocated()/2**30,2))
