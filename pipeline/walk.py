import numpy as np, cv2, sys
from scipy import ndimage as ndi
n=sys.argv[1]; wall=np.load(n+'_wall.npy').astype(np.uint8)
H,W=wall.shape
wall[int(0.79*H):,int(0.81*W):]=0
rows=wall.sum(1)>0.55*W; cols=wall.sum(0)>0.55*H
wall[rows,:]=0; wall[:,cols]=0
el=lambda r:cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
# kill sheet frame + title block: keep only largest connected wall cluster region
big=cv2.dilate(wall,el(15))
lab,nl=ndi.label(big); s=ndi.sum(big,lab,range(1,nl+1))
# frame component = one touching near-border; building = largest component not spanning whole sheet
cands=[]
for i in range(nl):
    sl=ndi.find_objects((lab==i+1).astype(int))[0]
    h=sl[0].stop-sl[0].start; w=sl[1].stop-sl[1].start
    cands.append((s[i], h<0.9*H and w<0.9*W, i+1, sl))
cands=[c for c in cands if c[1]]; cands.sort(key=lambda c:-c[0])
bid=cands[0][2]; sl=cands[0][3]
keep=(lab==bid)
wall=wall*keep
# exterior
sealed=cv2.morphologyEx(wall,cv2.MORPH_CLOSE,el(15))
ext=np.ones_like(wall,bool); ext[sealed>0]=False
el2,_=ndi.label(ext); ext=el2==el2[0,0]
free=(cv2.dilate(wall,el(1))==0)&~ext
# door-sealed walls
import json
D=json.load(open(n+'_meta.json'))['doors']
ws=wall.copy()
for dd in D:
    ex,ey=2*dd['x']-dd['hx'],2*dd['y']-dd['hy']
    cv2.line(ws,(int(round(dd['hx'])),int(round(dd['hy']))),(int(round(ex)),int(round(ey))),1,3)
ws=cv2.morphologyEx(ws,cv2.MORPH_CLOSE,el(3))
fs=free&(cv2.dilate(ws,el(1))==0)
lab,nl=ndi.label(fs); sz=ndi.sum(fs,lab,range(1,nl+1))
order=np.argsort(-sz)
print(n,'comps',nl,[round(sz[i]*0.01) for i in order[:10]])
rng=np.random.default_rng(3); cols=rng.integers(90,230,(nl+1,3)).astype(np.uint8); cols[0]=255
img=cols[lab]; img[free&(lab==0)]=(255,0,0) ; img[wall>0]=0
img[lab==order[0]+1]=(255,140,0)
y0,y1,x0,x1=sl[0].start,sl[0].stop,sl[1].start,sl[1].stop
cv2.imwrite(n+'_comp2.png',img[y0:y1,x0:x1,::-1])
np.savez(n+'_walk.npz',wall=wall,free=free,lab=lab,crop=np.array([y0,y1,x0,x1]),order=order)
