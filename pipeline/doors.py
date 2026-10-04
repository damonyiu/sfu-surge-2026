import numpy as np, cv2, sys, json, math
n=sys.argv[1]; d=np.load(n+'_walk.npz'); free,lab,wall=d['free'],d['lab'],d['wall']
D=json.load(open(n+'_meta.json'))['doors']
H,W=lab.shape
def comp_at(x,y,r=6):
    best=None
    for rr in range(0,r+1):
        for dy in range(-rr,rr+1):
            for dx in range(-rr,rr+1):
                yy,xx=int(round(y+dy)),int(round(x+dx))
                if 0<=yy<H and 0<=xx<W and lab[yy,xx]>0: return int(lab[yy,xx])
    return 0
out=[]
for dd in D:
    vx,vy=dd['x']-dd['hx'],dd['y']-dd['hy']; L=math.hypot(vx,vy) or 1
    nx,ny=-vy/L,vx/L
    a=comp_at(dd['x']+nx*5,dd['y']+ny*5,3); b=comp_at(dd['x']-nx*5,dd['y']-ny*5,3)
    # swing side
    sx,sy=dd['sx']-dd['x'],dd['sy']-dd['y']
    swing_a = (sx*nx+sy*ny)>0
    dd.update(a=a,b=b,swing_into=a if swing_a else b)
    out.append(dd)
ok=[o for o in out if o['a'] and o['b'] and o['a']!=o['b']]
deg={}
for o in ok:
    for c in (o['a'],o['b']): deg[c]=deg.get(c,0)+1
sz=dict(zip(*np.unique(lab,return_counts=True)))
print(n,'doors',len(out),'linking',len(ok),'one-sided',sum(1 for o in out if bool(o['a'])!=bool(o['b'])))
json.dump({'doors':out,'deg':deg},open(n+'_graph.json','w'))
y0,y1,x0,x1=d['crop']
rng=np.random.default_rng(5); cols=rng.integers(120,235,(lab.max()+1,3)).astype(np.uint8); cols[0]=255
img=cols[lab]; img[wall>0]=0
for o in out:
    c=(0,170,0) if o in ok else (0,0,230)
    cv2.circle(img,(int(o['x']),int(o['y'])),4,c,-1)
cv2.imwrite(n+'_graph.png',img[y0:y1,x0:x1])
