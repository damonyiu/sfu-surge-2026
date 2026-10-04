import numpy as np, cv2, sys, json, base64, re, math
from config import PDF
import pymupdf as fitz
from scipy import ndimage as ndi
n=sys.argv[1]
from config import FLOORS as _F
NAME=_F[n]['name']
d=np.load(n+'_walk.npz'); free,lab,wall=d['free'],d['lab'],d['wall']; y0,y1,x0,x1=[int(v) for v in d['crop']]
G=json.load(open(n+'_graph.json')); doors=G['doors']; deg={int(k):v for k,v in G['deg'].items()}
labels=json.load(open(n+'_labels.json'))
from config import FLOORS
FLOOR=FLOORS[n]['floor']
def fix(t):
    t=t.replace('V','7').replace('R','2').replace('X','1')
    head=re.match(r'^[A-Z0-9]+',t).group(0); tail=t[len(head):]
    conf='3568BSG'
    if len(head)==5 and head[0] in conf and head[1] in conf: head=head[1:]
    if head and head[0] in conf: head=FLOOR+head[1:]
    head=head[0]+re.sub('[SB]','8' if FLOOR=='8' else '3',head[1:]) if head else head
    return head+tail
meta=json.load(open(n+'_meta.json'))
# component -> label (from OCR point inside comp)
clab={}
for L in labels:
    t=fix(L['t'])
    if t in('2222',) or not re.fullmatch(r'\d{3,4}(\.\d)?[A-Z]?',t) or t==FLOOR+'000': continue
    yy,xx=int(L['y']),int(L['x'])
    c=lab[yy,xx] if 0<=yy<lab.shape[0] and 0<=xx<lab.shape[1] else 0
    if c==0:
        win=lab[max(yy-6,0):yy+7,max(xx-6,0):xx+7]; v=win[win>0]
        c=np.bincount(v).argmax() if v.size else 0
    if c: clab.setdefault(int(c),t)
corr=lambda c: deg.get(c,0)>=3
# downsample to 0.2 m
F=2
def pool(a): 
    h,w=a.shape[0]//F*F,a.shape[1]//F*F
    return a[:h,:w].reshape(h//F,F,w//F,F).mean((1,3))
fr=free[y0:y1,x0:x1].astype(float)
walk=pool(fr)>=0.4
# cost: near walls + rooms penalty
dist=ndi.distance_transform_edt(walk)
cost=1+np.clip(3-dist,0,3)*1.5
labc=lab[y0:y1,x0:x1][:walk.shape[0]*F:F,:walk.shape[1]*F:F]
roommask=np.isin(labc,[c for c in np.unique(labc) if c and not corr(int(c))])
cost[roommask]*=6
costb=np.where(walk,np.clip(np.round(cost),1,250),0).astype(np.uint8)
H,W=costb.shape
# doors -> dots
dots=[]
for i,o in enumerate(doors):
    x=(o['x']-x0)/F; y=(o['y']-y0)/F
    if not(0<=x<W and 0<=y<H): continue
    a,b=o['a'],o['b']
    room=None
    for c in (a,b):
        if c and not corr(c): room=c
    if room is None and a and b: room=o.get('swing_into')
    name=clab.get(room) if room else None
    kind='room'
    if (bool(a)!=bool(b)): kind='exit'
    dots.append({'id':i,'x':round(x,1),'y':round(y,1),'label':name or '','kind':kind})
# disambiguate duplicate labels
seen={}
for dt in dots:
    if dt['label']:
        seen[dt['label']]=seen.get(dt['label'],0)+1
        if seen[dt['label']]>1: dt['label']+=f" (door {seen[dt['label']]})"
# background render
p=fitz.open(PDF(n))[0]; s=meta['px_per_pt']
BZ=2
pix=p.get_pixmap(matrix=fitz.Matrix(s*BZ,s*BZ),colorspace=fitz.csGRAY)
full=np.frombuffer(pix.samples,np.uint8).reshape(pix.h,pix.w)
img=full[y0*BZ:y1*BZ,x0*BZ:x1*BZ]
img=np.where(img<235,img,255).astype(np.uint8)
_,png=cv2.imencode('.png',img,[cv2.IMWRITE_PNG_COMPRESSION,9])
out={'name':NAME,'res':0.2,'w':W,'h':H,'cost':base64.b64encode(costb.tobytes()).decode(),
     'bg':'data:image/png;base64,'+base64.b64encode(png).decode(),'bgW':img.shape[1],'bgH':img.shape[0],
     'dots':dots,'cropCells':[x1-x0,y1-y0]}
json.dump(out,open(n+'_floor.json','w'))
print(n,W,H,'dots',len(dots),'labelled',sum(1 for x in dots if x['label']),'bg',img.shape,'KB',len(json.dumps(out))//1024)
