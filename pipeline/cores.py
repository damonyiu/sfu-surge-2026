"""Find elevator shafts (box with an X through it) and stair symbols (UP/DN text or
runs of parallel tread lines) on each floor, in page points (rotated view)."""
import pymupdf as fitz, math, json, sys
from config import PDF, FLOORS, MAT
def xboxes(n):
    p=fitz.open(PDF(n))[0]; m=MAT(p,n); mpp=0.0254/72*FLOORS[n]['scale']
    diag=[]
    for d in p.get_drawings():
        c=d.get('color') or (0,0,0)
        if c[0]>0.5: continue
        for it in d['items']:
            if it[0]!='l': continue
            a,b=it[1]*m,it[2]*m; dx,dy=b.x-a.x,b.y-a.y; L=math.hypot(dx,dy)*mpp
            if 1.2<L<4.5 and 0.35<abs(dx)/(abs(dy)+1e-9)<2.8: diag.append((a,b))
    out=[]
    for i,(a,b) in enumerate(diag):
        for c_,d_ in diag[i+1:]:
            # same bounding box, opposite slopes
            r1=fitz.Rect(a,b).normalize(); r2=fitz.Rect(c_,d_).normalize()
            if max(abs(r1.x0-r2.x0),abs(r1.x1-r2.x1),abs(r1.y0-r2.y0),abs(r1.y1-r2.y1))<0.6 and \
               (b.x-a.x)*(b.y-a.y)*(d_.x-c_.x)*(d_.y-c_.y)<0:
                out.append({'x':(r1.x0+r1.x1)/2,'y':(r1.y0+r1.y1)/2,'w':r1.width*mpp,'h':r1.height*mpp})
    # dedupe
    ded=[]
    for o in out:
        if all(math.hypot(o['x']-q['x'],o['y']-q['y'])>2 for q in ded): ded.append(o)
    words=[(w[4],(w[0]+w[2])/2,(w[1]+w[3])/2) for w in p.get_text('words')]
    stairs=[]
    for t,x,y in words:
        if t in('UP','DN','DOWN'):
            P=fitz.Point(x,y)*m
            stairs.append({'x':P.x,'y':P.y,'t':t})
    return ded,stairs
if __name__=='__main__':
    res={}
    for n in FLOORS:
        e,s=xboxes(n); res[n]={'xbox':e,'stairtext':s}; print(n,'xboxes',len(e),'stair text',len(s))
    json.dump(res,open('cores.json','w'))
