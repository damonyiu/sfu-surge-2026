from config import PDF
import pymupdf as fitz, numpy as np, cv2, sys, json, math
from config import FLOORS, MAT, WALL_LAYERS, layer, PDF
SCALE={k:v['scale'] for k,v in FLOORS.items()}
RES=0.10
def circ(a,b,c):
    ax,ay=a;bx,by=b;cx,cy=c
    d=2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by))
    if abs(d)<1e-9: return None
    ux=((ax*ax+ay*ay)*(by-cy)+(bx*bx+by*by)*(cy-ay)+(cx*cx+cy*cy)*(ay-by))/d
    uy=((ax*ax+ay*ay)*(cx-bx)+(bx*bx+by*by)*(ax-cx)+(cx*cx+cy*cy)*(bx-ax))/d
    return (ux,uy),math.dist((ux,uy),a)
def bez(p,t):
    return tuple((1-t)**3*p[0][k]+3*(1-t)**2*t*p[1][k]+3*(1-t)*t*t*p[2][k]+t**3*p[3][k] for k in (0,1))
def load(n):
    p=fitz.open(PDF(n))[0]; mpp=0.0254/72*SCALE[n]
    z=mpp/RES*2; m=MAT(p,n)
    segs=[]  # (kind, pts, fill, closed)
    arcs=[]
    for d in p.get_drawings():
        if layer(d) not in WALL_LAYERS: continue
        c=d.get('color') or d.get('fill') or (0,0,0)
        if c[0]>0.5: continue
        R=d['rect']*m; PW,PH=p.rect.width,p.rect.height
        if R.width>0.6*PW or R.height>0.6*PH: continue          # sheet frame
        if R.x0>0.8*PW and R.y0>0.75*PH: continue               # title block
        its=d['items']
        if len(its)>=4 and all(it[0]=='l' for it in its) and 'f' not in d['type']:
            # polyline door swings (AQ 1000 draws arcs as short lines; double doors put two arcs in one path)
            chains=[[]]
            for it in its:
                a,b=tuple(it[1]*m),tuple(it[2]*m)
                ch=chains[-1]
                if ch:
                    pa,pb=ch[-1]
                    turn=abs(math.atan2(b[1]-a[1],b[0]-a[0])-math.atan2(pb[1]-pa[1],pb[0]-pa[0]))
                    turn=min(turn,2*math.pi-turn)
                    if math.dist(pb,a)>0.05/mpp or turn>0.5: chains.append([]); ch=chains[-1]
                ch.append((a,b))
            rest=[]; found=0
            for ch in chains:
                ok=False
                if len(ch)>=4:
                    pts=[ch[0][0]]+[q[1] for q in ch]
                    r=circ(pts[0],pts[len(pts)//2],pts[-1])
                    if r and 0.6<=r[1]*mpp<=1.25 and max(abs(math.dist(q,r[0])-r[1]) for q in pts)<0.06*r[1]:
                        ang=abs(math.atan2(pts[0][1]-r[0][1],pts[0][0]-r[0][0])-math.atan2(pts[-1][1]-r[0][1],pts[-1][0]-r[0][0]))
                        ang=min(ang,2*math.pi-ang)
                        if 1.2<ang<1.8:
                            arcs.append({'P':[pts[0],pts[0],pts[-1],pts[-1]],'c':r[0],'r':r[1]}); ok=True; found+=1
                if not ok: rest+=ch
            if found:
                for a,b in rest: segs.append(['l',[a,b],d])
                continue
        for it in d['items']:
            if it[0]=='l': segs.append(['l',[tuple(it[1]*m),tuple(it[2]*m)],d])
            elif it[0]=='c':
                P=[tuple(it[k]*m) for k in (1,2,3,4)]
                r=circ(P[0],bez(P,.5),P[3])
                if r and 0.6<=r[1]*mpp<=1.25 and 'f' not in d['type']:
                    arcs.append({'P':P,'c':r[0],'r':r[1]}); continue
                segs.append(['c',P,d])
            elif it[0]=='re': segs.append(['q',list(it[1].quad*m),d])
            elif it[0]=='qu': segs.append(['q',list(it[1]*m),d])
    # merge arcs sharing a hinge (split arcs)
    doors=[]
    used=[False]*len(arcs)
    for i,a in enumerate(arcs):
        if used[i]: continue
        grp=[a]; used[i]=True
        for j in range(i+1,len(arcs)):
            if not used[j] and math.dist(arcs[j]['c'],a['c'])<0.15/mpp: grp.append(arcs[j]); used[j]=True
        ends=[g['P'][0] for g in grp]+[g['P'][3] for g in grp]
        doors.append({'hinge':a['c'],'r':a['r'],'ends':ends})
    # drop leaf lines: line with an endpoint at a hinge and length ~ r
    keep=[]
    for s in segs:
        if s[0]=='l':
            L=math.dist(*s[1]); drop=False
            for dd in doors:
                if abs(L-dd['r'])<0.25*dd['r'] and min(math.dist(s[1][0],dd['hinge']),math.dist(s[1][1],dd['hinge']))<0.12/mpp:
                    drop=True; dd['leaf']=s[1][1] if math.dist(s[1][0],dd['hinge'])<math.dist(s[1][1],dd['hinge']) else s[1][0]; break
            if drop: continue
        keep.append(s)
    for dd in doors:
        # closed end = arc end farthest from leaf tip
        lt=dd.get('leaf')
        ce=max(dd['ends'],key=lambda e:math.dist(e,lt)) if lt else dd['ends'][0]
        dd['center']=((dd['hinge'][0]+ce[0])/2,(dd['hinge'][1]+ce[1])/2)
        dd['swing']=(lt or dd['ends'][-1])
    out=fitz.open(); q=out.new_page(width=p.rect.width,height=p.rect.height); sh=q.new_shape()
    for k,P,d in keep:
        if k=='l': sh.draw_line(*P)
        elif k=='c': sh.draw_bezier(*P)
        else: sh.draw_quad(fitz.Quad(*P))
        sh.finish(color=(0,0,0),fill=(0,0,0) if 'f' in d['type'] else None,width=0.5,closePath=False)
    sh.commit()
    pix=q.get_pixmap(matrix=fitz.Matrix(z,z),colorspace=fitz.csGRAY)
    a=np.frombuffer(pix.samples,np.uint8).reshape(pix.h,pix.w)
    b=(a<200)[:pix.h//2*2,:pix.w//2*2]
    wall=b.reshape(pix.h//2,2,pix.w//2,2).any(axis=(1,3))  # max-pool so thin glazing lines survive
    s=z/2
    # Pick each door's closed end: the wall continues straight on past the hinge in the
    # closed direction (the door sits in that wall), but not in the open-leaf direction.
    H_,W_=wall.shape
    def hits(hx,hy,ex,ey):
        L=math.hypot(ex-hx,ey-hy) or 1; ux,uy=(ex-hx)/L,(ey-hy)/L; n=0
        for t in np.arange(0.12,0.6,0.06)/mpp:
            X,Y=int(round((hx-ux*t)*s)),int(round((hy-uy*t)*s))
            if 1<=X<W_-1 and 1<=Y<H_-1 and wall[Y-1:Y+2,X-1:X+2].any(): n+=1
        return n
    for dd in doors:
        hx,hy=dd['hinge']; ends=dd['ends']
        sc=[hits(hx,hy,*e) for e in ends]
        ce=ends[int(np.argmax(sc))]
        if max(sc)==0 and dd.get('leaf'): ce=max(ends,key=lambda e:math.dist(e,dd['leaf']))
        op=max(ends,key=lambda e:math.dist(e,ce))
        dd['center']=((hx+ce[0])/2,(hy+ce[1])/2); dd['swing']=op
        # carve the doorway out of the wall raster: plans often draw a thin threshold or
        # frame line straight across the opening, which would otherwise block the door
        L=math.dist((hx,hy),ce) or 1; ux,uy=(ce[0]-hx)/L,(ce[1]-hy)/L; t0=0.08/mpp
        a_=(int(round((hx+ux*t0)*s)),int(round((hy+uy*t0)*s))); b_=(int(round((ce[0]-ux*t0)*s)),int(round((ce[1]-uy*t0)*s)))
        w8=wall.astype(np.uint8); cv2.line(w8,a_,b_,0,7); wall=w8>0
    D=[{'x':dd['center'][0]*s,'y':dd['center'][1]*s,'hx':dd['hinge'][0]*s,'hy':dd['hinge'][1]*s,'sx':dd['swing'][0]*s,'sy':dd['swing'][1]*s} for dd in doors]
    return wall,D,s,mpp
if __name__=='__main__':
    n=sys.argv[1]; wall,D,s,mpp=load(n)
    # building outline (the gross-area polygon): used only to tell inside from outside, never as a wall
    p_=fitz.open(PDF(n))[0]; m_=MAT(p_,n); ol=np.zeros(wall.shape,np.uint8)
    for d in p_.get_drawings():
        if layer(d)!='GROS': continue
        for it in d['items']:
            if it[0]=='l':
                a_,b_=it[1]*m_,it[2]*m_; cv2.line(ol,(int(a_.x*s),int(a_.y*s)),(int(b_.x*s),int(b_.y*s)),1,2)
    np.save(n+'_outline.npy',ol>0)
    np.save(n+'_wall.npy',wall); json.dump({'doors':D,'px_per_pt':s,'m_per_pt':mpp},open(n+'_meta.json','w'))
    print(n,wall.shape,'door arcs',len(D))
