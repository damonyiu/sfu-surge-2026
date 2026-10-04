from config import PDF
import pymupdf as fitz, numpy as np, cv2, sys, json, math
from config import FLOORS
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
    z=mpp/RES*2; m=p.rotation_matrix
    segs=[]  # (kind, pts, fill, closed)
    arcs=[]
    for d in p.get_drawings():
        c=d.get('color') or d.get('fill') or (0,0,0)
        if c[0]>0.5: continue
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
    wall=cv2.resize((a<200).astype(np.uint8),(pix.w//2,pix.h//2),interpolation=cv2.INTER_AREA)>0
    s=z/2
    D=[{'x':dd['center'][0]*s,'y':dd['center'][1]*s,'hx':dd['hinge'][0]*s,'hy':dd['hinge'][1]*s,'sx':dd['swing'][0]*s,'sy':dd['swing'][1]*s} for dd in doors]
    return wall,D,s,mpp
if __name__=='__main__':
    n=sys.argv[1]; wall,D,s,mpp=load(n)
    np.save(n+'_wall.npy',wall); json.dump({'doors':D,'px_per_pt':s,'m_per_pt':mpp},open(n+'_meta.json','w'))
    print(n,wall.shape,'door arcs',len(D))
