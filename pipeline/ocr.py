import tempfile, os
from config import PDF
import pymupdf as fitz, numpy as np, cv2, sys, json, subprocess, re
n=sys.argv[1]; p=fitz.open(PDF(n))[0]; m=p.rotation_matrix
meta=json.load(open(n+'_meta.json')); s=meta['px_per_pt']
Z=int(sys.argv[2]) if len(sys.argv)>2 else 6
out=fitz.open(); q=out.new_page(width=p.rect.width,height=p.rect.height); sh=q.new_shape()
for d in p.get_drawings():
    c=d.get('color') or (0,0,0)
    if c[0]>0.5 or d['type']!='s' or (d.get('width') or 0)>0.3: continue
    r=d['rect']*m
    if r.width>25 or r.height>25: continue   # glyphs are small
    for it in d['items']:
        if it[0]=='l': sh.draw_line(it[1]*m,it[2]*m)
        elif it[0]=='c': sh.draw_bezier(*[it[k]*m for k in (1,2,3,4)])
    sh.finish(color=(0,0,0),width=float(sys.argv[3]) if len(sys.argv)>3 else 0.35)
sh.commit()
import itertools
W,H=q.rect.width,q.rect.height; T=3000/Z; O=40
rows=[]
for tx in np.arange(0,W,T-O):
  for ty in np.arange(0,H,T-O):
    clip=fitz.Rect(tx,ty,min(tx+T,W),min(ty+T,H))
    pix=p.get_pixmap(matrix=fitz.Matrix(Z,Z),colorspace=fitz.csGRAY,clip=clip)
    a=np.frombuffer(pix.samples,np.uint8).reshape(pix.h,pix.w)
    if a.min()>128: continue
    a=np.where(a<110,0,255).astype(np.uint8)
    cv2.imwrite(os.path.join(tempfile.gettempdir(),f'tile_{n}.png'),a)
    r=subprocess.run(['tesseract',os.path.join(tempfile.gettempdir(),f'tile_{n}.png'),'-','--psm','11','-c','tessedit_char_whitelist=0123456789.ABCDEFGHJKLMNPRSTUVWXYZ','tsv'],capture_output=True,text=True).stdout
    for line in r.splitlines()[1:]:
        f=line.split('\t')
        if len(f)>=12: f[6]=str(int(f[6])+tx*Z); f[7]=str(int(f[7])+ty*Z); rows.append('\t'.join(f))
labels=[]
for line in rows:
    f=line.split('\t')
    if len(f)<12 or not f[11].strip(): continue
    t=f[11].strip(); conf=float(f[10])
    if not re.fullmatch(r'[A-Z]{0,4}\d{2,5}(\.\d{1,2})?[A-Z]?',t) or conf<40: continue
    x=(float(f[6])+int(f[8])/2)/Z*s; y=(float(f[7])+int(f[9])/2)/Z*s
    labels.append({'t':t,'x':x,'y':y,'conf':conf})
import collections
ded={}
for l in labels: ded[(l['t'],round(l['x']/3),round(l['y']/3))]=l
labels=list(ded.values())
print(n,len(labels),[l['t'] for l in labels][:60])
json.dump(labels,open(n+'_labels.json','w'))
