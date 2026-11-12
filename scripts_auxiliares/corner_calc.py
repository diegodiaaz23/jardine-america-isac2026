import sys, json, gc, warnings
warnings.filterwarnings("ignore"); sys.path.append("../src")
import numpy as np, pandas as pd
from framework_analitico import normalizar_direccion_ataque, identificar_goles
EQ="América"
ev=pd.read_csv("../data/raw/events_jardine_america.csv",low_memory=False)
keep=[c for c in ev.columns if c in {"match_id","period","minute","second","type","team","location","possession_team","shot_outcome","index","player","shot_statsbomb_xg","x","y","play_pattern","pass_type","pass_technique","pass_end_location","pass_outcome","possession"}]
ev=ev[keep].copy()
ids=list(ev.match_id.unique()); rows=[]; shots=[]
for k in range(0,len(ids),15):
    sub=normalizar_direccion_ataque(ev[ev.match_id.isin(ids[k:k+15])].copy(),equipo=EQ)
    goles=identificar_goles(sub)
    sub["t"]=sub.minute*60+sub.second
    for mid,g in sub.groupby("match_id"):
        gl=goles[goles.match_id==mid]
        # corners
        c=g[(g.type=="Pass")&(g.pass_type=="Corner")]
        for _,r in c.iterrows():
            prev=gl[(gl.period<r.period)|((gl.period==r.period)&(gl.t_segundos<r.t))]
            gf=int((prev.team==EQ).sum()); gc_=int((prev.team!=EQ).sum())
            # jugada: mismo possession id
            jug=g[(g.possession==r.possession)&(g.play_pattern=="From Corner")]
            sh=jug[jug.type=="Shot"]
            propio=(r.team==EQ)
            rows.append({"match_id":int(mid),"propio":bool(propio),"minuto":int(r.minute),"period":int(r.period),"dif":gf-gc_ if propio else gc_-gf,
                         "remate":int(len(sh)>0),"xg":float(sh.shot_statsbomb_xg.sum()) if len(sh) else 0.0,"tecnica":r.pass_technique if isinstance(r.pass_technique,str) else None})
            for _,s in sh.iterrows():
                shots.append({"match_id":int(mid),"propio":bool(propio),"x":float(s.x),"y":float(s.y),"xg":float(s.shot_statsbomb_xg) if pd.notna(s.shot_statsbomb_xg) else 0})
    del sub; gc.collect()
json.dump({"corners":rows,"shots":shots},open("corner_events.json","w"))
d=pd.DataFrame(rows); print(len(d))
for p in (True,False):
    x=d[d.propio==p]; print("propio" if p else "rival", len(x), round(x.remate.mean()*100,1), round(x.xg.sum(),2))
a=d[d.propio]
def tr(m): return "0-30" if m<30 else "30-60" if m<60 else "60-75" if m<75 else "75+"
a["tr"]=a.minuto.map(tr); a["est"]=np.where(a.dif>0,"ganando",np.where(a.dif<0,"perdiendo","empate"))
print(a.groupby("tr").remate.agg(["size","mean"]).round(3)); print(a.groupby("est").remate.agg(["size","mean"]).round(3)); print(a.groupby("tecnica").remate.agg(["size","mean"]).round(3))
r=d[~d.propio].copy(); r["tr"]=r.minuto.map(tr); r["est"]=np.where(r.dif<0,"ganando_am",np.where(r.dif>0,"perdiendo_am","empate"))
print(r.groupby("est").remate.agg(["size","mean"]).round(3))
s=pd.DataFrame(shots)
for p in (True,False):
    x=s[s.propio==p]; print("remates", "propios" if p else "rival", len(x), "área chica(x>=114,y30-50):", round(((x.x>=114)&(x.y.between(30,50))).mean()*100,1),"punto penal(x 102-114, y30-50):",round(((x.x.between(102,114))&(x.y.between(30,50))).mean()*100,1), "xG/remate", round(x.xg.mean(),3))
