import sys, json, warnings
warnings.filterwarnings("ignore"); sys.path.append("../src")
import pandas as pd, numpy as np
from framework_analitico import normalizar_direccion_ataque
ruta_ev, ruta_p, EQ, salida, entren = sys.argv[1:6]
leer=lambda r: pd.read_csv(r,low_memory=False) if r.endswith(".csv") else pd.read_pickle(r)
ev=leer(ruta_ev); P=leer(ruta_p)
rec=json.load(open("../data/processed/recorrido_final.json"))
finales=set()
for t in rec:
    if t["entrenador"]==entren and str(t["resultado"]).startswith("Campe"):
        for l in t["liguilla"]:
            if l["etapa"]=="Final": finales.add(l["fecha"])
ids=list(ev.match_id.unique()); res=[]
for mid in ids:
    sub=ev[ev.match_id==mid].copy()
    r=P[P.match_id==mid].iloc[0]
    en=normalizar_direccion_ataque(sub.copy(),equipo=EQ)
    en=en[en.x.notna()]
    if not len(en): continue
    mx=int(en.m.max()) if "m" in en else int(en.minute.max())
    m=en.minute
    a=en[(en.team==EQ)&(en.x>80)].groupby("minute").size().reindex(range(mx+1),fill_value=0)
    q=en[(en.team!=EQ)&(en.x<40)].groupby("minute").size().reindex(range(mx+1),fill_value=0)
    tot=a.sum()/(a.sum()+q.sum()) if a.sum()+q.sum() else .5
    home=r.home_team==EQ; gf=int(r.home_score if home else r.away_score); gc=int(r.away_score if home else r.home_score)
    res.append(dict(id=int(mid),date=str(r.match_date),rival=r.away_team if home else r.home_team,home=bool(home),gf=gf,gc=gc,ft=float(tot),mx=mx,final=str(r.match_date) in finales,en=en,a=a,q=q))
def pick():
    s=[]; used=set()
    def add(x,tag):
        if x is not None and x["id"] not in used: used.add(x["id"]); x["tag"]=tag; s.append(x)
    add(next(iter([x for x in res if x["final"]]),None) if any(x["final"] for x in res) else None,"Final de torneo campeón")
    add(max(res,key=lambda x:x["ft"]),"Más territorio de todos sus partidos")
    add(min(res,key=lambda x:x["ft"]),"Menos territorio de todos sus partidos")
    L=[x for x in res if x["gf"]<x["gc"]]; 
    if L: add(max(L,key=lambda x:x["ft"]),"Derrota con más territorio")
    W=[x for x in res if x["gf"]>x["gc"]]
    if W: add(min(W,key=lambda x:x["ft"]),"Victoria con menos territorio")
    C=[x for x in res if x["gf"]>=x["gc"] and x["gc"]>=1]
    if C: add(max(C,key=lambda x:x["gc"]*10+x["gf"]),"Partido con más goles en contra sin perder")
    return s
out=[]
for x in pick():
    en=x["en"]; mx=x["mx"]
    g=np.zeros((mx+1,12,8),dtype=int); own=en[en.team==EQ]
    for mm,xx,yy in zip(own.minute,own.x,own.y):
        if mm<=mx: g[int(mm),min(11,int(xx//10)),min(7,int(yy//10))]+=1
    shots=[{"m":int(e.minute),"team":"A" if e.team==EQ else "R","xg":round(float(e.shot_statsbomb_xg),2) if e.shot_statsbomb_xg==e.shot_statsbomb_xg else 0.03,"goal":bool(e.shot_outcome=="Goal"),"x":float(e.x),"y":float(e.y)} for _,e in en[en.type=="Shot"].iterrows()]
    subs=[{"m":int(e.minute),"team":"A" if e.team==EQ else "R"} for _,e in en[en.type=="Substitution"].iterrows()]
    out.append(dict(id=x["id"],date=x["date"],rival=x["rival"],home=x["home"],gf=x["gf"],gc=x["gc"],ft=round(x["ft"]*100,1),mx=mx,tag=x["tag"],a=x["a"].tolist(),q=x["q"].tolist(),grid=g.tolist(),shots=shots,subs=subs))
json.dump({"equipo":EQ,"partidos":out},open(salida,"w"),ensure_ascii=False)
print(EQ,len(res),[(o["date"],o["rival"],o["gf"],o["gc"],o["ft"],o["tag"]) for o in out])
