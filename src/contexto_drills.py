import pandas as pd, numpy as np, json, ast, sys
sys.path.append("."); from framework_analitico import normalizar_direccion_ataque
EQ="América"
cols=["match_id","period","minute","second","type","team","location","shot_statsbomb_xg","shot_outcome","play_pattern","possession_team","pass_type","pass_outcome","index"]
ev=pd.read_csv("../data/raw/events_jardine_america.csv",usecols=cols,low_memory=False)
P=pd.read_csv("../data/raw/partidos_jardine_america.csv")
ev["t"]=ev.minute+ev.second/60
def partido(mid):
    e=normalizar_direccion_ataque(ev[ev.match_id==mid].copy(),equipo=EQ).dropna(subset=["x"])
    return e
out={}
def tramo(e,a,b):
    s=e[(e.t>=a)&(e.t<b)]
    mi=s[s.team==EQ]; ri=s[s.team!=EQ]
    xg=lambda d:round(float(d.shot_statsbomb_xg.fillna(0)[d.type=="Shot"].sum()),2)
    sh=lambda d:int((d.type=="Shot").sum())
    ft_a=(mi.x>80).sum(); ft_b=(ri.x<40).sum()
    ft=round(100*ft_a/(ft_a+ft_b),0) if ft_a+ft_b else None
    pas=lambda d:(d.type=="Pass").sum(); tot=pas(mi)+pas(ri)
    pos=round(100*pas(mi)/tot,0) if tot else None
    return {"xg_am":xg(mi),"xg_rv":xg(ri),"tiros_am":sh(mi),"tiros_rv":sh(ri),"field_tilt":ft,"posesion":pos}
# ---- Drill 1: Toluca - América 2025-11-09
m1=int(P[P.match_date=="2025-11-09"].match_id.iloc[0]); e=partido(m1)
r=P[P.match_id==m1].iloc[0]; print(r[["home_team","away_team","home_score","away_score"]].tolist())
h1=e[e.period==1]
out["d1"]={"match_id":m1,"tramos":{f"{a}-{b}":tramo(h1,a,b) for a,b in [(0,15),(15,30),(30,50)]},
  "goles":[(int(g.minute),g.team) for _,g in e[(e.type=="Shot")&(e.shot_outcome=="Goal")].iterrows()],
  "ult10":tramo(h1,35,50)}
# promedio de América en los 15 min finales del 1T y primeros 45 en todos los partidos
tr=[];
for mid in P.match_id.unique():
    ee=normalizar_direccion_ataque(ev[(ev.match_id==mid)&(ev.period==1)].copy(),equipo=EQ).dropna(subset=["x"])
    if len(ee)==0: continue
    tr.append((tramo(ee,0,15),tramo(ee,15,30),tramo(ee,30,50)))
def avg(i,k): return round(float(np.nanmean([t[i][k] for t in tr if t[i][k] is not None])),2)
out["d1"]["prom_temp"]={f"{i}":{k:avg(i,k) for k in ["xg_am","xg_rv","tiros_am","field_tilt","posesion"]} for i in range(3)}
# ---- Drill 2: Cruz Azul - América 2025-05-19, minuto 56
m2=int(P[P.match_date=="2025-05-19"].match_id.iloc[0]); e=partido(m2); r=P[P.match_id==m2].iloc[0]
print(r[["home_team","away_team","home_score","away_score","competition_stage"]].tolist())
g=e[(e.type=="Shot")&(e.shot_outcome=="Goal")][["minute","team"]].values.tolist()
out["d2"]={"match_id":m2,"goles":g,"antes_gol":tramo(e,41,56),"hasta56":tramo(e,0,56),
  "sustituciones":[(int(a),b) for a,b in ev[(ev.match_id==m2)&(ev.type=="Substitution")][["minute","team"]].values]}
# ---- Drill 3: córner perdiendo por 1 desde el 75'
rows=[]
for mid in P.match_id.unique():
    e=ev[ev.match_id==mid].sort_values("index")
    gol=e[(e.type=="Shot")&(e.shot_outcome=="Goal")][["t","team"]]
    def marcador(t):
        a=(gol[(gol.t<t)&(gol.team==EQ)]).shape[0]; b=(gol[(gol.t<t)&(gol.team!=EQ)]).shape[0]; return a-b
    for _,c in e[(e.team==EQ)&(e.type=="Pass")&(e.pass_type=="Corner")&(e.t>=75)].iterrows():
        rows.append({"match_id":mid,"t":c.t,"dif":marcador(c.t)})
c=pd.DataFrame(rows); print(len(c), c.dif.value_counts().to_dict())
perd=c[c.dif==-1]
# remates tras el córner: Shot con play_pattern From Corner por el mismo equipo en los 15 s siguientes
def remate(r):
    e=ev[(ev.match_id==r.match_id)]; s=e[(e.team==EQ)&(e.type=="Shot")&(e.t>r.t)&(e.t<=r.t+0.25)]
    return len(s)>0
perd["rem"]=perd.apply(remate,axis=1)
out["d3"]={"n_corners_75_perdiendo_1":int(len(perd)),"pct_remate":round(float(perd.rem.mean()*100),1) if len(perd) else None,
   "n_partidos_perdiendo_1_al_75":int(len(perd.match_id.unique()))}
# tiros de América por 10 min cuando pierde por 1 vs ritmo normal
tir=[];base=[]
for mid in P.match_id.unique():
    e=ev[ev.match_id==mid]; gol=e[(e.type=="Shot")&(e.shot_outcome=="Goal")][["t","team"]]
    ta=e[(e.team==EQ)&(e.type=="Shot")]
    base.append(len(ta[(ta.t>=60)&(ta.t<75)])*10/15)
    for t0 in [75]:
        a=(gol[(gol.t<t0)&(gol.team==EQ)]).shape[0]; b=(gol[(gol.t<t0)&(gol.team!=EQ)]).shape[0]
        if a-b==-1: tir.append(len(ta[(ta.t>=75)&(ta.t<90)])*10/15)
out["d3"]["tiros_10min_perdiendo_1_final"]=round(float(np.mean(tir)),2) if tir else None
out["d3"]["tiros_10min_60_75_todos"]=round(float(np.mean(base)),2)
json.dump(out,open("../data/processed/contexto_drills.json","w"),ensure_ascii=False,indent=1)
print(json.dumps(out,ensure_ascii=False,indent=1))
