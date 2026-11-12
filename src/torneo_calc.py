"""Por torneo: puntos, xG a favor/en contra, PPDA, field tilt y continuidad de titulares vs el torneo anterior.
Uso: python3 torneo_calc.py <events.pkl.gz> <partidos.pkl.gz> "<Equipo>" <salida.json>"""
import sys, json, gc, warnings, ast
warnings.filterwarnings("ignore"); sys.path.append(".")
import numpy as np, pandas as pd
from framework_analitico import normalizar_direccion_ataque, metricas_por_partido, extraer_alineaciones_titulares
ruta_ev, ruta_p, EQ, salida = sys.argv[1:5]
ev = pd.read_pickle(ruta_ev); P = pd.read_pickle(ruta_p)
P["match_date"] = pd.to_datetime(P.match_date)
P["torneo"] = P.match_date.map(lambda d: f"{'Apertura' if d.month>=7 else 'Clausura'} {d.year}")
simple = {"Apertura","Clausura","Regular Season"}
P = P[P.competition_stage.isin(simple)].copy()
P["gf"] = np.where(P.home_team==EQ, P.home_score, P.away_score); P["gc"] = np.where(P.home_team==EQ, P.away_score, P.home_score)
P["pts"] = np.where(P.gf>P.gc,3,np.where(P.gf==P.gc,1,0))
ids = [i for i in ev.match_id.unique() if i in set(P.match_id)]
mp=[]; al=[]
for k in range(0,len(ids),20):
    sub = ev[ev.match_id.isin(ids[k:k+20])].copy()
    en = normalizar_direccion_ataque(sub, equipo=EQ)
    mp.append(metricas_por_partido(en, equipo=EQ)); al.append(extraer_alineaciones_titulares(sub, EQ))
    del sub,en; gc.collect()
mp = pd.concat(mp); al = pd.concat(al)
d = P.merge(mp, on="match_id").sort_values("match_date")
filas=[]
tit = {}
for t,g in d.groupby("torneo", sort=False):
    cnt = {}
    for _,r in al[al.match_id.isin(g.match_id)].iterrows():
        for j in r.jugadores: cnt[j]=cnt.get(j,0)+1
    tit[t]=cnt
    filas.append({"torneo":t,"partidos":int(len(g)),"puntos":int(g.pts.sum()),"pts_por_partido":round(float(g.pts.mean()),2),
        "xg_favor":round(float(g.xg_favor.mean()),2),"xg_contra":round(float(g.xg_contra.mean()),2),
        "goles_favor":round(float(g.gf.mean()),2),"goles_contra":round(float(g.gc.mean()),2),
        "ppda":round(float(g.ppda.mean()),2),"field_tilt_rel":round(float(g.field_tilt_relativo_pct.mean()),1)})
orden=[f["torneo"] for f in filas]
for i,f in enumerate(filas):
    if i==0: f["continuidad_titulares_pct"]=None; continue
    prev=tit[orden[i-1]]; actual=tit[f["torneo"]]
    habituales={j for j,c in prev.items() if c>=5}
    tot=sum(actual.values()); mant=sum(c for j,c in actual.items() if j in habituales)
    f["continuidad_titulares_pct"]=round(100*mant/tot,1) if tot else None
    f["vs_torneo"]=orden[i-1]
json.dump({"equipo":EQ,"torneos":filas}, open(salida,"w"), ensure_ascii=False, indent=1)
for f in filas: print(EQ,f)
