"""Rejillas para mapas de calor (presión, territorio) y altura de línea por técnico.
Uso: python3 mapas_calc.py <events> "<Equipo>" "<Entrenador>" <salida.json>"""
import sys, json, gc, warnings
warnings.filterwarnings("ignore")
sys.path.append(".")
import numpy as np, pandas as pd
from framework_analitico import normalizar_direccion_ataque, altura_linea_defensiva
ruta, EQ, ENT, salida = sys.argv[1:5]
cols = ["match_id","period","type","team","location"]
ev = pd.read_csv(ruta, usecols=cols, low_memory=False) if ruta.endswith(".csv") else pd.read_pickle(ruta)[cols]
ids = list(ev.match_id.unique())
NX, NY = 12, 8
xe, ye = np.linspace(0,120,NX+1), np.linspace(0,80,NY+1)
H = lambda: np.zeros((NX, NY))
pres, team_all, riv_all, defa = H(), H(), H(), H()
alt = []; ndef_x = np.zeros(12)
for k in range(0, len(ids), 20):
    sub = ev[ev.match_id.isin(ids[k:k+20])].copy()
    en = normalizar_direccion_ataque(sub, equipo=EQ)
    en = en.dropna(subset=["x","y"])
    t = en[en.team == EQ]; r = en[en.team != EQ]
    h = lambda d: np.histogram2d(d.x.clip(0,119.99), d.y.clip(0,79.99), bins=[xe,ye])[0]
    pres += h(t[t.type=="Pressure"]); team_all += h(t); riv_all += h(r)
    d = t[t.type.isin(["Interception","Clearance","Duel","Block"])]; defa += h(d)
    alt.append(altura_linea_defensiva(en, EQ))
    del sub, en; gc.collect()
alt = pd.concat(alt).altura_linea_x
n = len(ids)
share = np.divide(team_all, team_all+riv_all, out=np.full((NX,NY),0.5), where=(team_all+riv_all)>0)
out = {"entrenador": ENT, "equipo": EQ, "n_partidos": n,
       "presion_por_partido": np.round(pres/n,3).tolist(),
       "territorio_pct": np.round(share*100,1).tolist(),
       "equipo_por_partido": np.round(team_all/n,3).tolist(), "rival_por_partido": np.round(riv_all/n,3).tolist(),
       "acciones_def_por_partido": np.round(defa/n,3).tolist(),
       "altura": {"media": round(float(alt.mean()),2), "p25": round(float(alt.quantile(.25)),2),
                  "p75": round(float(alt.quantile(.75)),2), "min": round(float(alt.min()),2), "max": round(float(alt.max()),2)}}
json.dump(out, open(salida,"w"))
print(EQ, n, out["altura"])
