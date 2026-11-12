import sys, json, gc, warnings
warnings.filterwarnings("ignore"); sys.path.append("../src")
import pandas as pd
from framework_analitico import normalizar_direccion_ataque, dataset_prediccion_resultado
ruta_ev, ruta_p, EQ, salida = sys.argv[1:5]
leer=lambda r: pd.read_csv(r,low_memory=False) if r.endswith(".csv") else pd.read_pickle(r)
ev=leer(ruta_ev); P=leer(ruta_p)
if "temporada" not in P.columns: P["temporada"]=P["season"].astype(str)
ids=list(ev.match_id.unique()); out=[]
for k in range(0,len(ids),20):
    sub=ev[ev.match_id.isin(ids[k:k+20])].copy()
    en=normalizar_direccion_ataque(sub,equipo=EQ)
    d=dataset_prediccion_resultado(en,equipo=EQ,partidos_df=P[P.match_id.isin(ids[k:k+20])])
    out.append(d[["match_id","situacion_al_descanso","resultado_final"]]); del sub,en; gc.collect()
d=pd.concat(out).merge(P[["match_id","match_date"]],on="match_id")
d.to_json(salida,orient="records",force_ascii=False); print("ok",len(d))
