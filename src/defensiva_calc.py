import sys, json
sys.path.append("../src")
import numpy as np, pandas as pd
from framework_analitico import normalizar_direccion_ataque, altura_linea_defensiva, tiempo_reaccion_perdida, agregar_contexto_partido
EQ="América"
ev=pd.read_pickle("/tmp/events.pkl"); partidos=pd.read_csv("../data/raw/partidos_jardine_america.csv")
en=normalizar_direccion_ataque(ev,equipo=EQ)
al=altura_linea_defensiva(en,EQ); al=agregar_contexto_partido(al,partidos,EQ)
print("ALTURA media",round(al.altura_linea_x.mean(),2),"std",round(al.altura_linea_x.std(),2))
print(al.groupby("temporada").altura_linea_x.mean().round(2).to_dict())
print(al.groupby("localidad").altura_linea_x.mean().round(2).to_dict())
slim=en[["match_id","period","index","minute","second","possession_team","team","type"]].copy()
del en, ev
import gc; gc.collect()
r=tiempo_reaccion_perdida(slim,EQ)
print("perdidas",len(r),"contrapresiono %",round(r.contrapresiono.mean()*100,1),"mediana reaccion s",r.tiempo_reaccion.median(),"media",round(r.tiempo_reaccion.mean(),2))
rp=r.groupby("match_id").agg(pct=("contrapresiono","mean"),med=("tiempo_reaccion","median")).reset_index()
rp=agregar_contexto_partido(rp,partidos,EQ)
print(rp.groupby("temporada")[["pct","med"]].mean().round(3).to_dict())
print(rp.groupby("localidad")[["pct","med"]].mean().round(3).to_dict())
print("CV altura",round(al.altura_linea_x.std()/al.altura_linea_x.mean(),3))
