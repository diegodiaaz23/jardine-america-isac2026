"""Recalcula metricas de los bloques 03/05 con coordenadas del rival corregidas."""
import sys, json
sys.path.append("../src")
import numpy as np, pandas as pd
from framework_analitico import (normalizar_direccion_ataque, perfil_ofensivo_defensivo, indice_directividad,
                                 metricas_por_partido, agregar_contexto_partido, resumen_variabilidad)
EQ = "América"
ev = pd.read_pickle("/tmp/events.pkl")
partidos = pd.read_csv("../data/raw/partidos_jardine_america.csv")
en = normalizar_direccion_ataque(ev, equipo=EQ)
glob = perfil_ofensivo_defensivo(en, equipo=EQ)
print("GLOBAL", glob)
print("DIRECT", indice_directividad(en, equipo=EQ))
mp = metricas_por_partido(en, equipo=EQ)
mp = agregar_contexto_partido(mp, partidos, EQ)
cols = ["posesion_pct","field_tilt_pct","field_tilt_relativo_pct","ppda","tiros_favor","tiros_contra","xg_favor","xg_contra","pct_pases_largos"]
print("MEDIA por localidad\n", mp.groupby("localidad")[cols].mean().round(2).T)
print("MEDIA por temporada\n", mp.groupby("temporada")[cols].mean().round(2).T)
print("CV\n", (mp[cols].std()/mp[cols].mean()).round(3))
print("MEDIA global por partido\n", mp[cols].mean().round(2))
mp.to_json("../data/processed/metricas_partido_corregidas.json", orient="records", force_ascii=False)
