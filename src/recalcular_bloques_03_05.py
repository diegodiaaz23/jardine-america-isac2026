"""Recalcula los bloques afectados por el bug de coordenadas del rival (PPDA, field tilt relativo)."""
import sys, json
sys.path.append("../src")
import numpy as np, pandas as pd
from framework_analitico import (normalizar_direccion_ataque, perfil_ofensivo_defensivo, indice_directividad,
    metricas_por_partido, agregar_contexto_partido, resumen_variabilidad, variabilidad_por_grupo)

EQUIPO = "América"
try:
    events = pd.read_pickle("/tmp/events.pkl")
except Exception:
    events = pd.read_csv("../data/raw/events_jardine_america.csv", low_memory=False)
partidos = pd.read_csv("../data/raw/partidos_jardine_america.csv")
en = normalizar_direccion_ataque(events, equipo=EQUIPO)

agg = perfil_ofensivo_defensivo(en, equipo=EQUIPO)
mp = metricas_por_partido(en, equipo=EQUIPO)
mp = agregar_contexto_partido(mp, partidos, EQUIPO)
cols = ["posesion_pct","field_tilt_pct","field_tilt_relativo_pct","ppda","tiros_favor","tiros_contra","xg_favor","xg_contra","pct_pases_largos"]
cols = [c for c in cols if c in mp.columns]
out = {
  "agregado": {k: (None if (isinstance(v,float) and np.isnan(v)) else v) for k,v in agg.items()},
  "cv": resumen_variabilidad(mp, cols).to_dict(orient="records"),
  "por_localidad": variabilidad_por_grupo(mp, "localidad", cols).to_dict(orient="records"),
  "por_temporada": variabilidad_por_grupo(mp, "temporada", cols).to_dict(orient="records"),
  "partidos": mp.assign(match_date=pd.to_datetime(mp["match_date"]).dt.strftime("%Y-%m-%d")).sort_values("match_date").round(3).to_dict(orient="records"),
}
json.dump(out, open("../data/processed/bloques_03_05_corregidos.json","w"), ensure_ascii=False, default=str)
print(json.dumps(out["agregado"], ensure_ascii=False, default=str))
print(len(mp))
