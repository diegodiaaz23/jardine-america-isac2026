"""Exporta los datos que alimentan la herramienta de coach drills (HTML)."""
import sys, json
sys.path.append("../src")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from framework_analitico import (normalizar_direccion_ataque, dataset_prediccion_resultado,
                                 dataset_causal_reaccion_gol_contra)

EQUIPO = "América"
events = pd.read_csv("../data/raw/events_jardine_america.csv", low_memory=False)
partidos = pd.read_csv("../data/raw/partidos_jardine_america.csv")
events_norm = normalizar_direccion_ataque(events, equipo=EQUIPO)
ds = dataset_prediccion_resultado(events_norm, equipo=EQUIPO, partidos_df=partidos)
ds["match_date"] = pd.to_datetime(ds["match_date"])
ds = ds.sort_values("match_date").reset_index(drop=True)

FEATURES = ["posesion_pct_1t","field_tilt_pct_1t","field_tilt_relativo_pct_1t","ppda_1t",
            "tiros_favor_1t","tiros_contra_1t","xg_favor_1t","xg_contra_1t",
            "pct_pases_largos_1t","longitud_promedio_m_1t","goles_favor_1t","goles_contra_1t"]
ds["es_local"] = (ds["localidad"] == "Local").astype(int)
F = FEATURES + ["es_local"]
train = ds[ds["temporada"] != "2025/2026"]
hold = ds[ds["temporada"] == "2025/2026"].copy()
sc = StandardScaler().fit(train[F])
m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(train[F]), train["resultado_final"])
P = m.predict_proba(sc.transform(hold[F]))
clases = list(m.classes_)
for i, c in enumerate(clases):
    hold["p_" + c] = P[:, i]
hold["match_date"] = hold["match_date"].dt.strftime("%Y-%m-%d")
cols = ["match_id","match_date","rival","localidad","resultado_final","dif_gol_final"] + FEATURES + ["p_" + c for c in clases]
hold_out = hold[cols].round(3).to_dict(orient="records")

# causal: goles en contra
cols_ev = ["match_id","period","minute","second","type","team","shot_outcome","player","substitution_replacement"]
ev_c = events[[c for c in cols_ev if c in events.columns]]
dc = dataset_causal_reaccion_gol_contra(ev_c, equipo=EQUIPO, partidos_df=partidos)
info = ds[["match_id","match_date","rival","localidad","resultado_final","dif_gol_final","goles_favor_1t","goles_contra_1t"]].copy()
info["match_date"] = info["match_date"].dt.strftime("%Y-%m-%d")
dc = dc.merge(info, on="match_id", how="left", suffixes=("", "_p"))
causal_out = dc.round(3).to_dict(orient="records")

json.dump({"clases": clases, "holdout": hold_out, "causal": causal_out},
          open("../data/processed/drills_data.json", "w"), ensure_ascii=False, default=str)
print("OK", len(hold_out), len(causal_out), clases)
