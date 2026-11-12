"""Aplica el framework completo (estilo, modelo al descanso, causal) a UN entrenador.
Uso: python3 comparacion_calc.py <events.pkl|csv> <partidos.pkl|csv> "<Equipo>" "<Entrenador>" <salida.json>
Se procesa por bloques de partidos para no agotar la memoria."""
import sys, json, gc, warnings
warnings.filterwarnings("ignore")
sys.path.append("../src")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from framework_analitico import (normalizar_direccion_ataque, dataset_prediccion_resultado, metricas_por_partido,
    altura_linea_defensiva, tiempo_reaccion_perdida, resumen_balon_parado,
    dataset_causal_reaccion_gol_contra, matching_estratificado)

ruta_ev, ruta_p, EQUIPO, ENTRENADOR, salida = sys.argv[1:6]
def leer(r):
    return pd.read_csv(r, low_memory=False) if r.endswith(".csv") else pd.read_pickle(r)
ev = leer(ruta_ev); partidos = leer(ruta_p)
if "temporada" not in partidos.columns:
    partidos["temporada"] = partidos["season"].astype(str)
assert (ev.team == EQUIPO).any(), f"{EQUIPO} no aparece en los eventos: {ev.team.unique()[:6]}"
ids = list(ev.match_id.unique()); print("partidos", len(ids), flush=True)

cols_causal = ["match_id","period","minute","second","type","team","shot_outcome","player","substitution_replacement"]
causal_ev = ev[[c for c in cols_causal if c in ev.columns]].copy()

ds_l, mp_l, alt_l, rea_l = [], [], [], []
corner = {"n": 0, "rem": 0, "xg": 0.0}
for k in range(0, len(ids), 20):
    sub_ids = ids[k:k+20]
    sub = ev[ev.match_id.isin(sub_ids)].copy()
    en = normalizar_direccion_ataque(sub, equipo=EQUIPO)
    pp = partidos[partidos.match_id.isin(sub_ids)]
    ds_l.append(dataset_prediccion_resultado(en, equipo=EQUIPO, partidos_df=pp))
    mp_l.append(metricas_por_partido(en, equipo=EQUIPO))
    alt_l.append(altura_linea_defensiva(en, EQUIPO))
    slim = en[["match_id","period","index","minute","second","possession_team","team","type"]].copy()
    rea_l.append(tiempo_reaccion_perdida(slim, EQUIPO))
    c = resumen_balon_parado(en, EQUIPO, "corner")
    corner["n"] += c["n_balones_parados"]; corner["rem"] += c["n_remates"]; corner["xg"] += c.get("xg_total", 0.0)
    del sub, en, slim; gc.collect()
    print("bloque", k // 20 + 1, flush=True)
del ev; gc.collect()
ds = pd.concat(ds_l, ignore_index=True); mp = pd.concat(mp_l, ignore_index=True)
alt = pd.concat(alt_l, ignore_index=True); rea = pd.concat(rea_l, ignore_index=True)

out = {"entrenador": ENTRENADOR, "equipo": EQUIPO, "n_partidos": int(len(ds)),
       "temporadas": sorted(ds["temporada"].dropna().unique().tolist())}
cols = ["posesion_pct","field_tilt_relativo_pct","ppda","tiros_favor","tiros_contra","xg_favor","xg_contra","pct_pases_largos"]
out["estilo"] = {c: round(float(mp[c].mean()), 3) for c in cols}
out["altura_linea_x"] = round(float(alt.altura_linea_x.mean()), 2)
out["contrapresion_pct"] = round(float(rea.contrapresiono.mean() * 100), 1)
out["reaccion_mediana_s"] = float(rea.tiempo_reaccion.median())
out["corner"] = {"n": corner["n"], "pct_remate": round(corner["rem"] / corner["n"] * 100, 1) if corner["n"] else None,
                 "xg_por_partido": round(corner["xg"] / len(ds), 3)}
out["resultados"] = ds.resultado_final.value_counts().to_dict()
perd = ds[ds.situacion_al_descanso == "perdiendo"]
out["perdiendo_descanso"] = {"n": int(len(perd)), "recupera_punto_pct": round(float((perd.resultado_final != "Pierde").mean() * 100), 1) if len(perd) else None}

# ---- modelo al descanso (70/30 cronologico) ----
ds["match_date"] = pd.to_datetime(ds["match_date"]); ds = ds.sort_values("match_date").reset_index(drop=True)
FEAT = ["posesion_pct_1t","field_tilt_pct_1t","field_tilt_relativo_pct_1t","ppda_1t","tiros_favor_1t","tiros_contra_1t",
        "xg_favor_1t","xg_contra_1t","pct_pases_largos_1t","longitud_promedio_m_1t","goles_favor_1t","goles_contra_1t"]
ds["es_local"] = (ds.localidad == "Local").astype(int); F = FEAT + ["es_local"]
corte = int(round(len(ds) * 0.7)); tr, ho = ds.iloc[:corte], ds.iloc[corte:]
CL = sorted(["Pierde","Empata","Gana"])
def alinear(m, P):
    full = np.full((len(P), 3), 1e-6)
    for i, c in enumerate(m.classes_): full[:, CL.index(c)] = P[:, i]
    return full / full.sum(axis=1, keepdims=True)
dist = tr.resultado_final.value_counts(normalize=True).reindex(CL).fillna(0).values
dist = np.clip(dist, 1e-6, None); dist /= dist.sum()
ll1 = log_loss(ho.resultado_final, np.tile(dist, (len(ho), 1)), labels=CL)
sc2 = StandardScaler().fit(tr[["goles_favor_1t","goles_contra_1t"]])
m2 = LogisticRegression(max_iter=2000).fit(sc2.transform(tr[["goles_favor_1t","goles_contra_1t"]]), tr.resultado_final)
ll2 = log_loss(ho.resultado_final, alinear(m2, m2.predict_proba(sc2.transform(ho[["goles_favor_1t","goles_contra_1t"]]))), labels=CL)
sc = StandardScaler().fit(tr[F]); m3 = LogisticRegression(max_iter=2000).fit(sc.transform(tr[F]), tr.resultado_final)
P3 = alinear(m3, m3.predict_proba(sc.transform(ho[F])))
ll3 = log_loss(ho.resultado_final, P3, labels=CL)
acc = accuracy_score(ho.resultado_final, [CL[i] for i in P3.argmax(axis=1)])
out["modelo"] = {"n_train": int(len(tr)), "n_holdout": int(len(ho)), "logloss_ingenuo": round(ll1, 3),
                 "logloss_marcador": round(ll2, 3), "logloss_completo": round(ll3, 3),
                 "mejora_vs_ingenuo_pct": round((ll1 - ll3) / ll1 * 100, 1), "accuracy": round(acc, 3)}

# ---- causal ----
dc = dataset_causal_reaccion_gol_contra(causal_ev, equipo=EQUIPO, partidos_df=partidos)
if len(dc) >= 10 and dc.tratamiento.sum() >= 3:
    r = matching_estratificado(dc, "tratamiento", "resultado", ["situacion", "minuto_bucket"])
    out["causal"] = {"n_eventos": int(len(dc)), "n_tratados": int(r["n_tratados_total"]), "ingenua_pp": round(r["diferencia_ingenua"] * 100, 1),
                     "emparejado_pp": round(r["efecto_estimado"] * 100, 1),
                     "ic95_pp": [round(r["ic95_bootstrap"][0] * 100, 1), round(r["ic95_bootstrap"][1] * 100, 1)]}
else:
    out["causal"] = {"n_eventos": int(len(dc)), "n_tratados": int(dc.tratamiento.sum()) if len(dc) else 0, "nota": "muestra insuficiente"}
json.dump(out, open(salida, "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
