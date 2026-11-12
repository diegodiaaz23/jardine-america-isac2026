"""Hasta donde llego cada equipo en cada torneo (Apertura/Clausura) con su entrenador."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "../data/raw/comparacion")
base = "../data/raw/"
fuentes = [("André Jardine", "América", pd.read_csv(base + "partidos_jardine_america.csv"))]
for n, eq, f in [("Jorge Almada", "Pachuca", "jorge_guillermo_almada_lves__pachuca"),
                 ("Diego Cocca", "Atlas", "diego_mart_n_cocca_pera__atlas"),
                 ("Antonio Mohamed", "Toluca", "antonio_ricardo_mohamed_matijevich__toluca")]:
    fuentes.append((n, eq, pd.read_pickle(base + f"comparacion/partidos_{f}.pkl.gz")))

def torneo(d):
    return f"{'Apertura' if d.month >= 7 else 'Clausura'} {d.year}"
def etapa(labels):
    t = " ".join(labels).lower()
    if "final" in t and "semi" not in t and "quarter" not in t: return "Final"
    if "semi" in t and "play-off" not in t: return "Semifinal"
    if "quarter" in t: return "Cuartos"
    return "Repechaje"
ORDEN = {"Repechaje": 1, "Cuartos": 2, "Semifinal": 3, "Final": 4}
simple = {"Apertura", "Clausura", "Regular Season"}
res = []
for entrenador, equipo, d in fuentes:
    d = d.copy(); d["match_date"] = pd.to_datetime(d.match_date); d = d.sort_values("match_date")
    d["torneo"] = d.match_date.map(torneo)
    d["gf"] = np.where(d.home_team == equipo, d.home_score, d.away_score)
    d["gc"] = np.where(d.home_team == equipo, d.away_score, d.home_score)
    d["rival"] = np.where(d.home_team == equipo, d.away_team, d.home_team)
    for t, g in d.groupby("torneo", sort=False):
        reg = g[g.competition_stage.isin(simple)]
        po = g[~g.competition_stage.isin(simple)]
        series = []
        for rival, s in po.groupby("rival"):
            s = s.sort_values("match_date")
            series.append({"rival": rival, "etapa": etapa(s.competition_stage.tolist()), "gf": int(s.gf.sum()), "gc": int(s.gc.sum()),
                           "partidos": int(len(s)), "fecha": s.match_date.iloc[0].strftime("%Y-%m-%d")})
        series.sort(key=lambda x: x["fecha"])
        tabla = {"entrenador": entrenador, "equipo": equipo, "torneo": t, "partidos_regulares": int(len(reg)), "liguilla": series}
        res.append(tabla)
json.dump(res, open("../data/processed/recorrido_torneos.json", "w"), ensure_ascii=False, indent=1)
for r in res:
    print(r["equipo"], r["torneo"], "reg", r["partidos_regulares"], [(s["etapa"], s["rival"], f'{s["gf"]}-{s["gc"]}', s["partidos"]) for s in r["liguilla"]])
