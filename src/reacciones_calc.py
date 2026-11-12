"""Reacción ante situaciones inesperadas para UN entrenador, procesando por bloques de partidos.
Uso: python3 reacciones_calc.py <events.pkl|csv> "<Equipo>" "<Entrenador>" <salida.json>
Línea base global exacta (aditiva) y métricas de ventana de 300 s tras cada evento disparador."""
import sys, json, gc, warnings
warnings.filterwarnings("ignore")
sys.path.append(".")
import numpy as np, pandas as pd
from framework_analitico import (normalizar_direccion_ataque, identificar_eventos_disparadores, metricas_ventana)
ruta, EQ, ENT, salida = sys.argv[1:5]
ev = pd.read_csv(ruta, low_memory=False) if ruta.endswith(".csv") else pd.read_pickle(ruta)
keep = [c for c in ev.columns if c in {"match_id","period","minute","second","type","team","location","possession_team","pass_length",
        "shot_outcome","foul_committed_card","bad_behaviour_card","index","player","home_team","away_team","shot_statsbomb_xg","x","y"}]
ev = ev[keep].copy(); gc.collect()
ids = list(ev.match_id.unique()); print("partidos", len(ids), flush=True)
chunks = [ids[k:k+15] for k in range(0, len(ids), 15)]
acc = dict(ev_tot=0, pos=0, pl_n=0, pl_d=0, pres=0, minutos=0.0, x_sum=0.0, x_n=0, tf=0, tc=0)
filas = []
for i, c in enumerate(chunks):
    sub = normalizar_direccion_ataque(ev[ev.match_id.isin(c)].copy(), equipo=EQ)
    et = sub[sub.team == EQ]
    acc["ev_tot"] += len(sub); acc["pos"] += int((sub.possession_team == EQ).sum())
    p = et[et.type == "Pass"]
    if "pass_length" in p.columns:
        acc["pl_d"] += int(p.pass_length.notna().sum()); acc["pl_n"] += int((p.pass_length * 0.9144 >= 18).sum())
    acc["pres"] += int((et.type == "Pressure").sum())
    acc["minutos"] += float(sub.groupby(["match_id","period"])["minute"].max().sum())
    xs = et["x"].dropna(); acc["x_sum"] += float(xs.sum()); acc["x_n"] += len(xs)
    acc["tf"] += int((et.type == "Shot").sum()); acc["tc"] += int((sub[sub.team != EQ].type == "Shot").sum())
    dis = identificar_eventos_disparadores(sub, EQ)
    for _, r in dis.iterrows():
        post = metricas_ventana(sub, EQ, r.match_id, r.period, r.t_segundos, 300)
        filas.append({"match_id": r.match_id, "tipo": r.tipo_disparador, **post})
    del sub, et; gc.collect(); print("bloque", i+1, "/", len(chunks), flush=True)
base = {"posesion_pct": acc["pos"]/acc["ev_tot"]*100, "pct_pases_largos": acc["pl_n"]/acc["pl_d"]*100,
        "presiones_por_min": acc["pres"]/acc["minutos"], "altura_promedio_x": acc["x_sum"]/acc["x_n"],
        "tiros_favor_por_min": acc["tf"]/acc["minutos"], "tiros_contra_por_min": acc["tc"]/acc["minutos"]}
filas = [{**f, **{k: (f[k] - base[k]) if f.get(k) is not None else None for k in base}} for f in filas]
df = pd.DataFrame(filas)
out = {"entrenador": ENT, "equipo": EQ, "base": base, "tipos": {}}
for t, g in df.groupby("tipo"):
    out["tipos"][t] = {"n": int(len(g)), **{k: (None if g[k].dropna().empty else round(float(g[k].mean()), 5)) for k in base}}
json.dump(out, open(salida, "w"), ensure_ascii=False, indent=1)
print(json.dumps(out["tipos"], ensure_ascii=False))
