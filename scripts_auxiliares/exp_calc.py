import sys, json, gc, warnings
warnings.filterwarnings("ignore"); sys.path.append("../src")
import numpy as np, pandas as pd
from framework_analitico import normalizar_direccion_ataque, identificar_eventos_disparadores, metricas_ventana, identificar_goles
EQ="América"
ev=pd.read_csv("../data/raw/events_jardine_america.csv",low_memory=False)
keep=[c for c in ev.columns if c in {"match_id","period","minute","second","type","team","location","possession_team","pass_length","shot_outcome","foul_committed_card","bad_behaviour_card","index","player","home_team","away_team","shot_statsbomb_xg","x","y","play_pattern","pass_type","pass_technique","pass_end_location","pass_outcome"}]
ev=ev[keep].copy()
P=pd.read_csv("../data/raw/partidos_jardine_america.csv")
P=P.set_index("match_id")
ids=list(ev.match_id.unique()); out=[]
for k in range(0,len(ids),15):
    sub=normalizar_direccion_ataque(ev[ev.match_id.isin(ids[k:k+15])].copy(),equipo=EQ)
    dis=identificar_eventos_disparadores(sub,EQ)
    dis=dis[dis.tipo_disparador.isin(["expulsion_propia","expulsion_rival"])]
    goles=identificar_goles(sub)
    for _,r in dis.iterrows():
        g=goles[(goles.match_id==r.match_id)]
        # marcador al momento (orden por t_segundos y periodo)
        gprev=g[(g.period<r.period)|((g.period==r.period)&(g.t_segundos<=r.t_segundos))]
        gf=int((gprev.team==EQ).sum()); gc_=int((gprev.team!=EQ).sum())
        post=metricas_ventana(sub,EQ,r.match_id,r.period,r.t_segundos,300)
        pre=metricas_ventana(sub,EQ,r.match_id,r.period,max(0,r.t_segundos-300),300)
        p=P.loc[r.match_id]
        local=p.home_team==EQ
        rival=p.away_team if local else p.home_team
        fin=(p.home_score,p.away_score) if local else (p.away_score,p.home_score)
        out.append({"match_id":int(r.match_id),"tipo":r.tipo_disparador,"fecha":str(p.match_date),"rival":rival,"local":bool(local),
          "minuto":int(r.t_segundos//60),"period":int(r.period),"detalle":r.detalle,"marcador_antes":[gf,gc_],"final":[int(fin[0]),int(fin[1])],"pre":pre,"post":post})
    del sub; gc.collect()
json.dump(out,open("exp_events.json","w"),ensure_ascii=False,indent=1,default=str)
print(len(out))
for o in out: print(o["tipo"],o["fecha"],o["rival"],o["minuto"],o["marcador_antes"],o["final"],o["post"]["posesion_pct"],o["pre"]["posesion_pct"])
