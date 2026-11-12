import sys, json, warnings
warnings.filterwarnings("ignore"); sys.path.append("../src")
import numpy as np, pandas as pd
from framework_analitico import normalizar_direccion_ataque, identificar_goles, identificar_sustituciones
EQ="América"
P=pd.read_csv("../data/raw/partidos_jardine_america.csv")
want=[("2024-02-18","Pachuca"),("2024-12-13","Monterrey")]
ev=pd.read_csv("../data/raw/events_jardine_america.csv",low_memory=False)
out={}
for fecha,riv in want:
    row=P[(P.match_date==fecha)&((P.home_team==riv)|(P.away_team==riv))].iloc[0]
    mid=row.match_id; sub=ev[ev.match_id==mid].copy()
    sub["t"]=sub.minute*60+sub.second
    en=normalizar_direccion_ategy if False else normalizar_direccion_ataque(sub.copy(),equipo=EQ)
    en["t"]=en.minute*60+en.second
    gol=identificar_goles(sub)
    print(fecha,riv,mid,row.home_team,row.home_score,row.away_score)
    print("goles",gol[["period","minute","team"]].values.tolist())
    sus=identificar_sustituciones(sub,EQ) if True else None
    print("sust",sus.head(10).to_string() if len(sus) else "ninguna")
    out[riv]={"mid":int(mid)}
    # ventana hasta el gol
    for nombre,(m0) in [("pachuca",55),("monterrey",34)]:
        pass
    mg=55 if riv=="Pachuca" else 34
    # t del gol del rival en el minuto mg
    g=gol[(gol.team!=EQ)&(gol.minute==mg)]
    tg=int(g.t_segundos.iloc[0]); per=int(g.period.iloc[0]); print("t gol",tg,per)
    def win(t0,t1):
        w=en[(en.period<=per)&(en.t>=t0)&(en.t<t1)] if False else en[(en.t>=t0)&(en.t<t1)&(en.period==per)]
        am=w[w.team==EQ]; rv=w[w.team!=EQ]
        d={"posesion":round((w.possession_team==EQ).mean()*100,1),
           "tiros_am":int((am.type=="Shot").sum()),"tiros_rv":int((rv.type=="Shot").sum()),
           "xg_am":round(float(am.shot_statsbomb_xg.sum()),2),"xg_rv":round(float(rv.shot_statsbomb_xg.sum()),2)}
        a=int((am.x>80).sum()); r=int((rv.x<40).sum()); d["field_tilt_rel"]=round(a/(a+r)*100,1) if a+r else None
        return d
    # partido hasta el gol: período 1 completo + período 2 hasta tg (si per==2)
    if per==2:
        w1=en[(en.period==1)]; w2=en[(en.period==2)&(en.t<tg)]; w=pd.concat([w1,w2])
    else:
        w=en[(en.period==1)&(en.t<tg)]
    am=w[w.team==EQ]; rv=w[w.team!=EQ]
    a=int((am.x>80).sum()); r=int((rv.x<40).sum())
    tot={"posesion":round((w.possession_team==EQ).mean()*100,1),"tiros_am":int((am.type=="Shot").sum()),"tiros_rv":int((rv.type=="Shot").sum()),
         "xg_am":round(float(am.shot_statsbomb_xg.sum()),2),"xg_rv":round(float(rv.shot_statsbomb_xg.sum()),2),"field_tilt_rel":round(a/(a+r)*100,1)}
    ult=win(tg-900,tg)
    print("acumulado",tot); print("ultimos 15 min",ult)
    # post: sustituciones y goles después
    posts=sus[(sus.period>per)|((sus.period==per)&((sus.minute*60+sus.second)>=tg))] if len(sus) and 'second' in sus.columns else sus
    print("sust post",posts.to_string() if len(posts) else "ninguna")
