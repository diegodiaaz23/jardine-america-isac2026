import sys, warnings
warnings.filterwarnings("ignore"); sys.path.append("../src")
import pandas as pd
from framework_analitico import normalizar_direccion_ataque
EQ="América"
P=pd.read_csv("../data/raw/partidos_jardine_america.csv")
ev=pd.read_csv("../data/raw/events_jardine_america.csv",low_memory=False)
for fecha,riv,mn in [("2024-10-20","Santos",57),("2026-01-15","San Luis",19)]:
    r=P[(P.match_date==fecha)&((P.home_team.str.contains(riv))|(P.away_team.str.contains(riv)))].iloc[0]
    sub=ev[ev.match_id==r.match_id].copy()
    print(fecha,r.home_team,r.home_score,r.away_score,r.match_id)
    en=normalizar_direccion_ataque(sub.copy(),equipo=EQ); en["t"]=en.minute*60+en.second
    cards=[c for c in en.columns if 'card' in c.lower()]
    cc=en[en[cards].notna().any(axis=1)] if cards else en.iloc[0:0]
    print(cc[["period","minute","second","team","player"]+cards].to_string()[:800])
    # cutoff: tarjeta roja en minuto mn
    red=cc[(cc.minute>=mn-1)&(cc.minute<=mn+1)]
    tc=int(red.t.iloc[0]) if len(red) else mn*60
    per=int(red.period.iloc[0]) if len(red) else 1
    def stats(w):
        am=w[w.team==EQ]; rv=w[w.team!=EQ]
        a=int((am.x>80).sum()); q=int((rv.x<40).sum())
        return dict(pos=round((w.possession_team==EQ).mean()*100,1),tiros=(int((am.type=="Shot").sum()),int((rv.type=="Shot").sum())),
          xg=(round(float(am.shot_statsbomb_xg.sum()),2),round(float(rv.shot_statsbomb_xg.sum()),2)),ft=round(a/(a+q)*100,1) if a+q else None,
          pases=(int((am.type=="Pass").sum()),int((rv.type=="Pass").sum())))
    w=en[(en.t<tc)&(en.period<=per)] if per==1 else en[(en.period==1)|((en.period==2)&(en.t<tc))]
    print("tc",tc,per,"acum",stats(w))
    w2=en[(en.period==per)&(en.t>=tc-900)&(en.t<tc)]; print("ult15",stats(w2))
    # goles posteriores
    g=en[(en.type=="Shot")&(en.shot_outcome=="Goal")][["period","minute","team"]]; print(g.values.tolist())
    s=en[en.type=="Substitution"][["period","minute","team"]]; print(s.values.tolist())
