import sys, json, warnings
warnings.filterwarnings("ignore"); sys.path.append("../src")
import pandas as pd, numpy as np
from framework_analitico import normalizar_direccion_ataque
EQ="América"
P=pd.read_csv("../data/raw/partidos_jardine_america.csv")
r=P[(P.match_date=="2025-05-19")].iloc[0]; print(r.home_team,r.away_team,r.home_score,r.away_score,r.match_id)
ev=pd.read_csv("../data/raw/events_jardine_america.csv",low_memory=False)
sub=ev[ev.match_id==r.match_id].copy()
en=normalizar_direccion_ataque(sub.copy(),equipo=EQ)
en=en[en.x.notna()].copy()
# minuto corrido: periodo 2 sigue del 45
en["m"]=np.where(en.period==1,en.minute,en.minute)  # StatsBomb minute ya es corrido
mx=int(en[en.period<=2].m.max())
a=en[(en.team==EQ)&(en.x>80)].groupby("m").size().reindex(range(mx+1),fill_value=0)
q=en[(en.team!=EQ)&(en.x<40)].groupby("m").size().reindex(range(mx+1),fill_value=0)
# rejilla 12x8 por minuto de acciones propias
g=np.zeros((mx+1,12,8),dtype=int)
own=en[en.team==EQ]
for m,x,y in zip(own.m,own.x,own.y):
    i=min(11,int(x//10)); j=min(7,int(y//10)); g[int(m),i,j]+=1
ev_=[]
for _,e in en[en.type=="Shot"].iterrows():
    ev_.append({"m":int(e.m),"team":"A" if e.team==EQ else "R","xg":round(float(e.shot_statsbomb_xg),2),"goal":bool(e.shot_outcome=="Goal"),"y":float(e.y) if e.y==e.y else 40,"x":float(e.x)})
sus=[{"m":int(e.m),"team":"A" if e.team==EQ else "R"} for _,e in en[en.type=="Substitution"].iterrows()]
json.dump({"home":r.home_team,"away":r.away_team,"hs":int(r.home_score),"as":int(r.away_score),"mx":mx,"a":a.tolist(),"q":q.tolist(),"grid":g.tolist(),"shots":ev_,"subs":sus},open("ft_min.json","w"),ensure_ascii=False)
print(mx,a.sum(),q.sum(),[ (s["m"],s["team"],s["xg"],s["goal"]) for s in ev_])
