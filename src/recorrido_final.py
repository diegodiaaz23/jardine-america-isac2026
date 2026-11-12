import json
r=json.load(open("../data/processed/recorrido_torneos.json"))
MAN={("Toluca","Apertura 2025"):"Campeón (global 2-2, penales)",("Atlas","Apertura 2021"):"Campeón (global 3-3, penales)"}
ORD={"Repechaje":1,"Cuartos":2,"Semifinal":3,"Final":4}
out=[]
for t in r:
    L=t["liguilla"]
    if t["equipo"]=="Atlas" and t["torneo"]=="Apertura 2021" and L: L[-1]["etapa"]="Final"
    if not L:
        res,nivel="No clasificó a liguilla",0
    else:
        u=L[-1]; nivel=ORD[u["etapa"]]
        if u["etapa"]=="Final":
            if (t["equipo"],t["torneo"]) in MAN: res=MAN[(t["equipo"],t["torneo"])]
            else: res="Campeón" if u["gf"]>u["gc"] else "Subcampeón"
            nivel=5 if res.startswith("Campe") else 4
        else: res="Eliminado en "+u["etapa"].lower().replace("cuartos","cuartos de final").replace("semifinal","semifinal")
    t["resultado"]=res; t["nivel"]=nivel
    t["en_curso"]=t["torneo"]=="Apertura 2026"
    if t["en_curso"]: t["resultado"]="En curso"
    out.append(t)
json.dump(out,open("../data/processed/recorrido_final.json","w"),ensure_ascii=False)
for t in out: print(t["equipo"],t["torneo"],t["partidos_regulares"],t["resultado"])
