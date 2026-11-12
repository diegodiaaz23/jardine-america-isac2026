"""
Framework analítico - Hackathon ISAC 2026
------------------------------------------
Definiciones y funciones para construir los objetos base del análisis táctico:
posesión, secuencia ofensiva, transición y fase de presión.

Estas definiciones deben citarse tal cual en la sección "Framework Analítico"
del entregable, ya que el reto exige coherencia interna: todas las métricas
y visualizaciones deben derivarse de estos mismos objetos y supuestos.

Requiere un events_df construido con statsbombpy (sb.events / sb.competition_events)
que incluya, como mínimo, las columnas:
    match_id, period, index, minute, second, timestamp, duration,
    type, team, possession, possession_team
Si concatenas varios partidos, asegúrate de que exista una columna match_id
que distinga cada uno (sb.events no siempre la incluye por default: agrégala
manualmente si hace falta, ej. events_df["match_id"] = match_id).
"""

import ast
import numpy as np
import pandas as pd


def _parse_coord_column(serie: pd.Series) -> pd.DataFrame:
    """
    Convierte una columna de coordenadas tipo [x, y] a dos columnas x, y.

    Soporta dos formatos:
      - Listas/tuplas reales de Python (cuando events_df viene directo de
        statsbombpy, sin pasar por un archivo).
      - Strings tipo "[60.5, 40.2]" (cuando events_df se guardó y volvió a
        cargar desde un CSV, que convierte las listas a texto).

    Los valores nulos/ausentes (ej. eventos sin ubicación) se devuelven
    como [NaN, NaN] en vez de causar un error.

    Nota: en los eventos de tipo "Shot", StatsBomb guarda 3 valores
    [x, y, z] (z = altura del balón sobre el suelo al momento del disparo).
    Para el resto de los eventos solo hay [x, y]. Esta función siempre
    se queda únicamente con los primeros dos valores (x, y) para que todas
    las filas generen un resultado de longitud consistente.
    """
    def _parsear(valor):
        if isinstance(valor, (list, tuple)):
            coords = valor
        elif isinstance(valor, str):
            try:
                coords = ast.literal_eval(valor)
            except (ValueError, SyntaxError):
                return [np.nan, np.nan]
        else:
            return [np.nan, np.nan]

        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            return [np.nan, np.nan]

        # Nos quedamos solo con x, y (descarta z si existe, ej. en disparos)
        return [coords[0], coords[1]]

    return pd.DataFrame(serie.apply(_parsear).tolist(), index=serie.index)

# ---------------------------------------------------------------------------
# 0. CONFIGURACIÓN DE SUPUESTOS (documentar tal cual en el entregable)
# ---------------------------------------------------------------------------
FRAMEWORK_CONFIG = {
    "min_passes_secuencia_ofensiva": 3,   # mínimo de pases para contar como secuencia ofensiva real
    "ventana_transicion_segundos": 8,     # segundos tras cambio de posesión que se consideran "transición"
    "gap_maximo_presion_segundos": 3,     # separación máxima entre eventos Pressure para agruparlos en una fase
}


# ---------------------------------------------------------------------------
# 1. POSESIÓN
# ---------------------------------------------------------------------------
def build_possessions(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Definición: una posesión es una secuencia continua de eventos de un mismo
    equipo, delimitada por el campo `possession` que ya calcula StatsBomb
    (cambia cuando el balón pasa al rival o hay un reinicio de balón parado).

    Devuelve un resumen por posesión: equipo, partido, número de eventos,
    número de pases, duración total y si terminó en tiro.
    """
    required_cols = {"match_id", "possession", "possession_team", "type", "duration"}
    missing = required_cols - set(events_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en events_df: {missing}")

    grouped = events_df.groupby(["match_id", "possession"])

    resumen = grouped.agg(
        team=("possession_team", "first"),
        n_eventos=("type", "count"),
        n_pases=("type", lambda s: (s == "Pass").sum()),
        duracion_total=("duration", "sum"),
        termina_en_tiro=("type", lambda s: "Shot" in s.values),
    ).reset_index()

    return resumen


# ---------------------------------------------------------------------------
# 2. SECUENCIA OFENSIVA
# ---------------------------------------------------------------------------
def extract_offensive_sequences(
    events_df: pd.DataFrame,
    min_passes: int = FRAMEWORK_CONFIG["min_passes_secuencia_ofensiva"],
) -> pd.DataFrame:
    """
    Definición: una secuencia ofensiva es una posesión (ver build_possessions)
    con al menos `min_passes` pases.

    Supuesto explícito: una posesión con menos pases que el umbral (ej. un
    saque que se pierde de inmediato) NO cuenta como secuencia ofensiva, ya
    que no representa una intención real de construcción/progresión.
    """
    posesiones = build_possessions(events_df)
    secuencias = posesiones[posesiones.n_pases >= min_passes].copy()
    secuencias["es_secuencia_ofensiva"] = True
    return secuencias


# ---------------------------------------------------------------------------
# 3. TRANSICIÓN
# ---------------------------------------------------------------------------
def identify_transitions(
    events_df: pd.DataFrame,
    window_seconds: float = FRAMEWORK_CONFIG["ventana_transicion_segundos"],
) -> pd.DataFrame:
    """
    Definición: una transición son los eventos ocurridos dentro de los
    `window_seconds` posteriores a un cambio de posesión (cambio de equipo
    en `possession_team`). Se etiqueta cada evento marcado como:
      - "ofensiva"  -> del equipo que ACABA de recuperar el balón
      - "defensiva" -> del equipo que ACABA de perderlo

    Supuesto explícito: la transición termina automáticamente a los
    `window_seconds`, sin importar si el equipo conserva el balón más
    tiempo. Esto delimita el objeto para que sea comparable entre partidos.
    """
    df = events_df.sort_values(["match_id", "period", "index"]).copy()
    df["t_segundos"] = df["minute"] * 60 + df["second"]
    df["cambio_posesion"] = (
        df.groupby("match_id")["possession_team"].shift(1) != df["possession_team"]
    )

    cambios = df[df["cambio_posesion"]]
    transiciones = []

    for _, row in cambios.iterrows():
        ventana = df[
            (df.match_id == row.match_id)
            & (df.period == row.period)
            & (df.t_segundos >= row.t_segundos)
            & (df.t_segundos < row.t_segundos + window_seconds)
        ].copy()

        ventana["tipo_transicion"] = np.where(
            ventana.possession_team == row.possession_team, "ofensiva", "defensiva"
        )
        ventana["transicion_id"] = f"{row.match_id}_{row.period}_{row.t_segundos}"
        transiciones.append(ventana)

    if not transiciones:
        return pd.DataFrame()

    return pd.concat(transiciones, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. FASE DE PRESIÓN
# ---------------------------------------------------------------------------
def identify_pressure_phases(
    events_df: pd.DataFrame,
    gap_seconds: float = FRAMEWORK_CONFIG["gap_maximo_presion_segundos"],
) -> pd.DataFrame:
    """
    Definición: una fase de presión agrupa eventos consecutivos de tipo
    "Pressure" de un mismo equipo en un mismo partido. Dos eventos Pressure
    pertenecen a la misma fase si la diferencia de tiempo entre ellos es
    menor o igual a `gap_seconds`.

    Supuesto explícito: esta primera versión solo usa criterio temporal (no
    espacial). Si el framework lo requiere, se puede refinar agregando un
    criterio de zona del campo (ej. distancia entre `location` de eventos
    consecutivos) para no agrupar presiones en zonas distintas del campo.
    """
    pressure_events = events_df[events_df.type == "Pressure"].copy()
    if pressure_events.empty:
        return pd.DataFrame()

    pressure_events["t_segundos"] = pressure_events["minute"] * 60 + pressure_events["second"]
    pressure_events = pressure_events.sort_values(["match_id", "team", "t_segundos"])

    pressure_events["gap"] = pressure_events.groupby(["match_id", "team"])["t_segundos"].diff()
    pressure_events["nueva_fase"] = (
        (pressure_events["gap"] > gap_seconds) | (pressure_events["gap"].isna())
    )
    pressure_events["fase_presion_id"] = (
        pressure_events.groupby(["match_id", "team"])["nueva_fase"].cumsum()
    )

    resumen_fases = (
        pressure_events.groupby(["match_id", "team", "fase_presion_id"])
        .agg(
            n_eventos_presion=("type", "count"),
            inicio_segundos=("t_segundos", "min"),
            fin_segundos=("t_segundos", "max"),
        )
        .reset_index()
    )
    resumen_fases["duracion_fase"] = resumen_fases.fin_segundos - resumen_fases.inicio_segundos

    return resumen_fases


# ---------------------------------------------------------------------------
# 5. NORMALIZACIÓN DE DIRECCIÓN DE ATAQUE (genérica, por equipo)
# ---------------------------------------------------------------------------
def normalizar_direccion_ataque(events_df: pd.DataFrame, equipo: str) -> pd.DataFrame:
    """
    Voltea las coordenadas x, y para que `equipo` siempre "ataque" hacia
    x=120, sin importar el periodo o si jugó de local/visitante, Y para que
    el rival quede en el MISMO marco de cancha compartido (atacando hacia
    x=0, defendiendo junto a x=120).

    Genérica: recibe el nombre del equipo como parámetro, así que funciona
    igual para Jardine/América que para cualquier otro entrenador/equipo
    (ver "Principio de reusabilidad" en el Plan Maestro).

    IMPORTANTE (verificado empíricamente sobre este dataset, 257/258
    combinaciones partido-periodo): StatsBomb NO graba el campo `location`
    en un único marco físico compartido por partido. Cada equipo registra
    sus propios eventos SIEMPRE "atacando" hacia x=120 en su propio marco,
    sin importar el lado real de la cancha o el periodo. Esto significa
    que las coordenadas de `equipo` y las del rival, en su forma cruda,
    NO son comparables entre sí: si se grafican juntas sin corregir esto,
    ambos equipos aparecen atacando el mismo extremo.

    Por eso esta función voltea cada grupo de eventos por separado:
      - Los eventos de `equipo` se orientan para que ataque hacia x=120
        (usando como referencia sus propios tiros, por partido y periodo;
        en la práctica casi nunca hace falta voltear, porque StatsBomb ya
        los entrega así, pero se deja la detección explícita por robustez
        y para no asumir el convenio a ciegas en otro dataset).
      - Los eventos del rival se voltean con la decisión CONTRARIA a la de
        `equipo` en ese mismo partido/periodo, para que terminen en el
        mismo marco compartido, atacando hacia x=0 (el extremo opuesto).

    Requiere la columna `location` cruda o columnas x, y ya parseadas.
    """
    df = events_df.copy()
    if "x" not in df.columns or "y" not in df.columns:
        df[["x", "y"]] = _parse_coord_column(df["location"])

    tiros = df[(df.team == equipo) & (df.type == "Shot")]
    referencia = tiros.groupby(["match_id", "period"])["x"].mean().reset_index()
    referencia["atacando_derecha"] = referencia["x"] > 60

    df = df.merge(referencia[["match_id", "period", "atacando_derecha"]], on=["match_id", "period"], how="left")
    df["atacando_derecha"] = df["atacando_derecha"].fillna(True)

    es_equipo = df["team"] == equipo
    # equipo: voltea cuando NO está atacando a la derecha.
    # rival: voltea con la decisión CONTRARIA (su propio marco ya "atacaba a
    # la derecha" en el sentido de equipo, así que hay que espejarlo).
    volteo = np.where(es_equipo, ~df["atacando_derecha"], df["atacando_derecha"])

    df.loc[volteo, "x"] = 120 - df.loc[volteo, "x"]
    df.loc[volteo, "y"] = 80 - df.loc[volteo, "y"]

    df = df.drop(columns=["atacando_derecha"])

    return df


# ---------------------------------------------------------------------------
# 6. PERFIL OFENSIVO / DEFENSIVO (equipo vs. rivales, dentro del mismo dataset)
# ---------------------------------------------------------------------------
def perfil_ofensivo_defensivo(
    events_df: pd.DataFrame,
    equipo: str,
    umbral_ppda_x: float = 72,
) -> dict:
    """
    Responde la pregunta 4 del Plan Maestro: ¿el equipo juega más ofensivo
    o más defensivo? Compara `equipo` contra el conjunto de sus rivales
    DENTRO del mismo dataset de partidos (nunca contra la liga completa,
    para no mezclar competiciones o épocas distintas).

    Genérica: recibe `equipo` como parámetro; correr esto para otro
    entrenador es cambiar ese argumento y el events_df de entrada.

    Requiere coordenadas x, y normalizadas (normalizar_direccion_ataque).
    Como esa normalización voltea TODO el partido/periodo según hacia dónde
    ataca `equipo`, el rival dentro de ese mismo partido/periodo queda
    atacando hacia x=0: eso es lo que permite comparar field tilt y PPDA de
    forma simétrica entre ambos equipos.

    Devuelve un diccionario con:
      - posesion_pct: % de posesiones que fueron del equipo.
      - field_tilt_pct: % de los PROPIOS eventos del equipo que ocurren en
        su tercio de ataque (x > 80) — una medida de cuánto juega el equipo
        metido en campo rival, sin comparar contra el rival.
      - field_tilt_relativo_pct: el "field tilt" en el sentido clásico de la
        industria — de todos los eventos de AMBOS equipos en sus respectivos
        tercios de ataque, qué % son del equipo (eventos propios en x>80
        frente a eventos del rival en x<40, su tercio de ataque equivalente).
      - ppda: pases permitidos al rival por cada acción defensiva propia
        (Pressure + Duel + Interception), ambos contados en x > umbral_ppda_x.
        Supuesto explícito: el valor por defecto (72) aproxima el 40% final
        del campo medido desde la portería rival — la definición estándar
        de PPDA en la industria (Trainor et al.) — en vez de medio campo
        completo (x>60), que sobreestima la zona de presión y deprime
        artificialmente el valor de PPDA.

        NOTA DE CAUTELA (documentada, no resuelta): aun con ese ajuste de
        zona, el PPDA calculado con eventos de StatsBomb sale muy por debajo
        de los benchmarks publicados (que suelen rondar 8-15; aquí sale
        ~2-2.5). La causa probable es que StatsBomb registra el evento
        `Duel` de forma mucho más granular que otros proveedores (ej. Opta,
        la fuente de la mayoría de los benchmarks públicos de PPDA) —
        prácticamente cada disputa individual de balón, no solo entradas —
        lo que infla las "acciones defensivas" del denominador. Por eso este
        PPDA NO debe compararse contra benchmarks externos; solo es válido
        de forma RELATIVA, entre partidos o entre equipos calculados con
        esta misma función y la misma fuente de datos.
      - tiros_favor / tiros_contra
      - xg_favor / xg_contra (si existe la columna shot_statsbomb_xg)
    """
    posesiones = build_possessions(events_df)
    posesion_pct = (posesiones.team == equipo).mean() * 100

    if "x" not in events_df.columns:
        raise ValueError(
            "events_df necesita coordenadas x, y normalizadas. "
            "Corre normalizar_direccion_ataque(events_df, equipo) primero."
        )

    eventos_equipo = events_df[events_df.team == equipo]
    eventos_rival = events_df[events_df.team != equipo]

    field_tilt_pct = (eventos_equipo.x > 80).mean() * 100  # tercio de ataque, x normalizada

    tercio_ataque_equipo = (eventos_equipo.x > 80).sum()
    tercio_ataque_rival = (eventos_rival.x < 40).sum()
    total_tercio_ataque = tercio_ataque_equipo + tercio_ataque_rival
    field_tilt_relativo_pct = (
        tercio_ataque_equipo / total_tercio_ataque * 100
        if total_tercio_ataque > 0 else np.nan
    )

    # PPDA: pases del rival permitidos por cada acción defensiva del equipo,
    # ambos contados en x > umbral_ppda_x (por defecto, 40% final del campo
    # rival, no medio campo completo)
    acciones_defensivas = events_df[
        (events_df.team == equipo)
        & (events_df.type.isin(["Pressure", "Duel", "Interception"]))
        & (events_df.x > umbral_ppda_x)
    ]
    pases_rival_permitidos = events_df[
        (events_df.team != equipo) & (events_df.type == "Pass") & (events_df.x > umbral_ppda_x)
    ]
    ppda = (
        len(pases_rival_permitidos) / len(acciones_defensivas)
        if len(acciones_defensivas) > 0 else np.nan
    )

    tiros_favor = (eventos_equipo.type == "Shot").sum()
    tiros_contra = (eventos_rival.type == "Shot").sum()

    resultado = {
        "equipo": equipo,
        "posesion_pct": round(posesion_pct, 1),
        "field_tilt_pct": round(field_tilt_pct, 1),
        "field_tilt_relativo_pct": (
            round(field_tilt_relativo_pct, 1) if not np.isnan(field_tilt_relativo_pct) else None
        ),
        "ppda": round(ppda, 2) if not np.isnan(ppda) else None,
        "umbral_ppda_x": umbral_ppda_x,
        "tiros_favor": int(tiros_favor),
        "tiros_contra": int(tiros_contra),
    }

    if "shot_statsbomb_xg" in events_df.columns:
        xg_favor = eventos_equipo.loc[eventos_equipo.type == "Shot", "shot_statsbomb_xg"].sum()
        xg_contra = eventos_rival.loc[eventos_rival.type == "Shot", "shot_statsbomb_xg"].sum()
        resultado["xg_favor"] = round(xg_favor, 2)
        resultado["xg_contra"] = round(xg_contra, 2)

    return resultado


def indice_directividad(
    events_df: pd.DataFrame, equipo: str, umbral_metros: float = 18
) -> dict:
    """
    Aproxima qué tan directo juega el equipo EN GENERAL (a diferencia de
    salida_corta_vs_larga, que solo mira pases desde el tercio defensivo):
    calcula el % de TODOS los pases del equipo que son largos
    (>= umbral_metros) y la longitud promedio de pase, usando pass_length
    de StatsBomb (en yardas) cuando existe, o la distancia end-start como
    fallback.

    Responde, junto con perfil_ofensivo_defensivo, la pregunta 4 del Plan
    Maestro (¿ofensivo o defensivo?, y dentro de ofensivo, ¿directo o
    elaborado?). Un índice alto sugiere un equipo que busca progresar
    rápido con pases largos en vez de construir con posesión controlada.

    Genérica: recibe `equipo` como parámetro.
    """
    pases = events_df[(events_df.team == equipo) & (events_df.type == "Pass")].copy()

    if "pass_length" in pases.columns:
        pases["longitud_m"] = pases["pass_length"] * 0.9144
    else:
        pases[["end_x", "end_y"]] = _parse_coord_column(pases["pass_end_location"])
        pases["longitud_m"] = np.sqrt(
            (pases.end_x - pases.x) ** 2 + (pases.end_y - pases.y) ** 2
        ) * 0.9144

    return {
        "pct_pases_largos": round((pases.longitud_m >= umbral_metros).mean() * 100, 1),
        "longitud_promedio_m": round(pases.longitud_m.mean(), 1),
        "total_pases": len(pases),
    }


# ---------------------------------------------------------------------------
# 7. CLASIFICACIÓN EMPÍRICA DE INTENSIDAD DE PRESIÓN (dos etapas)
# ---------------------------------------------------------------------------
def clasificar_intensidad_presion(
    fases_presion: pd.DataFrame,
    equipo: str,
    percentil_sostenida: float = 50,
) -> tuple:
    """
    Clasifica cada fase de presión de `equipo` en baja/media/alta, EN DOS
    ETAPAS. Este criterio en dos etapas (y no un corte directo por
    percentiles 33/66 sobre toda la duración) se definió después de graficar
    la distribución real: en la mayoría de los equipos, ~70-80% de las
    fases de presión son un solo evento Pressure aislado (sin otro evento
    del mismo equipo a ≤gap_maximo_presion_segundos de distancia), con
    duración 0s. Cuando la mayoría de los valores son idénticos, un corte
    por percentiles 33/66 colapsa (ambos cortes caen en el mismo valor y la
    categoría "media" queda vacía) — ver
    outputs/figures/distribucion_fases_presion.png para la evidencia visual.

    Etapa 1 — "puntual" vs. "sostenida": una fase con un solo evento
    Pressure (n_eventos_presion <= 1) se clasifica directamente como "baja"
    (presión puntual, no sostenida).

    Etapa 2 — dentro de las fases SOSTENIDAS (n_eventos_presion >= 2), se
    calcula un corte por percentil (`percentil_sostenida`, mediana = 50 por
    defecto) sobre `duracion_fase`, SOLO en ese subconjunto: igual o por
    debajo del corte es "media", por encima es "alta".

    Genérica: el corte de la etapa 2 se recalcula sobre las fases
    SOSTENIDAS de ESE equipo, así que se adapta al estilo de cada
    entrenador (ej. Almada/Pachuca tendrá su propio corte, distinto al de
    Jardine/América).

    Devuelve (fases_clasificadas, info) donde `info` es un diccionario con
    el % de fases puntuales, el corte de duración usado entre media/alta
    (en segundos, solo dentro de las sostenidas) y el conteo por categoría,
    para poder documentar y defender el criterio con el organizador.
    """
    fases_equipo = fases_presion[fases_presion.team == equipo].copy()

    puntuales = fases_equipo["n_eventos_presion"] <= 1
    sostenidas = fases_equipo.loc[~puntuales]

    fases_equipo["intensidad"] = "baja"  # etapa 1: puntuales por defecto

    corte = np.nan
    if not sostenidas.empty:
        corte = np.percentile(sostenidas["duracion_fase"], percentil_sostenida)
        es_media = ~puntuales & (fases_equipo["duracion_fase"] <= corte)
        es_alta = ~puntuales & (fases_equipo["duracion_fase"] > corte)
        fases_equipo.loc[es_media, "intensidad"] = "media"
        fases_equipo.loc[es_alta, "intensidad"] = "alta"

    info = {
        "pct_puntual": round(puntuales.mean() * 100, 1),
        "corte_sostenida_segundos": round(corte, 2) if not np.isnan(corte) else None,
        "percentil_usado": percentil_sostenida,
        "conteos": fases_equipo["intensidad"].value_counts().to_dict(),
    }

    return fases_equipo, info


# ---------------------------------------------------------------------------
# 8. FASE OFENSIVA (construcción, progresión, generación de ocasiones)
# ---------------------------------------------------------------------------
GOL_RIVAL = np.array([120, 40])  # centro de la portería rival, coordenadas StatsBomb normalizadas


def clasificar_zona(x: float) -> str:
    """Tercio del campo según x normalizada (0-120): defensivo / medio / ofensivo."""
    if x < 40:
        return "defensivo"
    elif x < 80:
        return "medio"
    return "ofensivo"


def identificar_pases_progresivos(
    events_df: pd.DataFrame, equipo: str, umbral_pct: float = 0.25
) -> pd.DataFrame:
    """
    Definición (StatsBomb): un pase es progresivo si reduce la distancia en
    línea recta al centro de la portería rival en al menos `umbral_pct`
    (25% por defecto — es el criterio que StatsBomb usa internamente, ver
    https://statsultra.com/progressive-passes-carries-received-explained/).

    Requiere coordenadas x, y normalizadas (normalizar_direccion_ataque) y
    la columna pass_end_location de StatsBomb.

    Supuesto explícito: un pase hacia atrás o lateral nunca cuenta (la
    reducción de distancia debe ser positiva), y se usa el mismo umbral sin
    importar la zona del campo.
    """
    pases = events_df[(events_df.team == equipo) & (events_df.type == "Pass")].copy()
    pases[["end_x", "end_y"]] = _parse_coord_column(pases["pass_end_location"])

    dist_inicio = np.sqrt((GOL_RIVAL[0] - pases.x) ** 2 + (GOL_RIVAL[1] - pases.y) ** 2)
    dist_fin = np.sqrt((GOL_RIVAL[0] - pases.end_x) ** 2 + (GOL_RIVAL[1] - pases.end_y) ** 2)
    reduccion_pct = (dist_inicio - dist_fin) / dist_inicio.replace(0, np.nan)

    pases["reduccion_distancia_pct"] = reduccion_pct
    pases["es_progresivo"] = reduccion_pct >= umbral_pct
    return pases


def identificar_conducciones_progresivas(
    events_df: pd.DataFrame, equipo: str, umbral_pct: float = 0.25
) -> pd.DataFrame:
    """
    Misma definición que identificar_pases_progresivos, aplicada a eventos
    `Carry` (conducciones) usando carry_end_location.
    """
    conducciones = events_df[(events_df.team == equipo) & (events_df.type == "Carry")].copy()
    conducciones[["end_x", "end_y"]] = _parse_coord_column(conducciones["carry_end_location"])

    dist_inicio = np.sqrt((GOL_RIVAL[0] - conducciones.x) ** 2 + (GOL_RIVAL[1] - conducciones.y) ** 2)
    dist_fin = np.sqrt((GOL_RIVAL[0] - conducciones.end_x) ** 2 + (GOL_RIVAL[1] - conducciones.end_y) ** 2)
    reduccion_pct = (dist_inicio - dist_fin) / dist_inicio.replace(0, np.nan)

    conducciones["reduccion_distancia_pct"] = reduccion_pct
    conducciones["es_progresiva"] = reduccion_pct >= umbral_pct
    return conducciones


def zona_inicio_secuencias(events_df: pd.DataFrame, equipo: str) -> pd.DataFrame:
    """
    Para cada secuencia ofensiva del equipo (posesión con ≥3 pases, ver
    extract_offensive_sequences), obtiene la zona de campo donde inició,
    usando el primer evento de esa posesión. Responde "¿desde dónde
    construye el equipo su juego?".
    """
    secuencias = extract_offensive_sequences(events_df)
    secuencias_equipo = secuencias[secuencias.team == equipo]

    primeros_eventos = (
        events_df.sort_values(["match_id", "index"])
        .groupby(["match_id", "possession"])
        .first()
        .reset_index()
    )
    merged = secuencias_equipo.merge(
        primeros_eventos[["match_id", "possession", "x"]], on=["match_id", "possession"]
    )
    merged["zona_inicio"] = merged["x"].apply(clasificar_zona)
    return merged


def salida_corta_vs_larga(
    events_df: pd.DataFrame, equipo: str, umbral_metros: float = 18
) -> pd.DataFrame:
    """
    Clasifica los pases del equipo que se originan en el tercio defensivo
    (salida) como "corta" (< umbral_metros) o "larga" (>= umbral_metros).
    Usa pass_length de StatsBomb (en yardas) cuando existe.

    Supuesto explícito: 18 metros (~20 yardas) es un corte convencional
    para separar un pase de salida corto de un pelotazo largo; ajustable
    según el estilo del equipo analizado.
    """
    pases = events_df[
        (events_df.team == equipo) & (events_df.type == "Pass") & (events_df.x < 40)
    ].copy()

    if "pass_length" in pases.columns:
        pases["longitud_m"] = pases["pass_length"] * 0.9144  # yardas -> metros
    else:
        pases[["end_x", "end_y"]] = _parse_coord_column(pases["pass_end_location"])
        pases["longitud_m"] = np.sqrt(
            (pases.end_x - pases.x) ** 2 + (pases.end_y - pases.y) ** 2
        ) * 0.9144

    pases["tipo_salida"] = np.where(pases.longitud_m >= umbral_metros, "larga", "corta")
    return pases


def generacion_ocasiones(events_df: pd.DataFrame, equipo: str) -> pd.DataFrame:
    """
    Para cada tiro del equipo, extrae su xG (shot_statsbomb_xg, si existe)
    y el tipo de jugada previa (play_pattern nativo de StatsBomb: Regular
    Play, From Corner, From Free Kick, From Counter, From Throw In, etc.).
    Responde directamente "¿de qué tipo de jugada nacen las ocasiones?".
    """
    tiros = events_df[(events_df.team == equipo) & (events_df.type == "Shot")].copy()
    cols = ["match_id", "minute", "second", "player", "play_pattern", "x", "y"]
    if "shot_statsbomb_xg" in tiros.columns:
        cols.append("shot_statsbomb_xg")
    return tiros[[c for c in cols if c in tiros.columns]]


# ---------------------------------------------------------------------------
# 9. FASE DEFENSIVA (zona de presión, intensidad, línea defensiva, contrapresión)
# ---------------------------------------------------------------------------
def fases_presion_con_zona(
    events_df: pd.DataFrame,
    gap_seconds: float = FRAMEWORK_CONFIG["gap_maximo_presion_segundos"],
) -> pd.DataFrame:
    """
    Igual que identify_pressure_phases, pero añade la zona de campo
    (defensivo/medio/ofensivo) donde ocurrió cada fase, usando el x promedio
    (normalizado) de los eventos Pressure que la componen.

    Requiere coordenadas x, y normalizadas con normalizar_direccion_ataque,
    porque la zona se define respecto a la dirección de ataque DEL EQUIPO QUE
    PRESIONA: x alto (zona "ofensivo") significa que presiona cerca de la
    portería rival (presión alta/adelantada); x bajo (zona "defensivo")
    significa que presiona cerca de su propia portería (bloque bajo).

    Responde directamente "¿dónde y con qué frecuencia presiona el equipo?".
    """
    if "x" not in events_df.columns:
        raise ValueError(
            "events_df necesita coordenadas x, y normalizadas. "
            "Corre normalizar_direccion_ataque(events_df, equipo) primero."
        )

    pressure_events = events_df[events_df.type == "Pressure"].copy()
    if pressure_events.empty:
        return pd.DataFrame()

    pressure_events["t_segundos"] = pressure_events["minute"] * 60 + pressure_events["second"]
    pressure_events = pressure_events.sort_values(["match_id", "team", "t_segundos"])

    pressure_events["gap"] = pressure_events.groupby(["match_id", "team"])["t_segundos"].diff()
    pressure_events["nueva_fase"] = (
        (pressure_events["gap"] > gap_seconds) | (pressure_events["gap"].isna())
    )
    pressure_events["fase_presion_id"] = (
        pressure_events.groupby(["match_id", "team"])["nueva_fase"].cumsum()
    )

    resumen = (
        pressure_events.groupby(["match_id", "team", "fase_presion_id"])
        .agg(
            n_eventos_presion=("type", "count"),
            inicio_segundos=("t_segundos", "min"),
            fin_segundos=("t_segundos", "max"),
            x_promedio=("x", "mean"),
        )
        .reset_index()
    )
    resumen["duracion_fase"] = resumen.fin_segundos - resumen.inicio_segundos
    resumen["zona"] = resumen["x_promedio"].apply(clasificar_zona)
    return resumen


def altura_linea_defensiva(events_df: pd.DataFrame, equipo: str) -> pd.DataFrame:
    """
    Aproxima la altura de la línea defensiva del equipo, partido por partido,
    como el x promedio (normalizado: 0 = línea de meta propia, 120 = meta
    rival) de sus acciones defensivas (Interception, Clearance, Duel, Block).
    Un x promedio alto sugiere un bloque adelantado; uno bajo, un repliegue.

    Requiere coordenadas x, y normalizadas (normalizar_direccion_ataque).

    Supuesto explícito: se usan estos 4 tipos de evento como proxy de "acción
    defensiva" porque StatsBomb no registra el marcaje posicional sin balón;
    es una aproximación, no la línea defensiva real en cada instante.
    """
    if "x" not in events_df.columns:
        raise ValueError(
            "events_df necesita coordenadas x, y normalizadas. "
            "Corre normalizar_direccion_ataque(events_df, equipo) primero."
        )

    tipos_defensivos = ["Interception", "Clearance", "Duel", "Block"]
    acciones = events_df[
        (events_df.team == equipo) & (events_df.type.isin(tipos_defensivos))
    ].copy()

    por_partido = (
        acciones.groupby("match_id")["x"]
        .mean()
        .reset_index(name="altura_linea_x")
    )
    return por_partido


def tiempo_reaccion_perdida(
    events_df: pd.DataFrame,
    equipo: str,
    ventana_segundos: float = FRAMEWORK_CONFIG["ventana_transicion_segundos"],
) -> pd.DataFrame:
    """
    Para cada pérdida de balón del equipo (cambio de possession_team desde
    `equipo` hacia el rival), mide cuántos segundos tarda el equipo en
    generar su primer evento Pressure dentro de la ventana de transición
    (ventana_segundos). Aproxima qué tan rápido se reorganiza para
    contrapresionar tras perder el balón.

    Supuesto explícito: si no hay ningún evento Pressure del equipo dentro
    de la ventana, esa pérdida se reporta con tiempo_reaccion = NaN y
    contrapresiono = False (no contrapresionó a tiempo, según este criterio).
    """
    df = events_df.sort_values(["match_id", "period", "index"]).copy()
    df["t_segundos"] = df["minute"] * 60 + df["second"]
    df["equipo_anterior"] = df.groupby("match_id")["possession_team"].shift(1)
    df["cambio_posesion"] = df["equipo_anterior"] != df["possession_team"]

    perdidas = df[
        df["cambio_posesion"]
        & (df["equipo_anterior"] == equipo)
        & (df["possession_team"] != equipo)
    ]

    # Versión vectorizada: para cada (partido, periodo) se ordenan los tiempos
    # de los Pressure del equipo y se busca con searchsorted el primero que
    # cae dentro de [t_perdida, t_perdida + ventana). Equivale al filtro
    # fila por fila original, pero en segundos en vez de minutos.
    presiones = (
        df[(df.team == equipo) & (df.type == "Pressure")]
        .groupby(["match_id", "period"])["t_segundos"]
        .apply(lambda s: np.sort(s.values))
        .to_dict()
    )

    resultados = []
    for (match_id, period), grupo in perdidas.groupby(["match_id", "period"]):
        tiempos = presiones.get((match_id, period), np.array([]))
        t_perdida = grupo["t_segundos"].values
        if len(tiempos):
            pos = np.searchsorted(tiempos, t_perdida, side="left")
            hay = pos < len(tiempos)
            primero = np.where(hay, tiempos[np.minimum(pos, len(tiempos) - 1)], np.nan)
            dentro = hay & (primero < t_perdida + ventana_segundos)
        else:
            primero = np.full(len(t_perdida), np.nan)
            dentro = np.zeros(len(t_perdida), dtype=bool)
        for tp, pr, ok in zip(t_perdida, primero, dentro):
            resultados.append(
                {
                    "match_id": match_id,
                    "period": period,
                    "t_perdida": tp,
                    "tiempo_reaccion": (pr - tp) if ok else np.nan,
                    "contrapresiono": bool(ok),
                }
            )

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# 7. EVENTOS DISPARADORES - situaciones inesperadas (genérico, por equipo)
# ---------------------------------------------------------------------------
def identificar_goles(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica todos los goles del dataset (de cualquier equipo), a partir
    de eventos tipo "Shot" con shot_outcome == "Goal".

    Devuelve una fila por gol con: match_id, period, minute, second,
    t_segundos, team (equipo que marca).

    Nota: los autogoles en StatsBomb se registran como evento "Own Goal For"
    del equipo que SE BENEFICIA (no como Shot), así que no quedan incluidos
    aquí. Si el dataset analizado tiene autogoles relevantes habría que
    sumarlos aparte filtrando type == "Own Goal For".
    """
    if "shot_outcome" not in events_df.columns:
        raise ValueError(
            "events_df necesita la columna 'shot_outcome' (viene de statsbombpy "
            "al pedir eventos con columnas de tiro incluidas)."
        )
    goles = events_df[
        (events_df.type == "Shot") & (events_df.shot_outcome == "Goal")
    ].copy()
    goles["t_segundos"] = goles["minute"] * 60 + goles["second"]
    return (
        goles[["match_id", "period", "minute", "second", "t_segundos", "team"]]
        .sort_values(["match_id", "t_segundos"])
        .reset_index(drop=True)
    )


def identificar_expulsiones(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica expulsiones (tarjeta roja directa o segunda amarilla) de
    cualquier equipo en el dataset.

    StatsBomb registra las tarjetas de dos formas distintas:
      - Ligadas a una falta: evento "Foul Committed" con
        foul_committed_card in {"Red Card", "Second Yellow"}.
      - Sin falta asociada (ej. protesta, conducta violenta): evento
        "Bad Behaviour" con bad_behaviour_card in {"Red Card", "Second Yellow"}.
    Esta función combina ambas fuentes.

    Devuelve una fila por expulsión con: match_id, period, minute, second,
    t_segundos, team (equipo del jugador expulsado), tipo_tarjeta.
    """
    partes = []
    if "foul_committed_card" in events_df.columns:
        f = events_df[
            (events_df.type == "Foul Committed")
            & (events_df.foul_committed_card.isin(["Red Card", "Second Yellow"]))
        ].copy()
        f["tipo_tarjeta"] = f["foul_committed_card"]
        partes.append(f)
    if "bad_behaviour_card" in events_df.columns:
        b = events_df[
            (events_df.type == "Bad Behaviour")
            & (events_df.bad_behaviour_card.isin(["Red Card", "Second Yellow"]))
        ].copy()
        b["tipo_tarjeta"] = b["bad_behaviour_card"]
        partes.append(b)

    columnas = ["match_id", "period", "minute", "second", "t_segundos", "team", "tipo_tarjeta"]
    if not partes:
        return pd.DataFrame(columns=columnas)

    expulsiones = pd.concat(partes, ignore_index=True)
    expulsiones["t_segundos"] = expulsiones["minute"] * 60 + expulsiones["second"]
    return (
        expulsiones[columnas]
        .sort_values(["match_id", "t_segundos"])
        .reset_index(drop=True)
    )


def marcador_al_descanso(events_df: pd.DataFrame, equipo: str) -> pd.DataFrame:
    """
    Para cada partido donde jugó `equipo`, calcula el marcador acumulado al
    final del primer tiempo (period == 1) y si el equipo se fue perdiendo,
    empatando o ganando al descanso. Se apoya en identificar_goles.

    Si no hay goles en el periodo 1 para ninguno de los dos equipos, el
    marcador es 0-0 (empate).
    """
    goles = identificar_goles(events_df)
    goles_1t = goles[goles.period == 1]

    resultados = []
    partidos = events_df.loc[events_df.team == equipo, "match_id"].unique()

    for match_id in partidos:
        goles_partido = goles_1t[goles_1t.match_id == match_id]
        equipos_partido = events_df.loc[events_df.match_id == match_id, "team"].unique()
        rival = next((e for e in equipos_partido if e != equipo), None)

        goles_favor = (goles_partido.team == equipo).sum()
        goles_contra = (goles_partido.team == rival).sum() if rival else 0

        if goles_favor > goles_contra:
            situacion = "ganando"
        elif goles_favor < goles_contra:
            situacion = "perdiendo"
        else:
            situacion = "empatando"

        resultados.append(
            {
                "match_id": match_id,
                "rival": rival,
                "goles_favor_1t": int(goles_favor),
                "goles_contra_1t": int(goles_contra),
                "situacion_al_descanso": situacion,
            }
        )

    return pd.DataFrame(resultados)


def identificar_eventos_disparadores(
    events_df: pd.DataFrame,
    equipo: str,
    minuto_limite_gol_temprano: float = 20,
) -> pd.DataFrame:
    """
    Responde la primera mitad de la pregunta 3 del Plan Maestro: define con
    criterio explícito qué cuenta como "situación inesperada" para `equipo`,
    a partir de 4 tipos de evento disparador:

      - "gol_recibido": el rival le marca un gol a `equipo` (cualquier minuto).
      - "gol_recibido_temprano": igual que arriba, pero antes del minuto
        `minuto_limite_gol_temprano` (20 por defecto). Se reporta aparte
        porque un gol temprano obliga a reaccionar con casi todo el partido
        por delante, a diferencia de uno sobre el final.
      - "expulsion_propia": a `equipo` le expulsan un jugador (queda en
        inferioridad numérica).
      - "expulsion_rival": al rival le expulsan un jugador (`equipo` queda
        con superioridad numérica).
      - "perdiendo_al_descanso": `equipo` se fue abajo en el marcador al
        final del primer tiempo. El evento se ancla al primer `t_segundos`
        real del segundo tiempo de ese partido (no a 0: StatsBomb NO
        reinicia el minuto al empezar el segundo tiempo, sigue acumulando
        desde donde terminó el primero, típicamente ~45), que es el primer
        momento en que el equipo puede reaccionar a esa situación.

    Devuelve una fila por evento disparador con: match_id, period,
    t_segundos, tipo_disparador, detalle (texto libre con el marcador o
    equipo involucrado).

    Supuesto explícito: un mismo partido puede aportar varios eventos
    disparadores (ej. un gol recibido Y una expulsión); cada uno se evalúa
    de forma independiente en comparar_metricas_pre_post_evento.
    """
    disparadores = []
    partidos_equipo = events_df.loc[events_df.team == equipo, "match_id"].unique()

    goles = identificar_goles(events_df)
    goles_recibidos = goles[
        goles.match_id.isin(partidos_equipo) & (goles.team != equipo)
    ]
    for _, row in goles_recibidos.iterrows():
        disparadores.append(
            {
                "match_id": row.match_id,
                "period": row.period,
                "t_segundos": row.t_segundos,
                "tipo_disparador": "gol_recibido",
                "detalle": f"gol de {row.team} al min {row.minute}",
            }
        )
        if row.minute < minuto_limite_gol_temprano:
            disparadores.append(
                {
                    "match_id": row.match_id,
                    "period": row.period,
                    "t_segundos": row.t_segundos,
                    "tipo_disparador": "gol_recibido_temprano",
                    "detalle": f"gol de {row.team} al min {row.minute} (< {minuto_limite_gol_temprano})",
                }
            )

    expulsiones = identificar_expulsiones(events_df)
    expulsiones_relevantes = expulsiones[expulsiones.match_id.isin(partidos_equipo)]
    for _, row in expulsiones_relevantes.iterrows():
        tipo = "expulsion_propia" if row.team == equipo else "expulsion_rival"
        disparadores.append(
            {
                "match_id": row.match_id,
                "period": row.period,
                "t_segundos": row.t_segundos,
                "tipo_disparador": tipo,
                "detalle": f"{row.tipo_tarjeta} a {row.team} al min {row.minute}",
            }
        )

    df_t = events_df.copy()
    df_t["t_segundos"] = df_t["minute"] * 60 + df_t["second"]
    inicio_segundo_tiempo = (
        df_t[df_t.period == 2].groupby("match_id")["t_segundos"].min()
    )

    marcador = marcador_al_descanso(events_df, equipo)
    partidos_perdiendo = marcador[marcador.situacion_al_descanso == "perdiendo"]
    for _, row in partidos_perdiendo.iterrows():
        if row.match_id not in inicio_segundo_tiempo.index:
            continue  # partido sin segundo tiempo registrado (dataset incompleto)
        disparadores.append(
            {
                "match_id": row.match_id,
                "period": 2,
                "t_segundos": inicio_segundo_tiempo[row.match_id],
                "tipo_disparador": "perdiendo_al_descanso",
                "detalle": f"{row.goles_favor_1t}-{row.goles_contra_1t} vs {row.rival} al descanso",
            }
        )

    if not disparadores:
        return pd.DataFrame(
            columns=["match_id", "period", "t_segundos", "tipo_disparador", "detalle"]
        )

    return (
        pd.DataFrame(disparadores)
        .sort_values(["match_id", "t_segundos"])
        .reset_index(drop=True)
    )


def metricas_ventana(
    events_df: pd.DataFrame,
    equipo: str,
    match_id,
    period: int,
    t_inicio: float,
    ventana_segundos: float,
) -> dict:
    """
    Calcula un set de métricas resumen de `equipo` dentro de una ventana de
    tiempo [t_inicio, t_inicio + ventana_segundos) de un partido/periodo
    específico. Pensada para medir el comportamiento del equipo justo
    después de un evento disparador (ver identificar_eventos_disparadores).

    Todas las métricas se expresan como tasas (%, por minuto) en vez de
    conteos crudos, precisamente para que sean comparables contra
    linea_base_equipo (que se calcula sobre todo el dataset, con mucha más
    duración) en comparar_metricas_pre_post_evento.

    Requiere coordenadas x, y normalizadas (normalizar_direccion_ataque) si
    se quiere una altura_promedio_x comparable entre partidos/periodos.

    Supuesto explícito: la posesión dentro de la ventana se aproxima a nivel
    de EVENTO (% de eventos con possession_team == equipo), no a nivel de
    posesión completa (build_possessions), porque una posesión puede
    empezar antes o terminar después del borde de la ventana y eso rompería
    la comparación entre ventanas cortas.
    """
    claves_vacias = {
        "n_eventos_equipo": 0, "posesion_pct": None, "pct_pases_largos": None,
        "presiones_por_min": None, "altura_promedio_x": None,
        "tiros_favor": 0, "tiros_contra": 0,
        "tiros_favor_por_min": None, "tiros_contra_por_min": None,
    }

    df = events_df[(events_df.match_id == match_id) & (events_df.period == period)].copy()
    if df.empty:
        return claves_vacias

    df["t_segundos"] = df["minute"] * 60 + df["second"]
    ventana = df[(df.t_segundos >= t_inicio) & (df.t_segundos < t_inicio + ventana_segundos)]
    if ventana.empty:
        return claves_vacias

    eventos_equipo = ventana[ventana.team == equipo]
    rival_nombre = next((e for e in ventana.team.unique() if e != equipo), None)
    eventos_rival = ventana[ventana.team == rival_nombre] if rival_nombre else ventana.iloc[0:0]

    posesion_pct = (ventana.possession_team == equipo).mean() * 100

    pases_equipo = eventos_equipo[eventos_equipo.type == "Pass"]
    if "pass_length" in pases_equipo.columns and len(pases_equipo) > 0:
        pct_pases_largos = (pases_equipo["pass_length"] * 0.9144 >= 18).mean() * 100
    else:
        pct_pases_largos = np.nan

    minutos_ventana = ventana_segundos / 60
    presiones_por_min = (
        (eventos_equipo.type == "Pressure").sum() / minutos_ventana if minutos_ventana > 0 else np.nan
    )

    altura_promedio_x = eventos_equipo["x"].mean() if "x" in eventos_equipo.columns else np.nan

    tiros_favor = int((eventos_equipo.type == "Shot").sum())
    tiros_contra = int((eventos_rival.type == "Shot").sum())
    tiros_favor_por_min = tiros_favor / minutos_ventana if minutos_ventana > 0 else np.nan
    tiros_contra_por_min = tiros_contra / minutos_ventana if minutos_ventana > 0 else np.nan

    return {
        "n_eventos_equipo": len(eventos_equipo),
        "posesion_pct": round(posesion_pct, 1),
        "pct_pases_largos": round(pct_pases_largos, 1) if not np.isnan(pct_pases_largos) else None,
        "presiones_por_min": round(presiones_por_min, 2) if not np.isnan(presiones_por_min) else None,
        "altura_promedio_x": round(altura_promedio_x, 1) if not np.isnan(altura_promedio_x) else None,
        "tiros_favor": tiros_favor,
        "tiros_contra": tiros_contra,
        "tiros_favor_por_min": round(tiros_favor_por_min, 3) if not np.isnan(tiros_favor_por_min) else None,
        "tiros_contra_por_min": round(tiros_contra_por_min, 3) if not np.isnan(tiros_contra_por_min) else None,
    }


def linea_base_equipo(events_df: pd.DataFrame, equipo: str) -> dict:
    """
    Calcula el "promedio normal" de `equipo` a lo largo de TODO el dataset
    (todos los partidos/minutos disponibles), con las mismas métricas de
    tasa que metricas_ventana (%, por minuto), para que una ventana corta
    post-evento disparador sea comparable contra esta línea base aunque su
    duración sea muy distinta.

    Requiere coordenadas x, y normalizadas (normalizar_direccion_ataque).
    """
    df = events_df.copy()
    eventos_equipo = df[df.team == equipo]
    eventos_rival = df[df.team != equipo]

    posesion_pct = (df.possession_team == equipo).mean() * 100

    pases_equipo = eventos_equipo[eventos_equipo.type == "Pass"]
    if "pass_length" in pases_equipo.columns and len(pases_equipo) > 0:
        pct_pases_largos = (pases_equipo["pass_length"] * 0.9144 >= 18).mean() * 100
    else:
        pct_pases_largos = np.nan

    minutos_totales = df.groupby(["match_id", "period"])["minute"].max().sum()
    presiones_por_min = (
        (eventos_equipo.type == "Pressure").sum() / minutos_totales if minutos_totales > 0 else np.nan
    )

    altura_promedio_x = eventos_equipo["x"].mean() if "x" in eventos_equipo.columns else np.nan

    tiros_favor_por_min = (eventos_equipo.type == "Shot").sum() / minutos_totales if minutos_totales > 0 else np.nan
    tiros_contra_por_min = (eventos_rival.type == "Shot").sum() / minutos_totales if minutos_totales > 0 else np.nan

    return {
        "posesion_pct": round(posesion_pct, 1),
        "pct_pases_largos": round(pct_pases_largos, 1) if not np.isnan(pct_pases_largos) else None,
        "presiones_por_min": round(presiones_por_min, 2) if not np.isnan(presiones_por_min) else None,
        "altura_promedio_x": round(altura_promedio_x, 1) if not np.isnan(altura_promedio_x) else None,
        "tiros_favor_por_min": round(tiros_favor_por_min, 3) if not np.isnan(tiros_favor_por_min) else None,
        "tiros_contra_por_min": round(tiros_contra_por_min, 3) if not np.isnan(tiros_contra_por_min) else None,
    }


def comparar_metricas_pre_post_evento(
    events_df: pd.DataFrame,
    equipo: str,
    eventos_disparadores: pd.DataFrame = None,
    ventana_segundos: float = 300,
) -> pd.DataFrame:
    """
    Responde la segunda mitad de la pregunta 3 del Plan Maestro: para cada
    evento disparador de `equipo` (ver identificar_eventos_disparadores),
    calcula las métricas de metricas_ventana en los `ventana_segundos`
    posteriores al evento (300s = 5 min por defecto) y las compara contra
    la línea base del equipo (linea_base_equipo) en todo el dataset.

    Si no se pasa `eventos_disparadores`, se calcula internamente llamando
    a identificar_eventos_disparadores(events_df, equipo) con sus valores
    por defecto.

    Requiere coordenadas x, y normalizadas (normalizar_direccion_ataque).

    Devuelve una fila por evento disparador con las métricas post-evento,
    columnas `base_*` con la línea base repetida (para lectura directa) y
    columnas `diff_*` con la diferencia (post - base): positivo significa
    que el equipo sube esa métrica tras el evento; negativo, que se repliega.
    """
    if eventos_disparadores is None:
        eventos_disparadores = identificar_eventos_disparadores(events_df, equipo)

    if eventos_disparadores.empty:
        return pd.DataFrame()

    base = linea_base_equipo(events_df, equipo)
    claves_comparables = [
        "posesion_pct", "pct_pases_largos", "presiones_por_min",
        "altura_promedio_x", "tiros_favor_por_min", "tiros_contra_por_min",
    ]
    filas = []

    for _, row in eventos_disparadores.iterrows():
        post = metricas_ventana(
            events_df, equipo, row.match_id, row.period, row.t_segundos, ventana_segundos
        )
        fila = {
            "match_id": row.match_id,
            "tipo_disparador": row.tipo_disparador,
            "detalle": row.detalle,
            **post,
        }
        for clave in claves_comparables:
            fila[f"base_{clave}"] = base.get(clave)
            if post.get(clave) is not None and base.get(clave) is not None:
                fila[f"diff_{clave}"] = round(post[clave] - base[clave], 3)
            else:
                fila[f"diff_{clave}"] = None
        filas.append(fila)

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 8. CONSISTENCIA Y VARIABILIDAD (genérico, por equipo)
# ---------------------------------------------------------------------------
def obtener_partidos_entrenador_equipo(
    matches_df: pd.DataFrame, entrenador: str, equipo: str
) -> pd.DataFrame:
    """
    Filtra una tabla de partidos de statsbombpy (sb.matches, concatenada
    para todas las temporadas de interés) a los partidos donde `entrenador`
    dirigió a `equipo`, ya sea de local o de visitante.

    `matches_df` debe incluir, como mínimo: match_id, home_team,
    home_managers, away_team, away_managers (home_managers/away_managers
    son strings con nombres separados por coma, tal como los entrega
    statsbombpy).

    Genérica: funciona igual para cualquier par entrenador/equipo, no solo
    Jardine/América (ver "Principio de reusabilidad" en el Plan Maestro).
    """
    registros = []
    for _, row in matches_df.iterrows():
        home_mgrs = [m.strip() for m in str(row.home_managers).split(",") if m.strip()]
        away_mgrs = [m.strip() for m in str(row.away_managers).split(",") if m.strip()]
        if entrenador in home_mgrs and row.home_team == equipo:
            registros.append(row)
        elif entrenador in away_mgrs and row.away_team == equipo:
            registros.append(row)
    return pd.DataFrame(registros).reset_index(drop=True)


def metricas_por_partido(
    events_df: pd.DataFrame, equipo: str, umbral_ppda_x: float = 72
) -> pd.DataFrame:
    """
    Aplica perfil_ofensivo_defensivo e indice_directividad PARTIDO POR
    PARTIDO (no agregados sobre todo el dataset), para poder estudiar la
    distribución de cada métrica entre partidos en vez de solo su promedio
    general — necesario para responder la pregunta de consistencia/
    variabilidad del Plan Maestro.

    Requiere coordenadas x, y normalizadas (normalizar_direccion_ataque).

    Devuelve una fila por partido con las mismas columnas que
    perfil_ofensivo_defensivo (posesion_pct, field_tilt_pct,
    field_tilt_relativo_pct, ppda, tiros_favor/contra, xg_favor/contra si
    aplica) más pct_pases_largos y longitud_promedio_m de indice_directividad.
    """
    filas = []
    partidos = events_df.loc[events_df.team == equipo, "match_id"].unique()
    for match_id in partidos:
        sub = events_df[events_df.match_id == match_id]
        perfil = perfil_ofensivo_defensivo(sub, equipo=equipo, umbral_ppda_x=umbral_ppda_x)
        directividad = indice_directividad(sub, equipo=equipo)
        fila = {"match_id": match_id, **perfil}
        fila["pct_pases_largos"] = directividad["pct_pases_largos"]
        fila["longitud_promedio_m"] = directividad["longitud_promedio_m"]
        filas.append(fila)

    return pd.DataFrame(filas)


def agregar_contexto_partido(
    metricas_df: pd.DataFrame, partidos_df: pd.DataFrame, equipo: str
) -> pd.DataFrame:
    """
    Añade columnas de contexto (localidad: "Local"/"Visitante", rival,
    temporada, fecha) a un DataFrame de métricas por partido (ver
    metricas_por_partido), a partir de la tabla de metadatos de partidos de
    statsbombpy (sb.matches / obtener_partidos_entrenador_equipo).

    `partidos_df` debe incluir, como mínimo: match_id, home_team, away_team.
    Si además incluye "temporada" y/o "match_date", esas columnas se
    conservan en el resultado.

    Genérica: la localidad y el rival se derivan comparando contra `equipo`,
    no contra un nombre fijo (ver "Principio de reusabilidad").
    """
    columnas_contexto = ["match_id", "home_team", "away_team"]
    columnas_contexto += [c for c in ["temporada", "match_date"] if c in partidos_df.columns]

    contexto = partidos_df[columnas_contexto].copy()
    contexto["localidad"] = np.where(contexto.home_team == equipo, "Local", "Visitante")
    contexto["rival"] = np.where(
        contexto.home_team == equipo, contexto.away_team, contexto.home_team
    )

    return metricas_df.merge(contexto, on="match_id", how="left")


def resumen_variabilidad(metricas_df: pd.DataFrame, columnas: list = None) -> pd.DataFrame:
    """
    Para cada métrica numérica indicada en `columnas` (o todas las
    numéricas del DataFrame si no se especifica), calcula media, desviación
    estándar y coeficiente de variación (CV = std / media) a lo largo de
    los partidos.

    Un CV alto indica mucha variabilidad partido a partido en esa métrica
    (estilo inconsistente); un CV bajo indica que el equipo repite un
    patrón similar partido tras partido (estilo consistente).

    Ordena el resultado por CV descendente (las métricas menos consistentes
    primero).
    """
    if columnas is None:
        columnas = metricas_df.select_dtypes(include=[np.number]).columns.tolist()
        columnas = [c for c in columnas if c != "match_id"]

    resumen = []
    for col in columnas:
        if col not in metricas_df.columns:
            continue
        serie = metricas_df[col].dropna()
        if serie.empty:
            continue
        media = serie.mean()
        std = serie.std()
        cv = std / media if media != 0 else np.nan
        resumen.append(
            {
                "metrica": col,
                "media": round(media, 2),
                "std": round(std, 2),
                "cv": round(cv, 3) if not np.isnan(cv) else None,
                "n_partidos": len(serie),
            }
        )

    return pd.DataFrame(resumen).sort_values("cv", ascending=False, na_position="last").reset_index(drop=True)


def variabilidad_por_grupo(
    metricas_df: pd.DataFrame, columna_grupo: str, columnas: list = None
) -> pd.DataFrame:
    """
    Igual que resumen_variabilidad, pero calculado por separado para cada
    valor de `columna_grupo` (ej. "localidad" o "temporada"), para ver si
    la consistencia del equipo cambia según el contexto del partido (ej.
    ¿es más variable de visitante que de local?).
    """
    if columnas is None:
        columnas = metricas_df.select_dtypes(include=[np.number]).columns.tolist()
        columnas = [c for c in columnas if c != "match_id"]

    filas = []
    for grupo, sub in metricas_df.groupby(columna_grupo):
        for col in columnas:
            if col not in sub.columns:
                continue
            serie = sub[col].dropna()
            if serie.empty:
                continue
            media = serie.mean()
            std = serie.std()
            cv = std / media if media != 0 else np.nan
            filas.append(
                {
                    columna_grupo: grupo,
                    "metrica": col,
                    "media": round(media, 2),
                    "std": round(std, 2),
                    "cv": round(cv, 3) if not np.isnan(cv) else None,
                    "n_partidos": len(serie),
                }
            )

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 9. USO DE JUGADORES (genérico, por equipo)
# ---------------------------------------------------------------------------
def _parse_tactics_lineup(valor):
    """
    Convierte el campo `tactics` de un evento "Starting XI" (que statsbombpy
    entrega como dict de Python, o como texto tipo "{'formation': 343,
    'lineup': [...]}" si el events_df se guardó y volvió a cargar desde
    CSV) a una lista simple de dicts {jugador, posicion, jersey_number}.

    Devuelve (formacion, lista_jugadores). Si no se puede parsear, devuelve
    (None, []).
    """
    if isinstance(valor, dict):
        tactics = valor
    elif isinstance(valor, str):
        try:
            tactics = ast.literal_eval(valor)
        except (ValueError, SyntaxError):
            return None, []
    else:
        return None, []

    if not isinstance(tactics, dict):
        return None, []

    formacion = tactics.get("formation")
    lineup = []
    for item in tactics.get("lineup", []):
        jugador = item.get("player", {}).get("name")
        posicion = item.get("position", {}).get("name")
        jersey = item.get("jersey_number")
        if jugador:
            lineup.append({"jugador": jugador, "posicion": posicion, "jersey_number": jersey})

    return formacion, lineup


def extraer_alineaciones_titulares(events_df: pd.DataFrame, equipo: str) -> pd.DataFrame:
    """
    Extrae la alineación titular de `equipo` en cada partido, a partir del
    evento "Starting XI" (campo `tactics`, que statsbombpy entrega anidado).

    Devuelve una fila por partido con: match_id, formacion, jugadores (tupla
    ordenada de nombres, para poder comparar fácilmente entre partidos) y
    detalle (lista de dicts jugador/posición/dorsal, para referencia).
    """
    filas = []
    titulares = events_df[(events_df.team == equipo) & (events_df.type == "Starting XI")]

    for _, row in titulares.iterrows():
        formacion, lineup = _parse_tactics_lineup(row.get("tactics"))
        if not lineup:
            continue
        jugadores = tuple(sorted(j["jugador"] for j in lineup))
        filas.append(
            {
                "match_id": row.match_id,
                "formacion": formacion,
                "jugadores": jugadores,
                "detalle": lineup,
            }
        )

    return pd.DataFrame(filas)


def resumen_titulares(alineaciones_df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de la salida de extraer_alineaciones_titulares, cuenta cuántas
    veces fue titular cada jugador y en qué posiciones jugó, para ver quién
    es indiscutible en el once de Jardine y quién rota más.
    """
    registros = []
    for _, row in alineaciones_df.iterrows():
        for item in row.detalle:
            registros.append({"jugador": item["jugador"], "posicion": item["posicion"], "match_id": row.match_id})

    if not registros:
        return pd.DataFrame(columns=["jugador", "n_partidos_titular", "posicion_principal"])

    df = pd.DataFrame(registros)
    resumen = (
        df.groupby("jugador")
        .agg(
            n_partidos_titular=("match_id", "nunique"),
            posicion_principal=("posicion", lambda s: s.value_counts().idxmax()),
        )
        .reset_index()
        .sort_values("n_partidos_titular", ascending=False)
        .reset_index(drop=True)
    )
    return resumen


def cambios_alineacion_consecutivos(alineaciones_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mide la continuidad de la alineación titular entre partidos
    CONSECUTIVOS. `alineaciones_df` debe venir YA ORDENADO cronológicamente
    (ej. por match_date, uniendo con los metadatos de partido del bloque 05)
    antes de llamar a esta función — el orden de las filas es lo que define
    qué cuenta como "el partido siguiente".

    Para cada par de partidos consecutivos calcula cuántos de los 11
    titulares se repiten y cuántos cambian. Un número bajo de cambios
    sostenido en el tiempo indica un once "fijo"; muchos cambios indican
    rotación constante.
    """
    filas = []
    jugadores_anterior = None
    match_anterior = None

    for _, row in alineaciones_df.iterrows():
        if jugadores_anterior is not None:
            repetidos = len(set(row.jugadores) & set(jugadores_anterior))
            filas.append(
                {
                    "match_id_anterior": match_anterior,
                    "match_id": row.match_id,
                    "jugadores_repetidos": repetidos,
                    "cambios": len(row.jugadores) - repetidos,
                }
            )
        jugadores_anterior = row.jugadores
        match_anterior = row.match_id

    return pd.DataFrame(filas)


def identificar_sustituciones(events_df: pd.DataFrame, equipo: str) -> pd.DataFrame:
    """
    Identifica las sustituciones de `equipo` en el dataset, a partir de
    eventos tipo "Substitution".

    Devuelve una fila por sustitución con: match_id, period, minute, second,
    t_segundos, jugador_sale, jugador_entra.
    """
    sustituciones = events_df[(events_df.team == equipo) & (events_df.type == "Substitution")].copy()
    sustituciones["t_segundos"] = sustituciones["minute"] * 60 + sustituciones["second"]
    sustituciones = sustituciones.rename(
        columns={"player": "jugador_sale", "substitution_replacement": "jugador_entra"}
    )
    columnas = ["match_id", "period", "minute", "second", "t_segundos", "jugador_sale", "jugador_entra"]
    return sustituciones[columnas].sort_values(["match_id", "t_segundos"]).reset_index(drop=True)


def comparar_metricas_pre_post_sustitucion(
    events_df: pd.DataFrame,
    equipo: str,
    sustituciones: pd.DataFrame = None,
    ventana_segundos: float = 300,
) -> pd.DataFrame:
    """
    Para cada sustitución de `equipo` (ver identificar_sustituciones),
    compara las métricas de metricas_ventana en los `ventana_segundos`
    ANTES del cambio contra los `ventana_segundos` DESPUÉS, para medir el
    efecto inmediato de cada sustitución (distinto de
    comparar_metricas_pre_post_evento, que compara contra la línea base de
    todo el dataset en vez de contra el propio "antes" de ese momento).

    Si no se pasa `sustituciones`, se calcula internamente llamando a
    identificar_sustituciones(events_df, equipo).

    Requiere coordenadas x, y normalizadas (normalizar_direccion_ataque).

    Devuelve una fila por sustitución con las métricas de la ventana ANTES
    (prefijo `antes_`), DESPUÉS (prefijo `despues_`) y la diferencia
    (`diff_`, despues - antes).
    """
    if sustituciones is None:
        sustituciones = identificar_sustituciones(events_df, equipo)

    if sustituciones.empty:
        return pd.DataFrame()

    claves_comparables = [
        "posesion_pct", "pct_pases_largos", "presiones_por_min",
        "altura_promedio_x", "tiros_favor_por_min", "tiros_contra_por_min",
    ]
    filas = []

    for _, row in sustituciones.iterrows():
        antes = metricas_ventana(
            events_df, equipo, row.match_id, row.period,
            max(row.t_segundos - ventana_segundos, 0), ventana_segundos,
        )
        despues = metricas_ventana(
            events_df, equipo, row.match_id, row.period, row.t_segundos, ventana_segundos
        )
        fila = {
            "match_id": row.match_id,
            "jugador_sale": row.jugador_sale,
            "jugador_entra": row.jugador_entra,
            "minute": row.minute,
        }
        for clave in claves_comparables:
            fila[f"antes_{clave}"] = antes.get(clave)
            fila[f"despues_{clave}"] = despues.get(clave)
            if antes.get(clave) is not None and despues.get(clave) is not None:
                fila[f"diff_{clave}"] = round(despues[clave] - antes[clave], 3)
            else:
                fila[f"diff_{clave}"] = None
        filas.append(fila)

    return pd.DataFrame(filas)


def ubicaciones_jugador(
    events_df: pd.DataFrame,
    equipo: str,
    jugador: str = None,
    tipos_evento: list = None,
) -> pd.DataFrame:
    """
    Devuelve las coordenadas x, y de los eventos de `equipo` (opcionalmente
    de un solo `jugador`, y opcionalmente filtradas a `tipos_evento`, ej.
    ["Pass", "Carry", "Pressure"]), para graficar un mapa de calor
    individual o por posición.

    Requiere coordenadas x, y normalizadas (normalizar_direccion_ataque).
    """
    df = events_df[events_df.team == equipo]
    if jugador is not None:
        df = df[df.player == jugador]
    if tipos_evento is not None:
        df = df[df.type.isin(tipos_evento)]

    columnas = ["match_id", "player", "position", "type", "x", "y"]
    columnas = [c for c in columnas if c in df.columns]
    return df[columnas].dropna(subset=["x", "y"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 10. BALÓN PARADO (genérico, por equipo)
# ---------------------------------------------------------------------------
_MAPA_BALON_PARADO = {
    "corner": {"play_pattern": "From Corner", "pass_type": "Corner"},
    "tiro_libre": {"play_pattern": "From Free Kick", "pass_type": "Free Kick"},
    "saque_banda": {"play_pattern": "From Throw In", "pass_type": "Throw-in"},
}


def _validar_tipo_balon_parado(tipo: str) -> dict:
    if tipo not in _MAPA_BALON_PARADO:
        raise ValueError(
            f"tipo debe ser uno de {list(_MAPA_BALON_PARADO)}, recibido: {tipo!r}"
        )
    return _MAPA_BALON_PARADO[tipo]


def jugadas_balon_parado(events_df: pd.DataFrame, equipo: str, tipo: str) -> pd.DataFrame:
    """
    Devuelve TODOS los eventos de una jugada originada en un balón parado de
    `equipo` (tipo: "corner", "tiro_libre" o "saque_banda"), usando el campo
    `play_pattern` que StatsBomb asigna a cada evento de la posesión que
    sigue al saque, hasta que cambia la posesión.

    Útil como base para remates_balon_parado y perfil_envios_balon_parado,
    o para inspeccionar la jugada completa.
    """
    mapa = _validar_tipo_balon_parado(tipo)
    return events_df[
        (events_df.possession_team == equipo) & (events_df.play_pattern == mapa["play_pattern"])
    ].reset_index(drop=True)


def resumen_balon_parado(events_df: pd.DataFrame, equipo: str, tipo: str) -> dict:
    """
    Resume la efectividad ofensiva de `equipo` en un tipo de balón parado:
    cuántos se sacaron, cuántos terminan en remate propio, y el xG total y
    promedio generado.

    `n_balones_parados` se cuenta por el pase de saque (type="Pass",
    pass_type correspondiente) — ej. número de córners realmente botados,
    no el total de eventos de la jugada.
    """
    mapa = _validar_tipo_balon_parado(tipo)
    saques = events_df[
        (events_df.team == equipo) & (events_df.type == "Pass") & (events_df.pass_type == mapa["pass_type"])
    ]
    remates = remates_balon_parado(events_df, equipo, tipo, propio=True)

    n_balones_parados = len(saques)
    n_remates = len(remates)
    resultado = {
        "tipo": tipo,
        "n_balones_parados": n_balones_parados,
        "n_remates": n_remates,
        "pct_remate": round(n_remates / n_balones_parados * 100, 1) if n_balones_parados > 0 else None,
    }
    if "shot_statsbomb_xg" in remates.columns and not remates.empty:
        resultado["xg_total"] = round(remates["shot_statsbomb_xg"].sum(), 2)
        resultado["xg_promedio_por_remate"] = round(remates["shot_statsbomb_xg"].mean(), 3)
    return resultado


def remates_balon_parado(
    events_df: pd.DataFrame, equipo: str, tipo: str, propio: bool = True
) -> pd.DataFrame:
    """
    Remates originados en un tipo de balón parado, para estudiar la zona de
    remate.

    `propio=True` (por defecto): remates DE `equipo` tras su propio balón
    parado (vista ofensiva — dónde remata el equipo).
    `propio=False`: remates DEL RIVAL tras SU balón parado, dentro de los
    partidos de `equipo` (vista defensiva — dónde es vulnerable el equipo
    ante el balón parado rival).

    Requiere coordenadas x, y normalizadas (normalizar_direccion_ataque)
    para que la zona de remate sea comparable entre partidos.
    """
    mapa = _validar_tipo_balon_parado(tipo)
    condicion_equipo = (events_df.team == equipo) if propio else (events_df.team != equipo)
    remates = events_df[
        condicion_equipo & (events_df.type == "Shot") & (events_df.play_pattern == mapa["play_pattern"])
    ].copy()

    columnas = ["match_id", "team", "player", "x", "y", "shot_outcome", "shot_statsbomb_xg", "minute"]
    columnas = [c for c in columnas if c in remates.columns]
    return remates[columnas].reset_index(drop=True)


def perfil_envios_balon_parado(events_df: pd.DataFrame, equipo: str, tipo: str) -> pd.DataFrame:
    """
    Caracteriza los saques de `equipo` en un tipo de balón parado (pensado
    sobre todo para "corner" y "tiro_libre", que tienen pass_technique):
    técnica del envío (Inswinging/Outswinging/Straight) y punto de llegada
    (end_location), para ver a qué zona apunta el equipo y con qué tipo de
    centro.

    Requiere coordenadas normalizadas (normalizar_direccion_ataque) si se
    quiere que la zona de llegada sea comparable entre partidos.
    """
    mapa = _validar_tipo_balon_parado(tipo)
    saques = events_df[
        (events_df.team == equipo) & (events_df.type == "Pass") & (events_df.pass_type == mapa["pass_type"])
    ].copy()

    if "pass_end_location" in saques.columns:
        saques[["end_x", "end_y"]] = _parse_coord_column(saques["pass_end_location"])

    columnas = ["match_id", "player", "pass_technique", "x", "y", "end_x", "end_y", "pass_outcome"]
    columnas = [c for c in columnas if c in saques.columns]
    return saques[columnas].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 11. MODELO PREDICTIVO (criterio "capacidad predictiva", 30%)
# ---------------------------------------------------------------------------
def metricas_por_partido_primer_tiempo(
    events_df: pd.DataFrame, equipo: str, umbral_ppda_x: float = 72
) -> pd.DataFrame:
    """
    Igual que metricas_por_partido, pero calculada SOLO con eventos del
    primer tiempo (period == 1) de cada partido.

    Existe para evitar fuga de información (data leakage) en el modelo
    predictivo: si usáramos las métricas del partido COMPLETO para predecir
    el resultado de ese mismo partido, estaríamos usando información que
    ocurre después (o al mismo tiempo que) el resultado que queremos
    predecir — el modelo "haría trampa". Cortando en el descanso, el
    modelo predice un resultado que todavía no ha ocurrido al momento de
    calcular las features, lo cual sí es una predicción genuina y,
    operativamente, es exactamente el momento en que un cuerpo técnico
    tomaría decisiones de ajuste (ver bloque de valor operativo).

    Genérica: recibe `equipo` como parámetro (Principio de reusabilidad).
    """
    filas = []
    partidos = events_df.loc[events_df.team == equipo, "match_id"].unique()
    for match_id in partidos:
        sub = events_df[(events_df.match_id == match_id) & (events_df.period == 1)]
        if sub.empty:
            continue
        perfil = perfil_ofensivo_defensivo(sub, equipo=equipo, umbral_ppda_x=umbral_ppda_x)
        directividad = indice_directividad(sub, equipo=equipo)
        fila = {"match_id": match_id, **perfil}
        fila["pct_pases_largos"] = directividad["pct_pases_largos"]
        fila["longitud_promedio_m"] = directividad["longitud_promedio_m"]
        filas.append(fila)

    df = pd.DataFrame(filas)
    # Prefijo "_1t" a las columnas de métricas para dejar explícito en el
    # dataset final que son features de "solo primer tiempo" y no se
    # confundan con una métrica del partido completo.
    columnas_metricas = [c for c in df.columns if c not in ("match_id", "equipo")]
    return df.rename(columns={c: f"{c}_1t" for c in columnas_metricas})


def dataset_prediccion_resultado(
    events_df: pd.DataFrame, equipo: str, partidos_df: pd.DataFrame, umbral_ppda_x: float = 72
) -> pd.DataFrame:
    """
    Arma el dataset partido-por-partido para el modelo predictivo: una fila
    por partido con (a) features calculadas ÚNICAMENTE con información
    disponible al descanso (primer tiempo), y (b) el resultado FINAL del
    partido como variable objetivo.

    Features (todas "_1t", es decir, cortadas al medio tiempo):
      - Las de metricas_por_partido_primer_tiempo (posesión, field tilt,
        PPDA, tiros/xG a favor y en contra, % pases largos, longitud
        promedio de pase).
      - goles_favor_1t, goles_contra_1t, situacion_al_descanso (de
        marcador_al_descanso).
      - localidad (Local/Visitante) y temporada (de agregar_contexto_partido).

    Variables objetivo (NO usar como feature, son lo que se predice):
      - resultado_final: "Gana" / "Empata" / "Pierde", desde la perspectiva
        de `equipo`, usando home_score/away_score de partidos_df.
      - dif_gol_final: goles a favor menos goles en contra al final del
        partido (equipo - rival), variable continua para el componente de
        regresión (R²) del criterio de capacidad predictiva.

    Requiere coordenadas normalizadas (normalizar_direccion_ataque) en
    events_df. Genérica: recibe `equipo` como parámetro.
    """
    metricas_1t = metricas_por_partido_primer_tiempo(events_df, equipo=equipo, umbral_ppda_x=umbral_ppda_x)
    marcador_1t = marcador_al_descanso(events_df, equipo=equipo)
    contexto = agregar_contexto_partido(metricas_1t[["match_id"]], partidos_df, equipo=equipo)

    df = metricas_1t.merge(marcador_1t.drop(columns=["rival"]), on="match_id", how="left")
    df = df.merge(contexto, on="match_id", how="left")

    resultados_finales = partidos_df[["match_id", "home_team", "away_team", "home_score", "away_score"]].copy()

    def _resultado_final(row):
        if row.home_team == equipo:
            gf, gc = row.home_score, row.away_score
        elif row.away_team == equipo:
            gf, gc = row.away_score, row.home_score
        else:
            return pd.Series({"dif_gol_final": np.nan, "resultado_final": None})
        dif = gf - gc
        if dif > 0:
            res = "Gana"
        elif dif < 0:
            res = "Pierde"
        else:
            res = "Empata"
        return pd.Series({"dif_gol_final": dif, "resultado_final": res})

    resultados_finales[["dif_gol_final", "resultado_final"]] = resultados_finales.apply(_resultado_final, axis=1)
    df = df.merge(resultados_finales[["match_id", "dif_gol_final", "resultado_final"]], on="match_id", how="left")

    df = df.dropna(subset=["resultado_final"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 12. ANÁLISIS CAUSAL (criterio "identificación causal", 25%)
# ---------------------------------------------------------------------------
def dataset_causal_reaccion_gol_contra(
    events_df: pd.DataFrame,
    equipo: str,
    partidos_df: pd.DataFrame = None,
    minuto_limite: float = 75,
    ventana_reaccion_min: float = 10,
    ventana_resultado_min: float = 15,
) -> pd.DataFrame:
    """
    Pregunta causal: cuando `equipo` recibe un gol que lo deja perdiendo o
    empatando (antes del minuto `minuto_limite`), ¿reaccionar con un cambio
    (sustitución) dentro de los `ventana_reaccion_min` minutos siguientes
    aumenta la probabilidad de anotar un gol en los `ventana_resultado_min`
    minutos posteriores al gol recibido?

    "Tratamiento" (tratamiento=1): hubo al menos una sustitución de `equipo`
    en (t_evento, t_evento + ventana_reaccion_min].
    "Resultado" (resultado=1): `equipo` anotó al menos un gol en
    (t_evento, t_evento + ventana_resultado_min].

    Por qué esto NO es simplemente correlación: no comparamos "antes vs.
    después" del mismo evento (eso no prueba nada — el equipo podría mejorar
    solo por el paso del tiempo, cansancio del rival, etc.). Comparamos
    eventos donde SÍ hubo reacción contra eventos donde NO la hubo,
    EMPAREJADOS por confounders (ver matching_estratificado): minuto en que
    ocurrió el gol en contra (bucket de 15 min) y situación resultante
    (perdiendo vs. empatando) — porque un gol que deja perdiendo al minuto
    10 no es comparable a uno que deja empatando al minuto 70.

    Devuelve una fila por evento "gol en contra elegible", con columnas:
    match_id, t_segundos, minuto, situacion ("perdiendo"/"empatando"),
    minuto_bucket, tratamiento, resultado, y (si se pasa partidos_df)
    localidad y temporada.

    Limitación explícita (no resuelta): un mismo partido puede aportar más
    de un evento elegible si el equipo recibe varios goles en situaciones
    distintas — se tratan como observaciones independientes, lo cual
    subestima la correlación dentro de un mismo partido (no hay ajuste por
    cluster). Tampoco se controla la fuerza del rival ni si la sustitución
    fue "ofensiva" o no (StatsBomb no etiqueta la posición del jugador que
    entra de forma directamente utilizable aquí) — el tratamiento es
    "hizo algún cambio", no específicamente "cambio ofensivo".

    Genérica: recibe `equipo` como parámetro.
    """
    goles = identificar_goles(events_df)
    sustituciones = identificar_sustituciones(events_df, equipo)
    partidos_equipo = events_df.loc[events_df.team == equipo, "match_id"].unique()

    filas = []
    for match_id in partidos_equipo:
        goles_partido = goles[goles.match_id == match_id].sort_values("t_segundos")
        subs_partido = sustituciones[sustituciones.match_id == match_id]
        diff = 0
        for _, g in goles_partido.iterrows():
            if g.team == equipo:
                diff += 1
                continue

            diff -= 1  # gol del rival
            if diff > 0 or g.minute >= minuto_limite:
                continue  # equipo sigue ganando, o ya pasó el minuto límite: no elegible

            t_evento = g.t_segundos
            situacion = "perdiendo" if diff < 0 else "empatando"

            tratamiento = int(
                (
                    (subs_partido.t_segundos > t_evento)
                    & (subs_partido.t_segundos <= t_evento + ventana_reaccion_min * 60)
                ).any()
            )
            goles_favor_posteriores = goles_partido[
                (goles_partido.team == equipo)
                & (goles_partido.t_segundos > t_evento)
                & (goles_partido.t_segundos <= t_evento + ventana_resultado_min * 60)
            ]
            resultado = int(len(goles_favor_posteriores) > 0)

            filas.append(
                {
                    "match_id": match_id,
                    "t_segundos": t_evento,
                    "minuto": g.minute,
                    "situacion": situacion,
                    "minuto_bucket": int(g.minute // 15) * 15,
                    "tratamiento": tratamiento,
                    "resultado": resultado,
                }
            )

    df = pd.DataFrame(filas)
    if df.empty or partidos_df is None:
        return df

    contexto = partidos_df[["match_id", "home_team", "away_team"]].copy()
    contexto["localidad"] = np.where(contexto.home_team == equipo, "Local", "Visitante")
    if "temporada" in partidos_df.columns:
        contexto["temporada"] = partidos_df["temporada"]

    return df.merge(contexto[["match_id", "localidad"] + (["temporada"] if "temporada" in contexto.columns else [])], on="match_id", how="left")


def _efecto_estratificado(df, columna_tratamiento, columna_resultado, columnas_estrato):
    """Efecto ponderado por estrato (None si ningún estrato tiene tratados y controles)."""
    suma_ponderada, suma_pesos = 0.0, 0.0
    for _, grupo in df.groupby(columnas_estrato):
        tratados = grupo[grupo[columna_tratamiento] == 1]
        controles = grupo[grupo[columna_tratamiento] == 0]
        if len(tratados) == 0 or len(controles) == 0:
            continue
        peso = len(grupo)
        suma_ponderada += (tratados[columna_resultado].mean() - controles[columna_resultado].mean()) * peso
        suma_pesos += peso
    return suma_ponderada / suma_pesos if suma_pesos > 0 else None


def matching_estratificado(
    df: pd.DataFrame,
    columna_tratamiento: str,
    columna_resultado: str,
    columnas_estrato: list,
    n_bootstrap: int = 2000,
    semilla: int = 42,
) -> dict:
    """
    Estima el efecto de un tratamiento binario sobre un resultado binario
    mediante matching exacto/"coarsened" por estratos — un diseño
    quasi-experimental simple e interpretable (sin "caja negra"): en vez de
    comparar tratados vs. no tratados en bruto (lo cual confunde el efecto
    del tratamiento con cualquier otra diferencia sistemática entre ambos
    grupos), se agrupa por `columnas_estrato` (los confounders que podrían
    explicar tanto quién recibe el tratamiento como el resultado) y se
    compara tratados contra controles SOLO dentro de cada estrato, donde sí
    son comparables entre sí.

    Devuelve un diccionario con:
      - "tabla_estratos": DataFrame con, por cada combinación de
        columnas_estrato, n_tratados, n_controles, tasa_tratados,
        tasa_controles, diferencia.
      - "efecto_estimado": la diferencia de tasas (tratados - controles)
        agregada entre estratos, ponderada por el tamaño de cada estrato
        (promedio ponderado, no un promedio simple de las diferencias) —
        esto es el análogo simplificado de un estimador tipo
        Mantel-Haenszel.
      - "n_tratados_total", "n_controles_total".
      - "estratos_sin_variacion": estratos que se excluyen del cálculo del
        efecto por tener solo tratados o solo controles (no hay con qué
        comparar dentro de ese estrato) — se listan explícitamente, no se
        ocultan.

    Genérica: no asume nada sobre el dominio (fútbol, equipo, etc.) — sirve
    para cualquier pregunta causal con tratamiento/resultado binarios y
    columnas de estrato ya construidas.
    """
    filas_estrato = []
    estratos_sin_variacion = []
    suma_ponderada = 0.0
    suma_pesos = 0.0

    for valores_estrato, grupo in df.groupby(columnas_estrato):
        tratados = grupo[grupo[columna_tratamiento] == 1]
        controles = grupo[grupo[columna_tratamiento] == 0]
        n_t, n_c = len(tratados), len(controles)

        if isinstance(valores_estrato, tuple):
            etiqueta = dict(zip(columnas_estrato, valores_estrato))
        else:
            etiqueta = {columnas_estrato[0]: valores_estrato}

        if n_t == 0 or n_c == 0:
            estratos_sin_variacion.append({**etiqueta, "n_tratados": n_t, "n_controles": n_c})
            continue

        tasa_t = tratados[columna_resultado].mean()
        tasa_c = controles[columna_resultado].mean()
        diferencia = tasa_t - tasa_c
        peso = n_t + n_c

        filas_estrato.append(
            {**etiqueta, "n_tratados": n_t, "n_controles": n_c, "tasa_tratados": round(tasa_t, 3),
             "tasa_controles": round(tasa_c, 3), "diferencia": round(diferencia, 3)}
        )
        suma_ponderada += diferencia * peso
        suma_pesos += peso

    efecto_estimado = suma_ponderada / suma_pesos if suma_pesos > 0 else None

    # Diferencia "ingenua" (sin emparejar): compara tratados vs. controles en
    # bruto, mezclando el efecto del tratamiento con cualquier confounder.
    # Se reporta junto al efecto emparejado para mostrar explícitamente la
    # diferencia entre correlación y efecto ajustado por confounders.
    tasa_t_bruta = df.loc[df[columna_tratamiento] == 1, columna_resultado].mean()
    tasa_c_bruta = df.loc[df[columna_tratamiento] == 0, columna_resultado].mean()
    diferencia_ingenua = tasa_t_bruta - tasa_c_bruta

    # Intervalo de confianza al 95% por bootstrap (remuestreo con reemplazo
    # de las observaciones): con muestras chicas el estimador puntual por sí
    # solo engaña; el intervalo dice qué tan distinguible de cero es.
    ic = None
    if n_bootstrap and n_bootstrap > 0:
        rng = np.random.default_rng(semilla)
        efectos = []
        for _ in range(n_bootstrap):
            muestra = df.iloc[rng.integers(0, len(df), len(df))]
            e = _efecto_estratificado(muestra, columna_tratamiento, columna_resultado, columnas_estrato)
            if e is not None:
                efectos.append(e)
        if efectos:
            ic = (round(float(np.percentile(efectos, 2.5)), 3), round(float(np.percentile(efectos, 97.5)), 3))

    return {
        "tabla_estratos": pd.DataFrame(filas_estrato),
        "efecto_estimado": round(efecto_estimado, 3) if efecto_estimado is not None else None,
        "ic95_bootstrap": ic,
        "diferencia_ingenua": round(float(diferencia_ingenua), 3),
        "tasa_tratados_bruta": round(float(tasa_t_bruta), 3),
        "tasa_controles_bruta": round(float(tasa_c_bruta), 3),
        "n_tratados_total": int(df[columna_tratamiento].sum()),
        "n_controles_total": int((df[columna_tratamiento] == 0).sum()),
        "estratos_sin_variacion": estratos_sin_variacion,
    }


def dibujar_zona_porteria(
    ax, lado: str = "ataque", color: str = "white", linewidth: float = 1.5, alpha: float = 0.8
) -> None:
    """
    Dibuja sobre `ax` el área penal, el área chica (área de meta) y la
    portería, usando el sistema de coordenadas de StatsBomb (x: 0-120,
    y: 0-80).

    lado="ataque" dibuja la portería en x=120 (el extremo hacia el que
    ataca `equipo` tras normalizar_direccion_ataque) — úsalo para las
    vistas ofensivas (zona de remate propio, punto de llegada del envío).
    lado="defensa" dibuja la portería en x=0 — úsalo para la vista de
    vulnerabilidad defensiva, donde los remates del rival van hacia ese
    extremo tras la misma normalización.

    Pensada como referencia visual en los scatter plots de balón parado
    (córners, tiros libres, vulnerabilidad defensiva), para que se note
    con claridad dónde está el área chica y el área penal sin tener que
    inferirlo de los límites de los ejes. No recibe `equipo`: es una
    ayuda de dibujo pura (geometría de cancha), sin lógica de análisis
    que deba generalizarse por equipo.

    Uso: dibujar_zona_porteria(ax, lado="ataque") antes o después de
    ax.scatter(...).
    """
    import matplotlib.patches as patches

    if lado not in ("ataque", "defensa"):
        raise ValueError(f"lado debe ser 'ataque' o 'defensa', recibido: {lado!r}")

    if lado == "ataque":
        x_area_penal, x_linea_meta, x_area_chica = 102, 120, 114
        theta1, theta2 = 127, 233
    else:
        x_area_penal, x_linea_meta, x_area_chica = 0, 0, 0
        theta1, theta2 = -53, 53

    # Área penal: 18 yardas de profundidad, 44 yardas de ancho centrada en y=40
    area_penal = patches.Rectangle(
        (x_area_penal, 18), 18, 44, fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha, zorder=1
    )
    ax.add_patch(area_penal)

    # Área chica (área de meta): 6 yardas de profundidad, 20 yardas de ancho centrada en y=40
    area_chica = patches.Rectangle(
        (x_area_chica, 30), 6, 20, fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha, zorder=1
    )
    ax.add_patch(area_chica)

    # Línea de meta completa (fondo del campo) y portería (8 yardas de ancho, centrada en y=40)
    ax.plot([x_linea_meta, x_linea_meta], [0, 80], color=color, linewidth=linewidth, alpha=alpha * 0.5, zorder=1)
    ax.plot([x_linea_meta, x_linea_meta], [36, 44], color=color, linewidth=linewidth * 2.5, alpha=alpha, zorder=1)

    # Punto penal
    x_penal = 108 if lado == "ataque" else 12
    ax.scatter([x_penal], [40], s=10, color=color, alpha=alpha, zorder=1)

    # Arco de media luna del área penal (aprox.)
    arco = patches.Arc(
        (x_penal, 40), 20, 20, angle=0, theta1=theta1, theta2=theta2, color=color, linewidth=linewidth, alpha=alpha, zorder=1
    )
    ax.add_patch(arco)


# ---------------------------------------------------------------------------
# USO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo de uso (reemplaza con tu carga real vía statsbombpy):
    #
    # from statsbombpy import sb
    # creds = {"user": "tu_email", "passwd": "tu_password"}
    # events_df = sb.events(match_id=TU_MATCH_ID, creds=creds)
    # events_df["match_id"] = TU_MATCH_ID   # solo si hace falta añadir la columna
    #
    # posesiones     = build_possessions(events_df)
    # secuencias     = extract_offensive_sequences(events_df)
    # transiciones   = identify_transitions(events_df)
    # fases_presion  = identify_pressure_phases(events_df)
    pass
