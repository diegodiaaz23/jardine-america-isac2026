# Jardine en América: principios tácticos y ajustes por contexto

Proyecto para el **Hackathon ISAC 2026** (ITAM Sports Analytics Conference, patrocinado por Club América).
Autores: Diego Díaz Abascal y Emiliano Chain Ruiz.

## Pregunta
¿Cuáles son los principios que definen a su equipo en fase ofensiva y defensiva, cómo se manifiestan estos patrones a lo largo del tiempo, y de qué manera el entrenador ajusta su comportamiento en función del contexto del partido (rival, marcador, localía o momento del juego)?

## Qué hicimos
- Analizamos 129 partidos y 442,202 eventos de StatsBomb del América de André Jardine (2023/24 a 2025/26).
- Lo comparamos con tres técnicos también campeones de Liga MX: Almada (Pachuca), Cocca (Atlas) y Mohamed (Toluca).
- Modelo de regresión logística con métricas del primer tiempo (log-loss 0.859 contra 1.115 de la referencia) y un semáforo de riesgo al descanso.
- Análisis causal (emparejamiento exacto grueso y bootstrap): efecto de −5.8 pp con intervalo [−24.2, +11.6] que incluye el cero. Es un resultado nulo con pocos casos.
- Simulador de decisiones al descanso con 12 escenarios reales.

## Estructura
- `notebooks/`: análisis, del 00 (preparación y framework) al 10 (comparación de entrenadores).
- `src/`: funciones del framework analítico y cálculos de cada bloque.
- `scripts_auxiliares/`: scripts de apoyo (balón parado, field tilt minuto a minuto, contexto del simulador).
- `entregables/`: informe interactivo en HTML y simulador.

## Datos y credenciales
Los datos de eventos son de StatsBomb y **no se incluyen** en este repositorio, ni tampoco las credenciales de acceso (que son privadas). Para reproducir el análisis necesitas tu propio acceso:

```bash
pip install -r requirements.txt
export SB_USERNAME="tu_usuario"
export SB_PASSWORD="tu_password"
jupyter notebook
```

Los notebooks se entregan sin salidas ejecutadas para no redistribuir datos de StatsBomb. Al ejecutarlos con tu acceso se regeneran. El notebook 00 contiene los marcadores `<TU_USUARIO_STATSBOMB>` y `<TU_PASSWORD_STATSBOMB>`: sustitúyelos o usa las variables de entorno.

## Notas metodológicas
- Field tilt relativo = acciones del equipo con x > 80 ÷ (acciones propias con x > 80 + acciones del rival con x < 40), volteando las coordenadas del rival.
- PPDA: menor valor significa más presión. Solo es comparable de forma relativa con datos de StatsBomb.
- Los umbrales del semáforo son exploratorios y las muestras por situación son pequeñas.
