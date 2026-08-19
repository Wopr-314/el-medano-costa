# El Médano · Costa

Estimación horaria de dónde llegarían los vertidos al mar en el litoral de
El Médano y La Tejita (Granadilla de Abona, Tenerife).

**No mide contaminación: estima transporte.** Las analíticas de aguas de baño del
Ayuntamiento y la bandera de la playa mandan siempre sobre esta página.

## Cómo funciona

Una GitHub Action se ejecuta cada hora, descarga el viento (Open-Meteo) y la
corriente superficial (Copernicus Marine) de la zona, y se los pasa a un emulador
neuronal que aprendió a imitar a [OpenDrift](https://opendrift.github.io/),
un simulador lagrangiano de deriva de partículas. El resultado se publica como
`pluma.png` (overlay georreferenciado) y `prediccion.json`; la página solo los
muestra.

El emulador se entrenó con 500 simulaciones lanzadas desde los diez puntos de
vertido activos del censo oficial de Canarias de 2025.

## Precisión

Contrastado con 75 simulaciones que la red nunca vio: el centro de la pluma
predicha cae a **350 m** de mediana respecto a la simulación física, y a **996 m**
en el peor decil. Basta para elegir playa, no para elegir un punto de la arena.

La posición de la señal a lo largo de la costa es la parte fiable; su extensión
mar adentro se exagera.

## Archivos

| Archivo | Qué es |
|---|---|
| `index.html` | La página |
| `emulador.onnx` | El modelo (484 mil parámetros, 1,9 MB) |
| `estaticos.bin` | Máscara de tierra y mapa de focos |
| `metadatos.json` | Normalización, geometría, focos y avisos |
| `12_prediccion_web.py` | Lo que ejecuta la Action cada hora |
| `pluma.png`, `prediccion.json` | La última predicción publicada |

## Puesta en marcha

1. Settings → Secrets and variables → Actions → New repository secret:
   `CMEMS_USUARIO` y `CMEMS_PASSWORD` (cuenta gratuita en
   [Copernicus Marine](https://data.marine.copernicus.eu/register)).
2. Settings → Pages → Deploy from a branch → `main` / `/ (root)`.
3. Actions → *Predicción horaria* → **Run workflow**, para no esperar a la hora.

## Fuentes

Censo de vertidos tierra-mar 2025, Gobierno de Canarias (SITCAN Open Data) ·
Copernicus Marine Service, producto IBI · Open-Meteo · OpenStreetMap ·
Esri World Imagery · Ayuntamiento de Granadilla de Abona.
