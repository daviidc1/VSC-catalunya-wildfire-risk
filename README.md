# Predicción de riesgo de incendios forestales en Catalunya

Modelo de Machine Learning (Random Forest) que predice el riesgo de
incendio forestal por comarca y día, entrenado con datos reales de
incendios (2011-2024) y meteorología histórica de la XEMA (Meteocat).

Lo hice como proyecto personal para aplicar lo aprendido en cursos de
Machine Learning for Environmental Engineering y poder experimentar de una manera más personal

# Datos

- Incendios: histórico 2011-2024 de Dades Obertes de la Generalitat
  de Catalunya, agregado por comarca y fecha.
- Meteorología: temperatura máxima, humedad relativa mínima y
  velocidad del viento, de las estaciones de la XEMA (Meteocat).
- NDVI: pendiente de incorporar vía Google Earth Engine (ver
  `ndvi_gee_snippet.py`) de momento el modelo solo usa meteorología.

# Resultado actual

Con 203.441 filas comarca-día en el rango 2011-2024 (6.579 con incendio, ~3.2%):

| Métrica | Valor |
|---|---|
| Accuracy | 0.63 |
| Precision | 0.061 |
| Recall | 0.72 |
| ROC-AUC | 0.74 |

La precision es baja debido principalmente al fuerte desbalance de clases y al umbral utilizado, 
que prioriza detectar incendios reales y genera más falsos positivos: con un evento tan poco frecuente, el
modelo tiende a marcar riesgo con facilidad para no dejar pasar
incendios reales (recall alto), a costa de más falsas alarmas. El ROC-AUC de 0,74
indica que el modelo tiene una capacidad moderada para distinguir 
entre días con y sin incendio.

La humedad relativa mínima resultó la variable con más peso, por
delante de la propia comarca y de la temperatura máxima.

# Cosas que se torcieron por el camino (y cómo las resolví)

- El CSV de la XEMA venía en formato "extremadamente largo" (una fila por
  estación+fecha+variable) y con las tildes rotas por un problema de
  codificación. Tuve que identificar los códigos de variable exactos
  (`1.001` temp. máxima, `1.102` humedad mínima, `1.503` viento) y
  pivotar a formato ancho antes de poder cruzarlo con nada.
- Las estaciones no traen la comarca directamente, hizo falta el
  dataset de metadatos de la XEMA para mapear cada código de estación a
  su comarca.
- Una comarca (Moianès) se quedó sin match porque las coordenadas que
  usé para el mapa no la incluían, es una comarca relativamente reciente 
  (Palau de la Generalitat, 2015) y algunos datasets aún no la reflejan
  bien.
- El desbalanceo de clases (98% de los días sin incendio) hizo que las
  primeras versiones del modelo tuvieran una precision muy engañosa si
  no se mira junto al recall y al AUC.
- El meteo de la XEMA cubre desde 1988 hasta hoy, pero el histórico de
  incendios solo llega de 2011 a 2024. Al cruzar ambos sin recortar,
  ~119.000 filas de fuera de ese rango quedaban etiquetadas como "no
  incendio" sin serlo realmente (simplemente no había dato de incendio
  para esos años); tuve que recortar el dataset al rango real de
  cobertura de ambas fuentes antes de entrenar.

# Cómo ejecutarlo

```bash
pip install -r requirements.txt
python preparar_meteo.py   # pivota el CSV de la XEMA
python unir_datos.py       # cruza incendios + meteo + comarca
python train_model.py      # entrena y evalúa el modelo
```

Genera `modelo_incendios_catalunya.pkl`, `metricas.csv` y las gráficas
(importancia de variables, curva ROC, matriz de confusión, mapa de
riesgo por comarca).

# Próximos pasos

- Incorporar NDVI (sequedad de la vegetación) vía Earth Engine -> Señalado en el script
- Probar XGBoost y comparar con Random Forest
- Validación temporal en vez de split aleatorio (entrenar con años
  anteriores, testear con el último)
- Añadir topografía (pendiente, orientación)

------------------

# Forest fire risk prediction in Catalonia

Machine Learning model (Random Forest) that predicts forest fire risk
by county (comarca) and day, trained on real fire records (2011-2024)
and historical weather data from the XEMA (Meteocat) network.

I built this as a personal project to apply what I learned in Machine Learning for Environmental Engineering programs
to a real case from my own region and also as a first-hand training.

# Data

- Fires: 2011-2024 historical records from Dades Obertes de la Generalitat de Catalunya,
  aggregated by county and date.
- Weather: daily max temperature, minimum relative humidity and wind speed,
  from XEMA (Meteocat) weather stations.
- NDVI: pstill pending, via Google Earth Engine (see `ndvi_gee_snippet.py`)
  for now the model only uses weather data.

# Current results

203,441 county-day rows in the 2011-2024 range (6,579 with a fire, ~3.2%):

| Metric | Value |
|---|---|
| Accuracy | 0.63 |
| Precision | 0.061 |
| Recall | 0.72 |
| ROC-AUC | 0.74 |

The precision is low mainly due to the strong class imbalance and the decision threshold used. With
such a rare event, the model tends to flag more cases as high risk to avoid missing real fires
(high recall), at the cost of more false alarms. The ROC-AUC of 0.74 indicates that the model has
a moderate ability to distinguish between days with and without fires.

Minimum relative humidity was the most important variable, followed by the region itself and
maximum temperature.


# Things that went wrong along the way (and how I fixed them)

- The XEMA CSV came in a "ultra long" format (one row per station+date+variable) and had broken
accents due to an encoding issue. I had to identify the exact variable codes
(`1.001` max. temperature, `1.102` minimum humidity, `1.503` wind speed) and pivot the
data into a wide format before joining it with the other datasets.

- The stations do not include the region directly, so I used the XEMA metadata dataset to
map each station code to its corresponding region.

- One region (Moianès) initially had no match because the coordinates used for the map did
not include it. It is a relatively recent region (Palau de la Generalitat, 2015), and some datasets
still do not represent it correctly.

- Class imbalance (98% of days without fires) made the precision in the first versions of
the model quite misleading when considered without recall and AUC.

- XEMA weather data goes back to 1988, while the fire dataset only covers 2011–2024. When
combining both datasets without restricting the date range, around 119,000 records outside
the fire dataset's coverage were incorrectly labelled as "no fire". I therefore restricted
the dataset to the period covered by both sources before training the model.

# How to run

```bash
pip install -r requirements.txt
python preparar_meteo.py   # pivota el CSV de la XEMA
python unir_datos.py       # cruza incendios + meteo + comarca
python train_model.py      # entrena y evalúa el modelo
```

This generates `modelo_incendios_catalunya.pkl`, `metricas.csv` and the following plots:
feature importance, ROC curve, confusion matrix and a regional risk map.

# Next steps

- Add NDVI (vegetation dryness) using Google Earth Engine
- Test XGBoost and compare it with Random Forest
- Use temporal validation instead of a random split 
  (train on previous years and test on the latest year)
- Add topographic variables such as slope and aspect