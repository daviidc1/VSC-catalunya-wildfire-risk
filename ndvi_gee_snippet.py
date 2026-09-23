"""
ndvi_gee_snippet.py

Snippet de referencia para extraer el NDVI real (Sentinel-2) usando
la API de Python de Google Earth Engine (Aún no ejecutado, pendiente de uso para mejora).

  pip install earthengine-api geemap
  earthengine authenticate     (una vez, te pedirá loguearte con Google)

"""

import ee
import pandas as pd

ee.Initialize()


def obtener_ndvi(lat, lon, fecha, dias_ventana=15):
    punto = ee.Geometry.Point([lon, lat])
    fecha_ee = ee.Date(fecha)
    inicio = fecha_ee.advance(-dias_ventana, "day")
    fin = fecha_ee.advance(dias_ventana, "day")

    coleccion = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(punto)
        .filterDate(inicio, fin)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    def calcular_ndvi(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return img.addBands(ndvi)

    con_ndvi = coleccion.map(calcular_ndvi)
    imagen_mediana = con_ndvi.select("NDVI").median()

    valor = imagen_mediana.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=punto.buffer(500), scale=10
    ).get("NDVI")

    try:
        return valor.getInfo()
    except Exception:
        return None


def procesar_dataframe(df, col_lat="lat", col_lon="lon", col_fecha="fecha"):
    ndvis = []
    for _, fila in df.iterrows():
        ndvi = obtener_ndvi(fila[col_lat], fila[col_lon], str(fila[col_fecha].date()))
        ndvis.append(ndvi)
    df = df.copy()
    df["ndvi"] = ndvis
    return df
