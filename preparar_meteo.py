"""
preparar_meteo.py

Convierte el CSV "largo" de meteo (una fila por estación+fecha+variable)
en un CSV "ancho" (una fila por estación+fecha, con columnas separadas
para temp_max, hum_rel_min, viento_kmh), listo para unir con los datos
de incendios.

Lee el archivo por trozos (chunks) para no reventar la memoria aunque
el CSV sea enorme, y solo se queda con las 3 variables que necesitamos.

"""

import pandas as pd

NOMBRE_ARCHIVO = "xema_dades.csv" 
TAMANO_CHUNK = 200_000 # Si vuestro ordenador es potente, puedes subirlo a 1_000_000 o incluso más


CODIGOS_DE_INTERES = {
    1.001: "temp_max",       # Temperatura màxima diària (°C)
    1.102: "hum_rel_min",    # Humitat relativa mínima diària (%)
    1.503: "viento_ms",      # Velocitat mitjana diària del vent 10m, escalar (m/s)
}


def limpiar_valor(serie):
    """Pasa de '23,3' (coma decimal) a 23.3 (float)."""
    return pd.to_numeric(
        serie.astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )


def procesar():
    trozos_filtrados = []

    # Si ves símbolos raros (�) al imprimir NOM_VARIABLE, prueba a
    # cambiar encoding="utf-8" por encoding="latin-1" aquí abajo
    lector = pd.read_csv(
        NOMBRE_ARCHIVO,
        usecols=["CODI_ESTACIO", "DATA_LECTURA", "CODI_VARIABLE", "VALOR"],
        chunksize=TAMANO_CHUNK,
        encoding="utf-8",
    )

    total_filas = 0
    for i, chunk in enumerate(lector):
        chunk = chunk[chunk["CODI_VARIABLE"].isin(CODIGOS_DE_INTERES.keys())].copy()
        if len(chunk):
            chunk["VALOR"] = limpiar_valor(chunk["VALOR"])
            trozos_filtrados.append(chunk)
        total_filas += len(chunk)
        print(f"Trozo {i+1} procesado, {len(chunk)} filas relevantes encontradas...")

    if not trozos_filtrados:
        print("No se encontró ninguna fila con esos códigos de variable. "
              "Revisar CODIGOS_DE_INTERES y los códigos que viste en listar_variables.py.")
        return

    df = pd.concat(trozos_filtrados, ignore_index=True)

    df = df.drop_duplicates(subset=["CODI_ESTACIO", "DATA_LECTURA", "CODI_VARIABLE"])

    df["fecha"] = pd.to_datetime(df["DATA_LECTURA"], dayfirst=True, errors="coerce")
    df["variable"] = df["CODI_VARIABLE"].map(CODIGOS_DE_INTERES)

    ancho = df.pivot_table(
        index=["CODI_ESTACIO", "fecha"],
        columns="variable",
        values="VALOR",
        aggfunc="mean",
    ).reset_index()

    ancho = ancho.rename(columns={"CODI_ESTACIO": "codi_estacio"})

    if "viento_ms" in ancho.columns:
        ancho["viento_kmh"] = (ancho["viento_ms"] * 3.6).round(1)
        ancho = ancho.drop(columns=["viento_ms"])

    ancho.to_csv("meteo_limpio.csv", index=False)
    print(f"\nListo. {len(ancho)} filas (estación+día) guardadas en meteo_limpio.csv")
    print(ancho.head())


if __name__ == "__main__":
    procesar()
