"""
Une incendios reales + meteo + comarca en dataset_final.csv, listo
para train_model.py.

Incendios: una fila por incendio (fecha + comarca) -> se convierte en
las filas positivas (incendio=1). El resto de días/comarcas sin
incendio registrado quedan como negativos (incendio=0).

Meteo: viene por estación, así que primero se mapea cada estación a su
comarca (archivo de metadatos de la XEMA) y se agrega por comarca+día.

"""

import pandas as pd

ARCHIVO_INCENDIOS = "Incendis_forestals_a_Catalunya._Anys_2011-2024_20260911.csv"
ARCHIVO_METEO = "meteo_limpio.csv"
ARCHIVO_ESTACIONES = "metadades_xema.csv"

COL_CODI_ESTACIO_EN_METADATOS = "CODI_ESTACIO"
COL_COMARCA_EN_METADATOS = "NOM_COMARCA"


def cargar_incendios():
    df = pd.read_csv(ARCHIVO_INCENDIOS)
    df = df.rename(columns={"DATA INCENDI": "fecha", "COMARCA": "comarca"})
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    # Nos quedamos con las combinaciones únicas comarca+fecha que tuvieron
    # incendio (puede haber varias filas el mismo día en la misma comarca
    # si hubo más de un incendio, o en municipios distintos)
    positivos = df[["comarca", "fecha"]].drop_duplicates()
    positivos["incendio"] = 1
    return positivos


def cargar_meteo_con_comarca():
    meteo = pd.read_csv(ARCHIVO_METEO, parse_dates=["fecha"])
    estaciones = pd.read_csv(ARCHIVO_ESTACIONES)

    estaciones = estaciones.rename(
        columns={
            COL_CODI_ESTACIO_EN_METADATOS: "codi_estacio",
            COL_COMARCA_EN_METADATOS: "comarca",
        }
    )[["codi_estacio", "comarca"]]

    meteo_comarca = meteo.merge(estaciones, on="codi_estacio", how="inner")

    filas_antes = len(meteo)
    filas_despues = len(meteo_comarca)
    print(f"Meteo: {filas_antes} filas -> {filas_despues} filas tras mapear a comarca "
          f"({filas_antes - filas_despues} filas de estaciones sin comarca conocida se descartaron)")

    # Agrega por comarca + fecha (media de todas las estaciones de esa comarca)
    agregado = (
        meteo_comarca.groupby(["comarca", "fecha"])[["temp_max", "hum_rel_min", "viento_kmh"]]
        .mean()
        .reset_index()
    )
    return agregado


def unir_todo():
    incendios = cargar_incendios()
    meteo = cargar_meteo_con_comarca()

    # Solo nos quedamos con el rango de fechas donde el dataset de
    # incendios tiene cobertura real. Fuera de ese rango, un "no
    # incendio" no significa nada (no es que no hubiera, es que no
    # tenemos el dato), así que esas filas meterían ruido/sesgo al modelo.
    fecha_min_incendios = incendios["fecha"].min()
    fecha_max_incendios = incendios["fecha"].max()
    print(f"Cobertura del histórico de incendios: {fecha_min_incendios.date()} a {fecha_max_incendios.date()}")

    antes = len(meteo)
    meteo = meteo[
        (meteo["fecha"] >= fecha_min_incendios) & (meteo["fecha"] <= fecha_max_incendios)
    ]
    print(f"Meteo recortado al rango de incendios: {antes} -> {len(meteo)} filas")

    # Comprobación: ¿hay comarcas en incendios que no aparecen en meteo
    # (por diferencias de acentos/mayúsculas)? Si esta lista no está
    # vacía, hay que revisar y corregir esos nombres antes de continuar.
    comarcas_incendios = set(incendios["comarca"].unique())
    comarcas_meteo = set(meteo["comarca"].unique())
    solo_en_incendios = comarcas_incendios - comarcas_meteo
    if solo_en_incendios:
        print("\n[AVISO] Estas comarcas del archivo de incendios NO aparecen "
              "en el archivo de estaciones (revisa acentos/mayúsculas):")
        print(sorted(solo_en_incendios))
        print()

    df = meteo.merge(incendios, on=["comarca", "fecha"], how="left")
    df["incendio"] = df["incendio"].fillna(0).astype(int)

    # Quita filas sin datos meteorológicos válidos
    antes = len(df)
    df = df.dropna(subset=["temp_max", "hum_rel_min", "viento_kmh"])
    print(f"Se eliminaron {antes - len(df)} filas con meteo incompleta "
          f"(quedan {len(df)} filas)")

    print(f"\nTotal filas: {len(df)}")
    print(f"Incendios: {df['incendio'].sum()}  |  No incendios: {(df['incendio']==0).sum()}")

    df.to_csv("dataset_final.csv", index=False)
    print("\nGuardado en dataset_final.csv — este es el archivo que usará train_model.py")
    print(df.head())


if __name__ == "__main__":
    unir_todo()
