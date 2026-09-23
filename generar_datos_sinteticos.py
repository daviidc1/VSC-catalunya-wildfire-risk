"""
generar_datos_sinteticos.py

"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

COMARQUES = {
    "Alt Camp": (41.286, 1.249),
    "Alt Empordà": (42.267, 2.961),
    "Alt Penedès": (41.345, 1.699),
    "Alt Urgell": (42.358, 1.463),
    "Alta Ribagorça": (42.407, 0.744),
    "Anoia": (41.579, 1.617),
    "Bages": (41.728, 1.824),
    "Baix Camp": (41.156, 1.107),
    "Baix Ebre": (40.812, 0.521),
    "Baix Empordà": (41.960, 3.036),
    "Baix Llobregat": (41.383, 2.044),
    "Baix Penedès": (41.220, 1.535),
    "Barcelonès": (41.385, 2.173),
    "Berguedà": (42.104, 1.845),
    "Cerdanya": (42.433, 1.928),
    "Conca de Barberà": (41.376, 1.161),
    "Garraf": (41.224, 1.725),
    "Garrigues": (41.522, 0.868),
    "Garrotxa": (42.181, 2.489),
    "Gironès": (41.983, 2.824),
    "Lluçanès": (42.017, 2.030),
    "Maresme": (41.538, 2.445),
    "Moianès": (41.812, 2.097),
    "Montsià": (40.706, 0.581),
    "Noguera": (41.790, 0.807),
    "Osona": (41.930, 2.254),
    "Pallars Jussà": (42.166, 0.895),
    "Pallars Sobirà": (42.412, 1.129),
    "Pla d'Urgell": (41.631, 0.895),
    "Pla de l'Estany": (42.119, 2.767),
    "Priorat": (41.146, 0.821),
    "Ribera d'Ebre": (41.093, 0.644),
    "Ripollès": (42.201, 2.191),
    "Segarra": (41.671, 1.272),
    "Segrià": (41.617, 0.620),
    "Selva": (41.858, 2.667),
    "Solsonès": (41.994, 1.517),
    "Tarragonès": (41.117, 1.255),
    "Terra Alta": (41.053, 0.437),
    "Urgell": (41.647, 1.139),
    "Vallès Occidental": (41.548, 2.107),
    "Vallès Oriental": (41.608, 2.288),
    "Val d'Aran": (42.702, 0.796),
}


COMARQUES_ALTO_RIESGO = {
    "Ribera d'Ebre", "Priorat", "Terra Alta", "Garrigues", "Baix Ebre",
    "Conca de Barberà", "Segrià", "Alt Camp", "Baix Camp", "Anoia",
    "Montsià",
}


def generar_dataset(dias_totales=1800, semilla=42):
    global RNG
    RNG = np.random.default_rng(semilla)

    comarques = list(COMARQUES.keys())
    n_comarques = len(comarques)

    fechas = pd.date_range("2019-01-01", periods=dias_totales, freq="D")

    filas = []
    for comarca in comarques:
        lat, lon = COMARQUES[comarca]
        es_alto_riesgo = comarca in COMARQUES_ALTO_RIESGO

        for fecha in fechas:
            mes = fecha.month
            es_verano = mes in (6, 7, 8, 9)

            temp_max = RNG.normal(24 if es_verano else 12, 6)
            hum_rel_min = RNG.normal(45 if es_verano else 60, 15)
            viento_kmh = RNG.gamma(2.0, 8.0)
            ndvi = RNG.normal(0.55, 0.15)

            riesgo = -6.0
            riesgo += 2.5 if es_verano else 0
            riesgo += 0.08 * max(temp_max - 25, 0)
            riesgo += 0.05 * max(35 - hum_rel_min, 0)
            riesgo += 0.04 * max(viento_kmh - 20, 0)
            riesgo += 3.0 * max(0.5 - ndvi, 0)
            riesgo += 1.2 if es_alto_riesgo else 0
            riesgo += RNG.normal(0, 1.0)

            prob_incendio = 1 / (1 + np.exp(-riesgo))
            incendio = 1 if RNG.random() < prob_incendio else 0

            filas.append(
                (comarca, lat, lon, fecha, mes, temp_max, hum_rel_min,
                 viento_kmh, ndvi, incendio)
            )

    df = pd.DataFrame(
        filas,
        columns=["comarca", "lat", "lon", "fecha", "mes", "temp_max",
                 "hum_rel_min", "viento_kmh", "ndvi", "incendio"],
    )

    df["temp_max"] = df["temp_max"].round(1)
    df["hum_rel_min"] = df["hum_rel_min"].clip(5, 100).round(1)
    df["viento_kmh"] = df["viento_kmh"].clip(0, None).round(1)
    df["ndvi"] = df["ndvi"].clip(-1, 1).round(3)

    positivos = df[df["incendio"] == 1]
    negativos = df[df["incendio"] == 0].sample(
        n=min(len(positivos) * 3, len(df[df["incendio"] == 0])),
        random_state=semilla,
    )
    df = pd.concat([positivos, negativos]).sample(
        frac=1, random_state=semilla
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = generar_dataset()
    out_path = "datos_incendios_sinteticos.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset sintético generado: {out_path} ({len(df)} filas)")
    print(f"Comarcas: {df['comarca'].nunique()}  |  Incendios: {df['incendio'].sum()}")
    print(df.head())
