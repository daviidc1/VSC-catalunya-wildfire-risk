"""
train_model.py

Pipeline completo: carga de datos -> limpieza -> entrenamiento ->
evaluación -> guardado del modelo y de las gráficas.

"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib

from generar_datos_sinteticos import generar_dataset

FEATURES_NUMERICAS_POSIBLES = ["temp_max", "hum_rel_min", "viento_kmh", "ndvi"]
FEATURE_CATEGORICA = "comarca"
TARGET = "incendio"


def cargar_datos():
    """
    Carga el dataset ya preparado por unir_datos.py.
    """
    print("Cargando dataset_final.csv (datos REALES)...\n")
    df = pd.read_csv("dataset_final.csv", parse_dates=["fecha"])
    return df


def limpiar_datos(df):
    global FEATURES_NUMERICAS
    FEATURES_NUMERICAS = [c for c in FEATURES_NUMERICAS_POSIBLES if c in df.columns]
    print(f"Variables numéricas disponibles: {FEATURES_NUMERICAS}")
    if "ndvi" not in FEATURES_NUMERICAS:
        print("(Sin NDVI todavía. El modelo usará solo meteorología. "
              "Añadir ndvi snippet cuando esté disponible)\n")

    antes = len(df)
    df = df.dropna(subset=FEATURES_NUMERICAS + [FEATURE_CATEGORICA, TARGET]).copy()
    despues = len(df)
    if antes != despues:
        print(f"Se eliminaron {antes - despues} filas con valores nulos.")
    return df


def entrenar_y_evaluar(df):
    dummies_comarca = pd.get_dummies(df[FEATURE_CATEGORICA], prefix="comarca")
    X = pd.concat([df[FEATURES_NUMERICAS], dummies_comarca], axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("=== Métricas en el conjunto de test ===")
    for k, v in metricas.items():
        print(f"  {k:10s}: {v:.3f}")

    return modelo, X_test, y_test, y_pred, y_proba, metricas


def graficar_resultados(modelo, X_test, y_test, y_pred, y_proba):
    importancias_raw = pd.Series(modelo.feature_importances_, index=X_test.columns)
    es_comarca = importancias_raw.index.str.startswith("comarca_")
    importancias = pd.concat(
        [
            importancias_raw[~es_comarca],
            pd.Series({"comarca": importancias_raw[es_comarca].sum()}),
        ]
    ).sort_values()

    fig, ax = plt.subplots(figsize=(6, 4))
    importancias.plot.barh(ax=ax, color="#d9480f")
    ax.set_title("Importancia de variables (Random Forest)")
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    fig.savefig("importancia_variables.png", dpi=150)
    plt.close(fig)

    # 2. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#1864ab", label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Falsos positivos")
    ax.set_ylabel("Verdaderos positivos")
    ax.set_title("Curva ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig("curva_roc.png", dpi=150)
    plt.close(fig)

    # 3. Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No incendio", "Incendio"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Oranges", colorbar=False)
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    fig.savefig("matriz_confusion.png", dpi=150)
    plt.close(fig)

    print("\nGráficas guardadas: importancia_variables.png, curva_roc.png, matriz_confusion.png")


def mapa_riesgo(df, modelo):
    from generar_datos_sinteticos import COMARQUES  # coordenadas de cada comarca

    dummies_comarca = pd.get_dummies(df[FEATURE_CATEGORICA], prefix="comarca")
    X = pd.concat([df[FEATURES_NUMERICAS], dummies_comarca], axis=1)
    df = df.copy()
    df["riesgo_predicho"] = modelo.predict_proba(X)[:, 1]

    riesgo_por_comarca = (
        df.groupby("comarca")["riesgo_predicho"].mean().reset_index()
    )
    riesgo_por_comarca["lat"] = riesgo_por_comarca["comarca"].map(
        lambda c: COMARQUES.get(c, (None, None))[0]
    )
    riesgo_por_comarca["lon"] = riesgo_por_comarca["comarca"].map(
        lambda c: COMARQUES.get(c, (None, None))[1]
    )
    sin_coords = riesgo_por_comarca["lat"].isna().sum()
    if sin_coords:
        print(f"[AVISO] {sin_coords} comarca(s) sin coordenadas conocidas, "
              f"no aparecerán en el mapa: "
              f"{riesgo_por_comarca[riesgo_por_comarca['lat'].isna()]['comarca'].tolist()}")
    riesgo_por_comarca = riesgo_por_comarca.dropna(subset=["lat", "lon"])

    fig, ax = plt.subplots(figsize=(6, 7))
    sc = ax.scatter(
        riesgo_por_comarca["lon"],
        riesgo_por_comarca["lat"],
        c=riesgo_por_comarca["riesgo_predicho"],
        cmap="YlOrRd",
        s=140,
        edgecolors="black",
        linewidths=0.4,
        zorder=3
    )
    # Nombre de cada comarca junto a su punto, para poder identificarlas en el mapa (UPDATED 23/09/2026)
    for _, fila in riesgo_por_comarca.iterrows():
            ax.annotate(
                fila["comarca"],
                (fila["lon"], fila["lat"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
                zorder=4,
            )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Riesgo medio de incendio por comarca (Catalunya)")
    plt.colorbar(sc, label="Probabilidad media de incendio")
    fig.tight_layout()
    fig.savefig("mapa_riesgo.png", dpi=150)
    plt.close(fig)
    print("Gráfica guardada: mapa_riesgo.png")

    top5 = riesgo_por_comarca.sort_values("riesgo_predicho", ascending=False).head(5)
    print("\nTop 5 comarcas con mayor riesgo medio predicho:")
    print(top5[["comarca", "riesgo_predicho"]].to_string(index=False))


def main():
    df = cargar_datos()
    df = limpiar_datos(df)

    print(f"Total de registros: {len(df)}")
    print(f"Incendios: {df[TARGET].sum()}  |  No incendios: {(df[TARGET]==0).sum()}\n")

    modelo, X_test, y_test, y_pred, y_proba, metricas = entrenar_y_evaluar(df)
    graficar_resultados(modelo, X_test, y_test, y_pred, y_proba)
    mapa_riesgo(df, modelo)

    joblib.dump(modelo, "modelo_incendios_catalunya.pkl")
    print("\nModelo guardado en modelo_incendios_catalunya.pkl")

    pd.Series(metricas).to_csv("metricas.csv")
    print("Métricas guardadas en metricas.csv")


if __name__ == "__main__":
    main()
