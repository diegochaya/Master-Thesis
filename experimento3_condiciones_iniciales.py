from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from Preprocesado import extraer_features


CARPETA_DATAFRAMES = Path("dataframesCarpetas") / "dataframesComprobacionesIC"
RUTA_INDICE = CARPETA_DATAFRAMES / "indice_simulaciones.csv"
CARPETA_SALIDA = Path("PreliminaryExperiments") / "experimento3_condiciones_iniciales"

EXCLUIR_BORDE = True
N_CENTER_CELLS = 196
N_COMPONENTES_PCA = 10

INIT_LABELS = {
    "hexagonal_perfect_14x14.dat": "I_perfect",
    "hexagonal_2000_1.dat": "I_2000",
    "hexagonal_1000_1.dat": "I_1000_1",
    "hexagonal_1000_2.dat": "I_1000_2",
}
INIT_ORDER = ["I_perfect", "I_2000", "I_1000_1", "I_1000_2"]


CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

indice = pd.read_csv(RUTA_INDICE)
indice = indice.loc[indice["status"].astype(str) == "ok"].copy()
indice["grupo_ic"] = indice["init_id"].map(INIT_LABELS)

if indice["grupo_ic"].isna().any():
    desconocidos = sorted(indice.loc[indice["grupo_ic"].isna(), "init_id"].unique())
    raise ValueError(f"Condiciones iniciales no esperadas: {desconocidos}")

indice["grupo_ic"] = pd.Categorical(
    indice["grupo_ic"],
    categories=INIT_ORDER,
    ordered=True,
)
indice = indice.sort_values(["grupo_ic", "replica"]).reset_index(drop=True)

print("Simulaciones ok por condicion inicial:")
print(indice.groupby("grupo_ic", observed=True).size().to_string())

filas_features = []
for _, fila_indice in indice.iterrows():
    sim_id = str(fila_indice["sim_id"])
    df_final = pd.read_pickle(Path(fila_indice["filepath"]))

    features = extraer_features(
        {sim_id: df_final},
        excluir_borde=EXCLUIR_BORDE,
        n_center_cells=N_CENTER_CELLS,
    )

    if features.empty:
        print(f"WARNING: no se pudieron extraer features de {sim_id}")
        continue

    fila_features = features.iloc[0].to_dict()
    fila_features["sim_id"] = sim_id
    fila_features["replica"] = int(fila_indice["replica"])
    fila_features["init_id"] = str(fila_indice["init_id"])
    fila_features["grupo_ic"] = str(fila_indice["grupo_ic"])
    filas_features.append(fila_features)

datos = pd.DataFrame(filas_features)
datos["grupo_ic"] = pd.Categorical(
    datos["grupo_ic"],
    categories=INIT_ORDER,
    ordered=True,
)
datos = datos.sort_values(["grupo_ic", "replica"]).reset_index(drop=True)

columnas_metadata = ["sim_id", "replica", "init_id", "grupo_ic", "lambda_cc", "gamma"]
columnas_features = [c for c in datos.columns if c not in columnas_metadata]
datos = datos[columnas_metadata + columnas_features]
datos.to_csv(CARPETA_SALIDA / "features_por_simulacion.csv", index=False)

scaler = StandardScaler()
X = scaler.fit_transform(datos[columnas_features].to_numpy(dtype=float))
y = datos["grupo_ic"].astype(str).to_numpy()

n_componentes = min(N_COMPONENTES_PCA, X.shape[0], X.shape[1])
pca = PCA(n_components=n_componentes)
X_pca = pca.fit_transform(X)

pca_scores = datos[columnas_metadata].copy()
for i in range(n_componentes):
    pca_scores[f"PC{i + 1}"] = X_pca[:, i]
pca_scores.to_csv(CARPETA_SALIDA / "pca_scores.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
colores = {
    grupo: color
    for grupo, color in zip(
        INIT_ORDER,
        ["tab:blue", "tab:orange", "tab:green", "tab:red"],
    )
}
for grupo, datos_grupo in pca_scores.groupby("grupo_ic", sort=False, observed=True):
    grupo = str(grupo)
    ax.scatter(
        datos_grupo["PC1"],
        datos_grupo["PC2"],
        label=grupo,
        s=42,
        alpha=0.85,
        color=colores[grupo],
        edgecolors="black",
        linewidths=0.35,
    )

for grupo, datos_grupo in pca_scores.groupby("grupo_ic", sort=False, observed=True):
    grupo = str(grupo)
    ax.scatter(
        datos_grupo["PC1"].mean(),
        datos_grupo["PC2"].mean(),
        s=150,
        marker="X",
        color=colores[grupo],
        edgecolors="black",
        linewidths=1.0,
    )

ax.axhline(0, color="0.85", linewidth=0.8)
ax.axvline(0, color="0.85", linewidth=0.8)
ax.set_title("PCA por condicion inicial")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)")
ax.legend(title="Condicion inicial", fontsize=8)
fig.savefig(CARPETA_SALIDA / "pca_2d_condiciones_iniciales.png", dpi=180, bbox_inches="tight")
plt.close(fig)

pca_varianza = pd.DataFrame(
    {
        "componente": [f"PC{i + 1}" for i in range(n_componentes)],
        "varianza_explicada": pca.explained_variance_ratio_,
        "varianza_explicada_acumulada": np.cumsum(pca.explained_variance_ratio_),
    }
)
pca_varianza.to_csv(CARPETA_SALIDA / "pca_varianza.csv", index=False)

pca_componentes = pd.DataFrame(
    pca.components_.T,
    index=columnas_features,
    columns=[f"PC{i + 1}" for i in range(n_componentes)],
)
pca_componentes["abs_max"] = pca_componentes.abs().max(axis=1)
pca_componentes = pca_componentes.sort_values("abs_max", ascending=False)
pca_componentes = pca_componentes.drop(columns="abs_max")
pca_componentes.to_csv(CARPETA_SALIDA / "pca_componentes.csv")

datos_escalados = datos[columnas_metadata].copy()
for i, columna in enumerate(columnas_features):
    datos_escalados[columna] = X[:, i]

centroides = (
    datos_escalados.groupby(["grupo_ic"], observed=True)[columnas_features]
    .mean()
    .reindex(INIT_ORDER)
    .reset_index()
)
centroides.to_csv(CARPETA_SALIDA / "centroides_escalados_por_grupo.csv", index=False)

distancias_centroide = []
for grupo, datos_grupo in datos_escalados.groupby("grupo_ic", sort=False, observed=True):
    grupo = str(grupo)
    matriz = datos_grupo[columnas_features].to_numpy(dtype=float)
    centroide = matriz.mean(axis=0)
    distancias = np.linalg.norm(matriz - centroide, axis=1)

    distancias_centroide.append(
        {
            "grupo_ic": grupo,
            "within_mean": distancias.mean(),
            "within_std": distancias.std(ddof=1),
            "within_median": np.median(distancias),
            "within_max": distancias.max(),
        }
    )

distancias_centroide = pd.DataFrame(distancias_centroide)
distancias_centroide.to_csv(
    CARPETA_SALIDA / "resumen_distancias_intra_grupo.csv",
    index=False,
)

distancias_entre_centroides = []
for i in range(len(centroides)):
    for j in range(i + 1, len(centroides)):
        centroide_i = centroides.loc[i, columnas_features].to_numpy(dtype=float)
        centroide_j = centroides.loc[j, columnas_features].to_numpy(dtype=float)
        distancia = np.linalg.norm(centroide_i - centroide_j)

        grupo_i = str(centroides.loc[i, "grupo_ic"])
        grupo_j = str(centroides.loc[j, "grupo_ic"])
        within_i = distancias_centroide.loc[
            distancias_centroide["grupo_ic"] == grupo_i, "within_mean"
        ].iloc[0]
        within_j = distancias_centroide.loc[
            distancias_centroide["grupo_ic"] == grupo_j, "within_mean"
        ].iloc[0]

        distancias_entre_centroides.append(
            {
                "grupo_1": grupo_i,
                "grupo_2": grupo_j,
                "distancia_centroides": distancia,
                "within_mean_1": within_i,
                "within_mean_2": within_j,
                "within_mean_promedio": (within_i + within_j) / 2,
                "separation_ratio": distancia / ((within_i + within_j) / 2),
            }
        )

distancias_entre_centroides = pd.DataFrame(distancias_entre_centroides)
distancias_entre_centroides.to_csv(
    CARPETA_SALIDA / "distancias_entre_centroides_y_ratios.csv",
    index=False,
)

silhouette_global = silhouette_score(X, y)
silhouette_por_muestra = silhouette_samples(X, y)

silhouette_detalle = datos[columnas_metadata].copy()
silhouette_detalle["silhouette"] = silhouette_por_muestra
silhouette_detalle.to_csv(CARPETA_SALIDA / "silhouette_por_simulacion.csv", index=False)

silhouette_resumen = (
    silhouette_detalle.groupby("grupo_ic", observed=True)["silhouette"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .reindex(INIT_ORDER)
    .reset_index()
)
silhouette_resumen.to_csv(CARPETA_SALIDA / "silhouette_por_grupo.csv", index=False)

predicciones = []
for i in range(len(datos_escalados)):
    grupo_real = str(datos_escalados.loc[i, "grupo_ic"])
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i)

    centroides_loo = []
    for grupo in INIT_ORDER:
        centroide = X_train[y_train == grupo].mean(axis=0)
        centroides_loo.append({"grupo": grupo, "centroide": centroide})

    distancias = [
        np.linalg.norm(X[i] - entrada["centroide"]) for entrada in centroides_loo
    ]
    idx_min = int(np.argmin(distancias))
    grupo_predicho = centroides_loo[idx_min]["grupo"]

    predicciones.append(
        {
            "sim_id": datos_escalados.loc[i, "sim_id"],
            "replica": datos_escalados.loc[i, "replica"],
            "init_id": datos_escalados.loc[i, "init_id"],
            "lambda_cc": datos_escalados.loc[i, "lambda_cc"],
            "gamma": datos_escalados.loc[i, "gamma"],
            "grupo_real": grupo_real,
            "grupo_predicho": grupo_predicho,
            "distancia_minima": distancias[idx_min],
            "acierto": grupo_real == grupo_predicho,
        }
    )

predicciones = pd.DataFrame(predicciones)
predicciones.to_csv(
    CARPETA_SALIDA / "clasificacion_nearest_centroid_leave_one_out.csv",
    index=False,
)

accuracy = predicciones["acierto"].mean()
accuracy_df = pd.DataFrame(
    [{"metrica": "accuracy_nearest_centroid_leave_one_out", "valor": accuracy}]
)
accuracy_df.to_csv(CARPETA_SALIDA / "clasificacion_accuracy.csv", index=False)

matriz_confusion = confusion_matrix(
    predicciones["grupo_real"],
    predicciones["grupo_predicho"],
    labels=INIT_ORDER,
)
matriz_confusion = pd.DataFrame(
    matriz_confusion,
    index=[f"real: {g}" for g in INIT_ORDER],
    columns=[f"pred: {g}" for g in INIT_ORDER],
)
matriz_confusion.to_csv(CARPETA_SALIDA / "matriz_confusion_nearest_centroid.csv")

print(f"\nGuardados resultados en: {CARPETA_SALIDA}")
print("\nVarianza PCA:")
print(pca_varianza.round(4).to_string(index=False))
print("\nDistancias entre centroides y separation ratio:")
print(
    distancias_entre_centroides[
        ["grupo_1", "grupo_2", "distancia_centroides", "separation_ratio"]
    ]
    .round(4)
    .to_string(index=False)
)
print(f"\nSilhouette global: {silhouette_global:.4f}")
print("\nSilhouette por grupo:")
print(silhouette_resumen.round(4).to_string(index=False))
print(f"\nAccuracy nearest centroid leave-one-out: {accuracy:.4f}")
print("\nMatriz de confusion:")
print(matriz_confusion.to_string())
