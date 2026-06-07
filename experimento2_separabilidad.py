from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


CARPETA_ENTRADA = Path("PreliminaryExperiments") / "experimento1_repetitividad"
CARPETA_SALIDA = Path("PreliminaryExperiments") / "experimento2_separabilidad"
RUTA_FEATURES = CARPETA_ENTRADA / "features_por_simulacion.csv"

N_COMPONENTES_PCA = 10


CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

datos = pd.read_csv(RUTA_FEATURES)
datos["grupo"] = (
    "lambda="
    + datos["lambda_cc"].map(lambda x: f"{x:g}")
    + ", gamma="
    + datos["gamma"].map(lambda x: f"{x:g}")
)
datos = datos.sort_values(["lambda_cc", "gamma", "replica"]).reset_index(drop=True)

columnas_metadata = ["sim_id", "replica", "init_id", "grupo", "lambda_cc", "gamma"]
columnas_features = [c for c in datos.columns if c not in columnas_metadata]

scaler = StandardScaler()
X = scaler.fit_transform(datos[columnas_features].to_numpy(dtype=float))
y = datos["grupo"].to_numpy()

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
        sorted(datos["grupo"].unique()),
        ["tab:blue", "tab:orange", "tab:green", "tab:red"],
    )
}
for grupo, datos_grupo in pca_scores.groupby("grupo", sort=True):
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

for grupo, datos_grupo in pca_scores.groupby("grupo", sort=True):
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
ax.set_title("PCA separabilidad por grupo")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)")
ax.legend(title="Grupo", fontsize=8)
fig.savefig(CARPETA_SALIDA / "pca_2d_grupos.png", dpi=180, bbox_inches="tight")
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
    datos_escalados.groupby(["lambda_cc", "gamma", "grupo"], as_index=False)[
        columnas_features
    ]
    .mean()
    .sort_values(["lambda_cc", "gamma"])
    .reset_index(drop=True)
)
centroides.to_csv(CARPETA_SALIDA / "centroides_escalados_por_grupo.csv", index=False)

distancias_centroide = []
for (lam, gam, grupo), datos_grupo in datos_escalados.groupby(
    ["lambda_cc", "gamma", "grupo"], sort=True
):
    matriz = datos_grupo[columnas_features].to_numpy(dtype=float)
    centroide = matriz.mean(axis=0)
    distancias = np.linalg.norm(matriz - centroide, axis=1)

    distancias_centroide.append(
        {
            "lambda_cc": lam,
            "gamma": gam,
            "grupo": grupo,
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

        grupo_i = centroides.loc[i, "grupo"]
        grupo_j = centroides.loc[j, "grupo"]
        within_i = distancias_centroide.loc[
            distancias_centroide["grupo"] == grupo_i, "within_mean"
        ].iloc[0]
        within_j = distancias_centroide.loc[
            distancias_centroide["grupo"] == grupo_j, "within_mean"
        ].iloc[0]

        distancias_entre_centroides.append(
            {
                "lambda_cc_1": centroides.loc[i, "lambda_cc"],
                "gamma_1": centroides.loc[i, "gamma"],
                "grupo_1": grupo_i,
                "lambda_cc_2": centroides.loc[j, "lambda_cc"],
                "gamma_2": centroides.loc[j, "gamma"],
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
    silhouette_detalle.groupby(["lambda_cc", "gamma", "grupo"])["silhouette"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .reset_index()
)
silhouette_resumen.to_csv(CARPETA_SALIDA / "silhouette_por_grupo.csv", index=False)

predicciones = []
for i in range(len(datos_escalados)):
    grupo_real = datos_escalados.loc[i, "grupo"]
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i)

    centroides_loo = []
    for grupo in sorted(np.unique(y_train)):
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

grupos_ordenados = sorted(datos["grupo"].unique())
matriz_confusion = confusion_matrix(
    predicciones["grupo_real"],
    predicciones["grupo_predicho"],
    labels=grupos_ordenados,
)
matriz_confusion = pd.DataFrame(
    matriz_confusion,
    index=[f"real: {g}" for g in grupos_ordenados],
    columns=[f"pred: {g}" for g in grupos_ordenados],
)
matriz_confusion.to_csv(CARPETA_SALIDA / "matriz_confusion_nearest_centroid.csv")

print(f"Guardados resultados en: {CARPETA_SALIDA}")
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
