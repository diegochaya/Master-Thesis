from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Preprocesado import extraer_features


CARPETA = "dataframesComprobaciones3"
RUTA_CARPETA = Path("dataframesCarpetas") / CARPETA
RUTA_INDICE = RUTA_CARPETA / "indice_simulaciones.csv"
CARPETA_SALIDA = Path("PreliminaryExperiments") / "experimento1_repetitividad"

LAMBDA_VALUES = [-0.1, 0.0]
GAMMA_VALUES = [0.05, 0.08]
REPLICAS_ESPERADAS = 30

EXCLUIR_BORDE = True
N_CENTER_CELLS = 196


CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

indice = pd.read_csv(RUTA_INDICE)
indice = indice.loc[indice["status"].astype(str) == "ok"].copy()
indice = indice.loc[indice["lambda_cc"].isin(LAMBDA_VALUES)].copy()
indice = indice.loc[indice["gamma"].isin(GAMMA_VALUES)].copy()
indice = indice.sort_values(["lambda_cc", "gamma", "replica"]).reset_index(drop=True)

conteo = indice.groupby(["lambda_cc", "gamma"]).size()
print("Simulaciones ok por grupo:")
print(conteo.to_string())

if len(indice) != len(LAMBDA_VALUES) * len(GAMMA_VALUES) * REPLICAS_ESPERADAS:
    print(
        "\nWARNING: el numero de simulaciones ok no coincide con "
        f"{len(LAMBDA_VALUES) * len(GAMMA_VALUES) * REPLICAS_ESPERADAS}."
    )

filas_features = []

for _, fila_indice in indice.iterrows():
    sim_id = str(fila_indice["sim_id"])
    ruta_df = Path(fila_indice["filepath"])
    df_final = pd.read_pickle(ruta_df)

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
    fila_features["grupo"] = (
        f"lambda={float(fila_indice['lambda_cc']):g}, "
        f"gamma={float(fila_indice['gamma']):g}"
    )
    filas_features.append(fila_features)

datos = pd.DataFrame(filas_features)
datos = datos.sort_values(["lambda_cc", "gamma", "replica"]).reset_index(drop=True)

columnas_metadata = ["sim_id", "replica", "init_id", "grupo", "lambda_cc", "gamma"]
columnas_features = [c for c in datos.columns if c not in columnas_metadata]
datos = datos[columnas_metadata + columnas_features]
datos.to_csv(CARPETA_SALIDA / "features_por_simulacion.csv", index=False)

scaler = StandardScaler()
features_escaladas = scaler.fit_transform(datos[columnas_features].to_numpy(dtype=float))
datos_escalados = datos[columnas_metadata].copy()
for i, columna in enumerate(columnas_features):
    datos_escalados[columna] = features_escaladas[:, i]

centroides = (
    datos.groupby(["lambda_cc", "gamma", "grupo"], as_index=False)[columnas_features]
    .mean()
    .sort_values(["lambda_cc", "gamma"])
)
centroides.to_csv(CARPETA_SALIDA / "centroides_por_grupo.csv", index=False)

distancias_centroide = []
for (lam, gam, grupo), datos_grupo in datos_escalados.groupby(
    ["lambda_cc", "gamma", "grupo"], sort=True
):
    matriz = datos_grupo[columnas_features].to_numpy(dtype=float)
    centroide = matriz.mean(axis=0)
    distancias = np.linalg.norm(matriz - centroide, axis=1)

    for sim_id, replica, distancia in zip(
        datos_grupo["sim_id"], datos_grupo["replica"], distancias
    ):
        distancias_centroide.append(
            {
                "lambda_cc": lam,
                "gamma": gam,
                "grupo": grupo,
                "sim_id": sim_id,
                "replica": replica,
                "distancia_a_centroide": distancia,
            }
        )

distancias_centroide = pd.DataFrame(distancias_centroide)
distancias_centroide.to_csv(
    CARPETA_SALIDA / "distancias_a_centroide.csv",
    index=False,
)

distancias_pareadas = []
for (lam, gam, grupo), datos_grupo in datos_escalados.groupby(
    ["lambda_cc", "gamma", "grupo"], sort=True
):
    matriz = datos_grupo[columnas_features].to_numpy(dtype=float)
    simulaciones = datos_grupo[["sim_id", "replica"]].reset_index(drop=True)

    for i in range(len(datos_grupo)):
        for j in range(i + 1, len(datos_grupo)):
            distancia = np.linalg.norm(matriz[i] - matriz[j])
            distancias_pareadas.append(
                {
                    "lambda_cc": lam,
                    "gamma": gam,
                    "grupo": grupo,
                    "sim_id_1": simulaciones.loc[i, "sim_id"],
                    "replica_1": int(simulaciones.loc[i, "replica"]),
                    "sim_id_2": simulaciones.loc[j, "sim_id"],
                    "replica_2": int(simulaciones.loc[j, "replica"]),
                    "distancia": distancia,
                }
            )

distancias_pareadas = pd.DataFrame(distancias_pareadas)
distancias_pareadas.to_csv(
    CARPETA_SALIDA / "distancias_pareadas_intra_grupo.csv",
    index=False,
)

resumen_distancias = (
    distancias_centroide.groupby(["lambda_cc", "gamma", "grupo"])[
        "distancia_a_centroide"
    ]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .reset_index()
)
resumen_distancias.to_csv(
    CARPETA_SALIDA / "resumen_distancias_a_centroide_por_grupo.csv",
    index=False,
)

estadisticas_largas = []
for (lam, gam, grupo), datos_grupo in datos.groupby(
    ["lambda_cc", "gamma", "grupo"], sort=True
):
    datos_grupo_escalado = datos_escalados.loc[datos_grupo.index]
    n = len(datos_grupo)

    for columna in columnas_features:
        valores = datos_grupo[columna].to_numpy(dtype=float)
        valores_escalados = datos_grupo_escalado[columna].to_numpy(dtype=float)
        media = valores.mean()
        std = valores.std(ddof=1)
        cv = std / abs(media) if abs(media) > 1e-12 else np.nan
        sem = std / np.sqrt(n)

        estadisticas_largas.append(
            {
                "lambda_cc": lam,
                "gamma": gam,
                "grupo": grupo,
                "feature": columna,
                "n": n,
                "mean": media,
                "std": std,
                "cv_abs": cv,
                "ci95_low": media - 1.96 * sem,
                "ci95_high": media + 1.96 * sem,
                "std_escalada": valores_escalados.std(ddof=1),
            }
        )

estadisticas_largas = pd.DataFrame(estadisticas_largas)
estadisticas_largas.to_csv(
    CARPETA_SALIDA / "estadisticas_features_por_grupo.csv",
    index=False,
)

features_mas_variables = (
    estadisticas_largas.sort_values(
        ["lambda_cc", "gamma", "std_escalada"],
        ascending=[True, True, False],
    )
    .groupby(["lambda_cc", "gamma", "grupo"], as_index=False)
    .head(10)
)
features_mas_variables.to_csv(
    CARPETA_SALIDA / "features_mas_variables_por_grupo.csv",
    index=False,
)

print(f"\nGuardados resultados en: {CARPETA_SALIDA}")
print("\nResumen distancias a centroide:")
print(resumen_distancias.round(4).to_string(index=False))
print("\nTop features mas variables por grupo:")
print(
    features_mas_variables[
        ["grupo", "feature", "std", "cv_abs", "std_escalada"]
    ]
    .round(4)
    .to_string(index=False)
)
