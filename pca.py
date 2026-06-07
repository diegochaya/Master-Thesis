from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from dataframes_v4 import leer_dataframes_barrido
from Preprocesado import extraer_features


CARPETA_DATAFRAMES = "dataframesBarrido"
CARPETA_SALIDA = Path("PreliminaryExperiments")


def cargar_matriz_features(carpeta=CARPETA_DATAFRAMES, excluir_borde=True, n_center_cells=196):
    dfs = leer_dataframes_barrido(
        carpeta=carpeta,
        cargar_todos=True,
        status="ok",
    )
    Xy = extraer_features(
        dfs,
        excluir_borde=excluir_borde,
        n_center_cells=n_center_cells,
    )
    if Xy.empty:
        raise ValueError(f"No hay datos disponibles en {carpeta!r}.")

    feature_cols = [c for c in Xy.columns if c not in ("lambda_cc", "gamma")]
    X = Xy[feature_cols].to_numpy(dtype=float)
    y = Xy[["lambda_cc", "gamma"]].copy()
    return X, y, feature_cols, Xy


def tabla_componentes(pca, feature_cols, n_componentes=5):
    n = min(n_componentes, pca.components_.shape[0])
    componentes = pca.components_[:n]
    tabla = pd.DataFrame(
        componentes.T,
        index=feature_cols,
        columns=[f"PC{i + 1}" for i in range(n)],
    )
    tabla["abs_max"] = tabla.abs().max(axis=1)
    tabla = tabla.sort_values("abs_max", ascending=False).drop(columns="abs_max")
    return tabla


def plot_pca_2d(X_pca, y, output_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    sc1 = axes[0].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y["lambda_cc"],
        cmap="viridis",
        s=28,
        alpha=0.85,
    )
    axes[0].set_title("PCA 2D coloreado por lambda")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.colorbar(sc1, ax=axes[0], label="lambda_cc")

    sc2 = axes[1].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y["gamma"],
        cmap="plasma",
        s=28,
        alpha=0.85,
    )
    axes[1].set_title("PCA 2D coloreado por gamma")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.colorbar(sc2, ax=axes[1], label="gamma")

    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    X, y, feature_cols, _ = cargar_matriz_features()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_componentes = min(5, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X_scaled)

    tabla = tabla_componentes(pca, feature_cols, n_componentes=n_componentes)
    ruta_tabla = CARPETA_SALIDA / "pca_componentes_dataframesReal2.csv"
    tabla.to_csv(ruta_tabla, index=True)

    resumen = pd.DataFrame(
        {
            "componente": [f"PC{i + 1}" for i in range(n_componentes)],
            "varianza_explicada": pca.explained_variance_ratio_,
            "varianza_explicada_acumulada": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    ruta_resumen = CARPETA_SALIDA / "pca_varianza_dataframesReal2.csv"
    resumen.to_csv(ruta_resumen, index=False)

    output_png = CARPETA_SALIDA / "pca_2d_dataframesReal2.png"
    plot_pca_2d(X_pca, y, output_png)

    print("Primeros 5 componentes principales:")
    print(tabla)
    print("\nVarianza explicada:")
    print(resumen)
    print(f"\nGuardado CSV componentes en: {ruta_tabla}")
    print(f"Guardado CSV varianza en: {ruta_resumen}")
    print(f"Guardado PCA 2D en: {output_png}")


if __name__ == "__main__":
    main()
