from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from dataframes_v4 import leer_dataframes_barrido
from metricas_resultados import calcular_metricas_parametros, plot_reales_vs_predichos_bivariado
from Preprocesado import extraer_features


def train_test_split_indices(n, test_size=0.2, seed=0):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = max(1, int(round(n * test_size)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    if len(train_idx) == 0:
        raise ValueError("No quedan muestras para train. Reduce test_size o usa mas simulaciones.")
    return train_idx, test_idx


def normalizar_train_test(X_train, X_test, y_train, y_test):
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0] = 1.0

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std == 0] = 1.0

    X_train_n = (X_train - x_mean) / x_std
    X_test_n = (X_test - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return X_train_n, X_test_n, y_train_n, y_test_n, stats


def desnormalizar_y(y_pred_n, stats):
    return y_pred_n * stats["y_std"] + stats["y_mean"]


def plot_feature_importances(model, feature_cols, output_png, top_n=20):
    importancias = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    top = importancias.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importancia")
    ax.set_title(f"Top {len(top)} features - Random Forest")
    ax.grid(True, axis="x", alpha=0.25)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close(fig)
    return importancias


def normalizar_lista_carpetas(carpetas):
    if isinstance(carpetas, (str, Path)):
        return [str(carpetas)]
    return [str(carpeta) for carpeta in carpetas]


def tag_carpetas(carpetas):
    return "_".join(Path(carpeta).name for carpeta in normalizar_lista_carpetas(carpetas))


def cargar_simulaciones_carpetas(carpetas):
    simulaciones = {}
    for carpeta in normalizar_lista_carpetas(carpetas):
        simulaciones_carpeta = leer_dataframes_barrido(
            carpeta=carpeta,
            cargar_todos=True,
            status="ok",
        )
        for sim_id, df in simulaciones_carpeta.items():
            simulaciones[f"{carpeta}::{sim_id}"] = df
    return simulaciones


def preparar_datos(carpetas, test_size, seed, excluir_borde=True, n_center_cells=196):
    simulaciones = cargar_simulaciones_carpetas(carpetas)

    Xy = extraer_features(
        simulaciones,
        excluir_borde=excluir_borde,
        n_center_cells=n_center_cells,
    )

    if len(Xy) < 2:
        raise ValueError("Hacen falta al menos 2 simulaciones validas para train/test.")

    feature_cols = [c for c in Xy.columns if c not in ("lambda_cc", "gamma")]
    X = Xy[feature_cols].to_numpy(dtype=np.float32)
    y = Xy[["lambda_cc", "gamma"]].to_numpy(dtype=np.float32)

    train_idx, test_idx = train_test_split_indices(len(Xy), test_size=test_size, seed=seed)

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    X_train_n, X_test_n, y_train_n, y_test_n, stats = normalizar_train_test(
        X_train, X_test, y_train, y_test
    )

    return {
        "Xy": Xy,
        "feature_cols": feature_cols,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train_n": X_train_n,
        "X_test_n": X_test_n,
        "y_train_n": y_train_n,
        "y_test_n": y_test_n,
        "y_train": y_train,
        "y_test": y_test,
        "stats": stats,
    }


def entrenar_modelo(
    X_train_n,
    y_train_n,
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    seed=0,
    n_jobs=-1,
):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(X_train_n, y_train_n)
    return model


def evaluar_modelo(model, X_test_n, y_test, stats):
    pred_n = model.predict(X_test_n)
    pred = desnormalizar_y(pred_n, stats)
    resumen = calcular_metricas_parametros(y_test, pred)
    return pred, resumen


def main():
    carpetas = ["dataframesBarrido"]
    carpetas_tag = tag_carpetas(carpetas)
    test_size = 0.2
    seed = 0
    incluir_borde = False
    n_center_cells = 196

    n_estimators = 500
    max_depth = None
    min_samples_leaf = 1
    n_jobs = -1

    datos = preparar_datos(
        carpetas=carpetas,
        test_size=test_size,
        seed=seed,
        excluir_borde=not incluir_borde,
        n_center_cells=n_center_cells,
    )

    print(f"carpetas={carpetas}")
    print(f"n_simulaciones_validas={len(datos['Xy'])}")
    print(f"n_features={len(datos['feature_cols'])}")
    print(f"train={len(datos['train_idx'])} test={len(datos['test_idx'])}")
    print(
        "random_forest="
        f"n_estimators={n_estimators}, "
        f"max_depth={max_depth}, "
        f"min_samples_leaf={min_samples_leaf}"
    )

    model = entrenar_modelo(
        datos["X_train_n"],
        datos["y_train_n"],
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
        n_jobs=n_jobs,
    )

    pred, resumen = evaluar_modelo(
        model,
        datos["X_test_n"],
        datos["y_test"],
        datos["stats"],
    )

    print("\nMetricas test:")
    for k, v in resumen.items():
        print(f"{k}: {v:.8f}")

    filas = []
    for idx_local, idx_global in enumerate(datos["test_idx"]):
        filas.append(
            {
                "idx_test": int(idx_global),
                "lambda_real": float(datos["y_test"][idx_local, 0]),
                "gamma_real": float(datos["y_test"][idx_local, 1]),
                "lambda_pred": float(pred[idx_local, 0]),
                "gamma_pred": float(pred[idx_local, 1]),
                "lambda_abs_err": float(abs(pred[idx_local, 0] - datos["y_test"][idx_local, 0])),
                "gamma_abs_err": float(abs(pred[idx_local, 1] - datos["y_test"][idx_local, 1])),
            }
        )

    df_pred = pd.DataFrame(filas)
    print("\nPredicciones test:")
    #print(df_pred.to_string(index=False))

    out_dir = Path("outputs") / f"random_forest_{carpetas_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "predicciones_test.csv"
    metricas_path = out_dir / "metricas_test.csv"
    importancias_path = out_dir / "feature_importances.csv"
    model_path = out_dir / "modelo_random_forest.joblib"
    plot_path = out_dir / "mapa_reales_vs_pred.png"
    importancias_plot_path = out_dir / "feature_importances.png"

    df_pred.to_csv(pred_path, index=False)
    pd.DataFrame([resumen]).to_csv(metricas_path, index=False)
    plot_reales_vs_predichos_bivariado(df_pred, plot_path, modelo="Random Forest")
    importancias = plot_feature_importances(
        model,
        datos["feature_cols"],
        importancias_plot_path,
    )
    importancias.to_csv(importancias_path, index=False)
    joblib.dump(
        {
            "model": model,
            "feature_cols": datos["feature_cols"],
            "stats": datos["stats"],
            "config": {
                "carpetas": carpetas,
                "test_size": test_size,
                "seed": seed,
                "n_center_cells": n_center_cells,
                "incluir_borde": incluir_borde,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "n_jobs": n_jobs,
            },
        },
        model_path,
    )

    print(f"\nGuardadas predicciones en: {pred_path}")
    print(f"Guardadas metricas en: {metricas_path}")
    print(f"Guardadas importancias en: {importancias_path}")
    print(f"Guardado mapa real/predicho en: {plot_path}")
    print(f"Guardada grafica de importancias en: {importancias_plot_path}")
    print(f"Guardado modelo en: {model_path}")


if __name__ == "__main__":
    main()
