import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataframes_v4 import leer_dataframes_barrido
from metricas_resultados import calcular_metricas_parametros, plot_reales_vs_predichos_bivariado
from Preprocesado import extraer_features


CARPETA = "dataframesBarrido"
OUTPUT_DIR = Path("outputs") / f"red_neuronal_classfication_{CARPETA}"

TEST_SIZE = 0.2
SEED = 0
EPOCHS = 400
BATCH_SIZE = 64
LR = 7e-4
WEIGHT_DECAY = 1e-4
HIDDEN_DIMS = [32,16]
EXCLUIR_BORDE = True
N_CENTER_CELLS = 196
N_ELLIPSES_PLOT = 15
ELLIPSE_PLOT_LEVEL = 0.90
EPS_SCALE = 1e-4

META_COLS = {"sim_id", "replica", "init_id", "grupo_id", "carpeta"}
TARGET_COLS = {"lambda_cc", "gamma"}


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[10]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.net(x)
        mu = raw[:, :2]
        raw_l11 = raw[:, 2]
        l21 = raw[:, 3]
        raw_l22 = raw[:, 4]
        l11 = torch.nn.functional.softplus(raw_l11) + EPS_SCALE
        l22 = torch.nn.functional.softplus(raw_l22) + EPS_SCALE
        return mu, l11, l21, l22


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_test_split_indices(n, test_size=0.2, seed=0):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = max(1, int(round(n * test_size)))
    test_idx = np.sort(indices[:n_test])
    train_idx = np.sort(indices[n_test:])
    if len(train_idx) == 0:
        raise ValueError("No quedan muestras para train. Reduce test_size.")
    return train_idx, test_idx


def normalizar_train_test(X_train, X_test, y_train, y_test):
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0] = 1.0

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std == 0] = 1.0

    stats = {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }

    X_train_n = (X_train - stats["x_mean"]) / stats["x_std"]
    X_test_n = (X_test - stats["x_mean"]) / stats["x_std"]
    y_train_n = (y_train - stats["y_mean"]) / stats["y_std"]
    y_test_n = (y_test - stats["y_mean"]) / stats["y_std"]
    return X_train_n, X_test_n, y_train_n, y_test_n, stats


def gaussian_nll_from_cholesky(y, mu, l11, l21, l22):
    diff0 = y[:, 0] - mu[:, 0]
    diff1 = y[:, 1] - mu[:, 1]
    z0 = diff0 / l11
    z1 = (diff1 - l21 * z0) / l22
    mahalanobis_d2 = z0**2 + z1**2
    logdet = 2.0 * torch.log(l11) + 2.0 * torch.log(l22)
    return 0.5 * (mahalanobis_d2 + logdet + 2.0 * math.log(2.0 * math.pi))


def covariance_from_cholesky_np(l11, l21, l22):
    cov = np.empty((len(l11), 2, 2), dtype=np.float64)
    cov[:, 0, 0] = l11**2
    cov[:, 0, 1] = l11 * l21
    cov[:, 1, 0] = cov[:, 0, 1]
    cov[:, 1, 1] = l21**2 + l22**2
    return cov


def transform_predictions_to_original(mu_n, cov_n, stats):
    y_mean = stats["y_mean"].astype(np.float64)
    y_std = stats["y_std"].astype(np.float64)

    mu = mu_n * y_std + y_mean
    cov = cov_n.copy()
    cov[:, 0, 0] *= y_std[0] ** 2
    cov[:, 0, 1] *= y_std[0] * y_std[1]
    cov[:, 1, 0] *= y_std[0] * y_std[1]
    cov[:, 1, 1] *= y_std[1] ** 2
    return mu, cov


def chi2_quantile_df2(prob):
    return -2.0 * math.log(max(1.0 - float(prob), 1e-12))


def gaussian_nll_np(y, mu, cov):
    diff = y - mu
    sign, logdet = np.linalg.slogdet(cov)
    if not np.all(sign > 0):
        raise ValueError("Covarianza no definida positiva en NLL.")
    inv_cov = np.linalg.inv(cov)
    d2 = np.einsum("ni,nij,nj->n", diff, inv_cov, diff)
    nll = 0.5 * (d2 + logdet + 2.0 * np.log(2.0 * np.pi))
    return nll, d2


def ellipse_area(cov, prob):
    q = chi2_quantile_df2(prob)
    det = np.linalg.det(cov)
    det = np.maximum(det, 0.0)
    return math.pi * q * np.sqrt(det)


def coverage_from_d2(mahalanobis_d2, prob):
    return float(np.mean(mahalanobis_d2 <= chi2_quantile_df2(prob)))


def cargar_dataset(carpeta, excluir_borde=True, n_center_cells=196):
    dfs = leer_dataframes_barrido(
        carpeta=carpeta,
        cargar_todos=True,
        status="ok",
    )
    indice = leer_dataframes_barrido(
        carpeta=carpeta,
        status="ok",
        devolver_indice=True,
    )
    indice = indice.set_index("sim_id", drop=False)

    filas = []
    for sim_id, df in sorted(dfs.items(), key=lambda item: str(item[0])):
        Xy_i = extraer_features(
            {str(sim_id): df},
            excluir_borde=excluir_borde,
            n_center_cells=n_center_cells,
        )
        if len(Xy_i) == 0:
            continue

        fila = Xy_i.iloc[0].to_dict()
        fila["sim_id"] = str(sim_id)
        fila["carpeta"] = str(carpeta)
        if str(sim_id) in indice.index:
            meta = indice.loc[str(sim_id)]
            fila["replica"] = int(meta["replica"]) if pd.notnull(meta["replica"]) else np.nan
            fila["init_id"] = meta["init_id"]
            fila["grupo_id"] = meta["grupo_id"]
        else:
            fila["replica"] = np.nan
            fila["init_id"] = pd.NA
            fila["grupo_id"] = pd.NA
        filas.append(fila)

    if not filas:
        raise ValueError(f"No hay simulaciones validas en {carpeta}.")

    Xy = pd.DataFrame(filas).fillna(0.0)
    Xy = Xy.sort_values(["lambda_cc", "gamma", "sim_id"]).reset_index(drop=True)
    return Xy


def preparar_datos(carpeta, test_size, seed, excluir_borde=True, n_center_cells=196):
    Xy = cargar_dataset(
        carpeta=carpeta,
        excluir_borde=excluir_borde,
        n_center_cells=n_center_cells,
    )
    if len(Xy) < 2:
        raise ValueError("Hacen falta al menos 2 simulaciones validas.")

    pair_counts = Xy.groupby(["lambda_cc", "gamma"]).size()
    n_repeated_pairs = int((pair_counts > 1).sum())
    if n_repeated_pairs > 0:
        print(
            "WARNING: hay pares (lambda_cc, gamma) repetidos. "
            "Este script asume muestras independientes sin replicas."
        )

    feature_cols = [
        c
        for c in Xy.columns
        if c not in TARGET_COLS and c not in META_COLS
    ]
    X = Xy[feature_cols].to_numpy(dtype=np.float32)
    y = Xy[["lambda_cc", "gamma"]].to_numpy(dtype=np.float32)

    train_idx, test_idx = train_test_split_indices(
        len(Xy),
        test_size=test_size,
        seed=seed,
    )

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    X_train_n, X_test_n, y_train_n, y_test_n, stats = normalizar_train_test(
        X_train,
        X_test,
        y_train,
        y_test,
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
    X_test_n,
    y_test_n,
    epochs=1000,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    seed=0,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train_n, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_n, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_n, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_n, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=min(batch_size, len(X_train_t)),
        shuffle=True,
    )

    model = GaussianMLP(X_train_t.shape[1], hidden_dims=HIDDEN_DIMS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            mu, l11, l21, l22 = model(xb)
            loss = gaussian_nll_from_cholesky(yb, mu, l11, l21, l22).mean()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(xb)

        train_loss = train_loss_sum / len(X_train_t)

        model.eval()
        with torch.no_grad():
            mu_test, l11_test, l21_test, l22_test = model(X_test_t.to(device))
            test_loss = gaussian_nll_from_cholesky(
                y_test_t.to(device),
                mu_test,
                l11_test,
                l21_test,
                l22_test,
            ).mean().item()

        history.append(
            {
                "epoch": epoch,
                "train_nll_normalized": train_loss,
                "test_nll_normalized": test_loss,
            }
        )

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:4d} "
                f"train_nll_norm={train_loss:.6f} "
                f"test_nll_norm={test_loss:.6f}"
            )

    return model, pd.DataFrame(history), device


def predecir(model, device, X_n):
    X_t = torch.tensor(X_n, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        mu_t, l11_t, l21_t, l22_t = model(X_t)
    mu_n = mu_t.cpu().numpy().astype(np.float64)
    l11 = l11_t.cpu().numpy().astype(np.float64)
    l21 = l21_t.cpu().numpy().astype(np.float64)
    l22 = l22_t.cpu().numpy().astype(np.float64)
    cov_n = covariance_from_cholesky_np(l11, l21, l22)
    return mu_n, cov_n


def evaluar_predicciones(y_true, mu_pred, cov_pred):
    metricas_reg = calcular_metricas_parametros(y_true, mu_pred)
    nll, mahalanobis_d2 = gaussian_nll_np(y_true, mu_pred, cov_pred)

    metricas = dict(metricas_reg)
    metricas.update(
        {
            "nll_media": float(np.mean(nll)),
            "mean_mahalanobis_d2": float(np.mean(mahalanobis_d2)),
            "coverage_50": coverage_from_d2(mahalanobis_d2, 0.50),
            "coverage_68": coverage_from_d2(mahalanobis_d2, 0.68),
            "coverage_90": coverage_from_d2(mahalanobis_d2, 0.90),
            "coverage_95": coverage_from_d2(mahalanobis_d2, 0.95),
            "mean_ellipse_area_50": float(np.mean(ellipse_area(cov_pred, 0.50))),
            "mean_ellipse_area_68": float(np.mean(ellipse_area(cov_pred, 0.68))),
            "mean_ellipse_area_90": float(np.mean(ellipse_area(cov_pred, 0.90))),
            "mean_ellipse_area_95": float(np.mean(ellipse_area(cov_pred, 0.95))),
        }
    )
    return metricas, nll, mahalanobis_d2


def construir_df_predicciones(Xy, indices, y_true, mu_pred, cov_pred, nll, mahalanobis_d2):
    filas = []
    area_65 = ellipse_area(cov_pred, 0.65)
    area_90 = ellipse_area(cov_pred, 0.90)

    for i, idx_global in enumerate(indices):
        fila_original = Xy.iloc[int(idx_global)]
        cov_i = cov_pred[i]
        filas.append(
            {
                "idx": int(idx_global),
                "sim_id": fila_original["sim_id"],
                "lambda_real": float(y_true[i, 0]),
                "gamma_real": float(y_true[i, 1]),
                "lambda_pred": float(mu_pred[i, 0]),
                "gamma_pred": float(mu_pred[i, 1]),
                "lambda_abs_err": float(abs(mu_pred[i, 0] - y_true[i, 0])),
                "gamma_abs_err": float(abs(mu_pred[i, 1] - y_true[i, 1])),
                "cov_lambda_lambda": float(cov_i[0, 0]),
                "cov_lambda_gamma": float(cov_i[0, 1]),
                "cov_gamma_gamma": float(cov_i[1, 1]),
                "mahalanobis_d2": float(mahalanobis_d2[i]),
                "nll": float(nll[i]),
                "ellipse_area_65": float(area_65[i]),
                "ellipse_area_90": float(area_90[i]),
            }
        )
    return pd.DataFrame(filas)


def plot_losses(history, output_png):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_nll_normalized"], label="Train NLL norm")
    ax.plot(history["epoch"], history["test_nll_normalized"], label="Test NLL norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL normalizada")
    ax.set_title("Train y test NLL por epoca")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close(fig)


def imprimir_metricas(nombre, metricas):
    print(f"\nMetricas {nombre}:")
    for k, v in metricas.items():
        print(f"{k}: {v:.8f}")


def main():
    datos = preparar_datos(
        carpeta=CARPETA,
        test_size=TEST_SIZE,
        seed=SEED,
        excluir_borde=EXCLUIR_BORDE,
        n_center_cells=N_CENTER_CELLS,
    )

    print(f"carpeta={CARPETA}")
    print(f"n_simulaciones_validas={len(datos['Xy'])}")
    print(f"n_features={len(datos['feature_cols'])}")
    print(f"train={len(datos['train_idx'])} test={len(datos['test_idx'])}")

    model, history, device = entrenar_modelo(
        datos["X_train_n"],
        datos["y_train_n"],
        datos["X_test_n"],
        datos["y_test_n"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
    )

    mu_train_n, cov_train_n = predecir(model, device, datos["X_train_n"])
    mu_test_n, cov_test_n = predecir(model, device, datos["X_test_n"])

    mu_train, cov_train = transform_predictions_to_original(
        mu_train_n,
        cov_train_n,
        datos["stats"],
    )
    mu_test, cov_test = transform_predictions_to_original(
        mu_test_n,
        cov_test_n,
        datos["stats"],
    )

    metricas_train, nll_train, d2_train = evaluar_predicciones(
        datos["y_train"],
        mu_train,
        cov_train,
    )
    metricas_test, nll_test, d2_test = evaluar_predicciones(
        datos["y_test"],
        mu_test,
        cov_test,
    )

    imprimir_metricas("train", metricas_train)
    imprimir_metricas("test", metricas_test)

    df_train = construir_df_predicciones(
        datos["Xy"],
        datos["train_idx"],
        datos["y_train"],
        mu_train,
        cov_train,
        nll_train,
        d2_train,
    )
    df_test = construir_df_predicciones(
        datos["Xy"],
        datos["test_idx"],
        datos["y_test"],
        mu_test,
        cov_test,
        nll_test,
        d2_test,
    )

    print("\nPrimeras predicciones train:")
    #print(df_train.head(12).to_string(index=False))
    print("\nPrimeras predicciones test:")
    #print(df_test.head(12).to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history_path = OUTPUT_DIR / "training_history.csv"
    train_pred_path = OUTPUT_DIR / "predicciones_train.csv"
    test_pred_path = OUTPUT_DIR / "predicciones_test.csv"
    train_metrics_path = OUTPUT_DIR / "metricas_train.csv"
    test_metrics_path = OUTPUT_DIR / "metricas_test.csv"
    model_path = OUTPUT_DIR / "modelo.pt"
    plot_path = OUTPUT_DIR / "mapa_reales_vs_pred_con_elipses.png"
    loss_plot_path = OUTPUT_DIR / "loss_train_test.png"

    history.to_csv(history_path, index=False)
    df_train.to_csv(train_pred_path, index=False)
    df_test.to_csv(test_pred_path, index=False)
    pd.DataFrame([metricas_train]).to_csv(train_metrics_path, index=False)
    pd.DataFrame([metricas_test]).to_csv(test_metrics_path, index=False)
    plot_losses(history, loss_plot_path)
    plot_reales_vs_predichos_bivariado(
        df_test,
        plot_path,
        modelo="Maximum Likelihood MLP",
        cov_pred=cov_test,
        n_ellipses=N_ELLIPSES_PLOT,
        ellipse_prob=ELLIPSE_PLOT_LEVEL,
        real_size=55,
        pred_size=55,
        line_width=1.0,
        line_alpha=0.75,
        real_linewidth=1.6,
        pred_linewidth=0.5,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": datos["feature_cols"],
            "stats": datos["stats"],
            "config": {
                "carpeta": CARPETA,
                "test_size": TEST_SIZE,
                "seed": SEED,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "hidden_dims": HIDDEN_DIMS,
                "excluir_borde": EXCLUIR_BORDE,
                "n_center_cells": N_CENTER_CELLS,
                "ellipse_plot_level": ELLIPSE_PLOT_LEVEL,
            },
        },
        model_path,
    )

    print(f"\nGuardado historial en: {history_path}")
    print(f"Guardadas predicciones train en: {train_pred_path}")
    print(f"Guardadas predicciones test en: {test_pred_path}")
    print(f"Guardadas metricas train en: {train_metrics_path}")
    print(f"Guardadas metricas test en: {test_metrics_path}")
    print(f"Guardada grafica train/test loss en: {loss_plot_path}")
    print(f"Guardado mapa real/predicho con elipses en: {plot_path}")
    print(f"Guardado modelo en: {model_path}")


if __name__ == "__main__":
    main()
