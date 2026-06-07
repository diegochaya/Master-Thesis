from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataframes_v4 import leer_dataframes_barrido
from metricas_resultados import calcular_metricas_parametros, plot_reales_vs_predichos_bivariado
from Preprocesado import extraer_features


class MLPRegresor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_test_split_indices(n, test_size=0.2, seed=0):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = max(1, int(round(n * test_size)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    if len(train_idx) == 0:
        raise ValueError("No quedan muestras para train. Reduce test_size o usa más simulaciones.")
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


def plot_losses(history, output_png):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["test_loss"], label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train y test loss por epoca")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()
    plt.close(fig)


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
        raise ValueError("Hacen falta al menos 2 simulaciones válidas para train/test.")

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
    X_test_n,
    y_test_n,
    epochs=300,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-4,
    seed=0,
):
    torch.manual_seed(seed)

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

    model = MLPRegresor(input_dim=X_train_t.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(xb)

        train_loss = train_loss_sum / len(X_train_t)

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t.to(device))
            test_loss = criterion(test_pred, y_test_t.to(device)).item()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
            }
        )

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:4d} "
                f"train_loss={train_loss:.6f} "
                f"test_loss={test_loss:.6f}"
            )

    return model, pd.DataFrame(history), device


def evaluar_modelo(model, device, X_test_n, y_test, stats):
    X_test_t = torch.tensor(X_test_n, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred_n = model(X_test_t).cpu().numpy()

    pred = desnormalizar_y(pred_n, stats)
    resumen = calcular_metricas_parametros(y_test, pred)
    return pred, resumen


def main():
    carpetas = ["dataframesBarrido"]
    carpetas_tag = tag_carpetas(carpetas)
    test_size = 0.2
    epochs = 500
    batch_size = 100
    lr = 1e-3
    weight_decay = 1e-4
    seed = 0
    incluir_borde = False
    n_center_cells = 196

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

    model, history, device = entrenar_modelo(
        datos["X_train_n"],
        datos["y_train_n"],
        datos["X_test_n"],
        datos["y_test_n"],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
    )

    pred, resumen = evaluar_modelo(
        model,
        device,
        datos["X_test_n"],
        datos["y_test"],
        datos["stats"],
    )

    print("\nMetricas test:")
    for k, v in resumen.items():
        print(f"{k}: {v:.8f}")

    filas = []
    for idx_local, idx_global in enumerate(datos["test_idx"]):
        fila_original = datos["Xy"].iloc[idx_global]
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

    out_dir = Path("outputs") / f"red_neuronal_{carpetas_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "training_history.csv"
    pred_path = out_dir / "predicciones_test.csv"
    model_path = out_dir / "modelo_parametros.pt"
    plot_path = out_dir / "mapa_reales_vs_pred.png"
    loss_plot_path = out_dir / "loss_train_test.png"

    history.to_csv(history_path, index=False)
    df_pred.to_csv(pred_path, index=False)
    plot_losses(history, loss_plot_path)
    plot_reales_vs_predichos_bivariado(df_pred, plot_path, modelo="Red neuronal")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": datos["feature_cols"],
            "stats": datos["stats"],
            "config": {
                "carpetas": carpetas,
                "test_size": test_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "seed": seed,
                "n_center_cells": n_center_cells,
                "incluir_borde": incluir_borde,
            },
        },
        model_path,
    )

    print(f"\nGuardado historial en: {history_path}")
    print(f"Guardadas predicciones en: {pred_path}")
    print(f"Guardada grafica train/test loss en: {loss_plot_path}")
    print(f"Guardado mapa real/predicho en: {plot_path}")
    print(f"Guardado modelo en: {model_path}")


if __name__ == "__main__":
    main()
