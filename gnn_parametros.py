from pathlib import Path
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, SAGEConv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_undirected

from dataframes_v4 import leer_dataframes_barrido
from metricas_resultados import calcular_metricas_parametros, plot_reales_vs_predichos_bivariado
from Preprocesado import _elongation_from_vertices, limpiar_df


CARPETAS = ["dataframesBarrido"]
EXCLUIR_BORDE = False
MAX_CENTER_CELLS = 196
MIN_CELLS = 5
TEST_SIZE = 0.20
SEED = 0
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4


def normalizar_lista_carpetas(carpetas):
    if isinstance(carpetas, (str, Path)):
        return [str(carpetas)]
    return [str(carpeta) for carpeta in carpetas]


def tag_carpetas(carpetas):
    return "_".join(Path(carpeta).name for carpeta in normalizar_lista_carpetas(carpetas))


OUT_DIR = Path("outputs") / f"gnn_{tag_carpetas(CARPETAS)}"

NODE_FEATURES = [
    "area",
    "shape_index",
    "elongation",
    "perimeter"
    ]


def parse_neighbors(neighbors):
    if isinstance(neighbors, str):
        neighbors = ast.literal_eval(neighbors)
    return [int(n) for n in neighbors if int(n) >= 0]


def train_test_split_indices(n, test_size=TEST_SIZE, seed=SEED):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = max(1, int(round(n * test_size)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    if len(train_idx) == 0:
        raise ValueError("No quedan muestras para train. Reduce test_size o usa mas simulaciones.")

    return train_idx, test_idx


def construir_edge_index(df_limpio):
    raw_to_local = {int(raw_idx): local_idx for local_idx, raw_idx in enumerate(df_limpio.index)}
    edges = []

    for raw_idx, neighbors in zip(df_limpio.index, df_limpio["neighbors"]):
        source = raw_to_local[int(raw_idx)]
        for raw_neighbor in parse_neighbors(neighbors):
            target = raw_to_local.get(raw_neighbor)
            if target is not None:
                edges.append((source, target))

    if not edges:
        raise ValueError("Grafo sin aristas despues de filtrar celulas.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=len(df_limpio))
    return edge_index


def construir_grafo(sim_id, df, lambda_cc, gamma):
    d = limpiar_df(
        df,
        excluir_borde=EXCLUIR_BORDE,
        n_center_cells=MAX_CENTER_CELLS,
    )

    if len(d) < MIN_CELLS:
        raise ValueError(
            f"{sim_id}: se necesitan al menos {MIN_CELLS} celulas y hay {len(d)}."
        )

    center_x_rel = d["center_x"].to_numpy(dtype=np.float32)
    center_y_rel = d["center_y"].to_numpy(dtype=np.float32)
    center_x_rel = center_x_rel - center_x_rel.mean()
    center_y_rel = center_y_rel - center_y_rel.mean()

    elongation = np.array(
        [_elongation_from_vertices(vertices) for vertices in d["vertices"]],
        dtype=np.float32,
    )

    feature_values = {
        "center_x_rel": center_x_rel,
        "center_y_rel": center_y_rel,
        "area": d["area"].to_numpy(dtype=np.float32),
        "perimeter": d["perimeter"].to_numpy(dtype=np.float32),
        "shape_index": d["shape_index"].to_numpy(dtype=np.float32),
        "elongation": elongation,
        "n_vertices": d["n_vertices"].to_numpy(dtype=np.float32),
        "n_neighbors": d["n_neighbors"].to_numpy(dtype=np.float32),
        "cell_type": d["cell_type"].to_numpy(dtype=np.float32),
    }
    x = np.column_stack([feature_values[name] for name in NODE_FEATURES])
    pos = np.column_stack([center_x_rel, center_y_rel]).astype(np.float32)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=construir_edge_index(d),
        pos=torch.tensor(pos, dtype=torch.float32),
        y=torch.tensor([[lambda_cc, gamma]], dtype=torch.float32),
    )
    data.y_raw = data.y.clone()
    data.sim_id = sim_id
    return data


def cargar_grafos(carpetas=CARPETAS):
    grafos = []
    for carpeta in normalizar_lista_carpetas(carpetas):
        indice = leer_dataframes_barrido(
            carpeta=carpeta,
            devolver_indice=True,
            status="ok",
        )
        objetivos = {
            str(row["sim_id"]): (float(row["lambda_cc"]), float(row["gamma"]))
            for _, row in indice.iterrows()
        }

        dataframes = leer_dataframes_barrido(
            carpeta=carpeta,
            cargar_todos=True,
            status="ok",
        )

        for sim_id, df in sorted(dataframes.items()):
            lambda_cc, gamma = objetivos[str(sim_id)]
            grafos.append(construir_grafo(f"{carpeta}::{sim_id}", df, lambda_cc, gamma))

    return grafos


def comprobar_grafos(grafos):
    for data in grafos:
        if data.num_nodes < MIN_CELLS or data.num_nodes > MAX_CENTER_CELLS:
            raise ValueError(
                f"{data.sim_id}: num_nodes={data.num_nodes}, "
                f"rango permitido=[{MIN_CELLS}, {MAX_CENTER_CELLS}]"
            )
        if not torch.isfinite(data.x).all():
            raise ValueError(f"{data.sim_id}: hay NaN/Inf en x")
        if data.x.shape[1] != len(NODE_FEATURES):
            raise ValueError(
                f"{data.sim_id}: num_features={data.x.shape[1]} "
                f"pero NODE_FEATURES tiene {len(NODE_FEATURES)}"
            )
        if not torch.isfinite(data.y_raw).all():
            raise ValueError(f"{data.sim_id}: hay NaN/Inf en y")
        if data.edge_index.numel() == 0:
            raise ValueError(f"{data.sim_id}: grafo sin aristas")
        if int(data.edge_index.min()) < 0 or int(data.edge_index.max()) >= data.num_nodes:
            raise ValueError(f"{data.sim_id}: edge_index fuera de rango")


def calcular_stats_normalizacion(grafos_train):
    x_train = torch.cat([data.x for data in grafos_train], dim=0)
    y_train = torch.cat([data.y_raw for data in grafos_train], dim=0)

    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0, unbiased=False)
    x_std[x_std == 0] = 1.0

    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0, unbiased=False)
    y_std[y_std == 0] = 1.0

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def aplicar_normalizacion(grafos, stats):
    for data in grafos:
        data.x = (data.x - stats["x_mean"]) / stats["x_std"]
        data.y = (data.y_raw - stats["y_mean"]) / stats["y_std"]


def desnormalizar_y(y_pred_n, stats):
    return y_pred_n * stats["y_std"].cpu().numpy() + stats["y_mean"].cpu().numpy()


class GraphSAGERegresor(nn.Module):
    def __init__(self, input_dim, hidden_dim=25, hidden_dim_2=12, output_dim=2, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        """
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        """
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, output_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        """
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        """
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        pooled = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)],
            dim=1,
        )
        return self.head(pooled)


def evaluar_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            total_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs

    return total_loss / total


def entrenar_modelo(grafos_train, grafos_test):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cpu"
    train_loader = DataLoader(grafos_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(grafos_test, batch_size=BATCH_SIZE, shuffle=False)

    model = GraphSAGERegresor(input_dim=len(NODE_FEATURES)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_graphs = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch.num_graphs
            train_graphs += batch.num_graphs

        train_loss = train_loss_sum / train_graphs
        test_loss = evaluar_loss(model, test_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
            }
        )

        if epoch == 1 or epoch % 25 == 0 or epoch == EPOCHS:
            print(
                f"epoch={epoch:4d} "
                f"train_loss={train_loss:.6f} "
                f"test_loss={test_loss:.6f}"
            )

    return model, pd.DataFrame(history), device


def evaluar_modelo(model, grafos_test, stats, device):
    loader = DataLoader(grafos_test, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    preds_n = []
    reales = []
    sim_ids = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).cpu().numpy()
            preds_n.append(pred)
            reales.append(batch.y_raw.cpu().numpy())
            sim_ids.extend(batch.sim_id)

    pred_n = np.vstack(preds_n)
    y_real = np.vstack(reales)
    y_pred = desnormalizar_y(pred_n, stats)
    errores = y_pred - y_real
    resumen = calcular_metricas_parametros(y_real, y_pred)

    filas = []
    for i, sim_id in enumerate(sim_ids):
        filas.append(
            {
                "sim_id": str(sim_id),
                "lambda_real": float(y_real[i, 0]),
                "gamma_real": float(y_real[i, 1]),
                "lambda_pred": float(y_pred[i, 0]),
                "gamma_pred": float(y_pred[i, 1]),
                "lambda_abs_err": float(abs(errores[i, 0])),
                "gamma_abs_err": float(abs(errores[i, 1])),
            }
        )

    return pd.DataFrame(filas), resumen


def plot_losses(history, output_png):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["test_loss"], label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GNN train y test loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close(fig)


def main():
    grafos = cargar_grafos(CARPETAS)
    comprobar_grafos(grafos)

    train_idx, test_idx = train_test_split_indices(len(grafos))
    grafos_train = [grafos[i] for i in train_idx]
    grafos_test = [grafos[i] for i in test_idx]

    stats = calcular_stats_normalizacion(grafos_train)
    aplicar_normalizacion(grafos_train + grafos_test, stats)

    print(f"carpetas={CARPETAS}")
    print(f"n_grafos={len(grafos)}")
    print(f"train={len(grafos_train)} test={len(grafos_test)}")
    node_counts = [data.num_nodes for data in grafos]
    print(
        f"nodos_por_grafo=min {min(node_counts)}, "
        f"max {max(node_counts)}, limite {MAX_CENTER_CELLS}"
    )
    print(f"features={NODE_FEATURES}")

    model, history, device = entrenar_modelo(grafos_train, grafos_test)
    df_pred, resumen = evaluar_modelo(model, grafos_test, stats, device)

    print("\nMetricas test:")
    for k, v in resumen.items():
        print(f"{k}: {v:.8f}")

    print("\nPredicciones test:")
    #print(df_pred.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history_path = OUT_DIR / "training_history.csv"
    pred_path = OUT_DIR / "predicciones_test.csv"
    model_path = OUT_DIR / "modelo_gnn.pt"
    loss_plot_path = OUT_DIR / "loss_train_test.png"
    pred_plot_path = OUT_DIR / "mapa_reales_vs_pred.png"

    history.to_csv(history_path, index=False)
    df_pred.to_csv(pred_path, index=False)
    plot_losses(history, loss_plot_path)
    plot_reales_vs_predichos_bivariado(df_pred, pred_plot_path, modelo="GNN")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_features": NODE_FEATURES,
            "stats": {k: v.cpu() for k, v in stats.items()},
            "config": {
                "carpetas": CARPETAS,
                "excluir_borde": EXCLUIR_BORDE,
                "max_center_cells": MAX_CENTER_CELLS,
                "min_cells": MIN_CELLS,
                "test_size": TEST_SIZE,
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
            },
        },
        model_path,
    )

    print(f"\nGuardado historial en: {history_path}")
    print(f"Guardadas predicciones en: {pred_path}")
    print(f"Guardada grafica train/test loss en: {loss_plot_path}")
    print(f"Guardado mapa real/predicho en: {pred_plot_path}")
    print(f"Guardado modelo en: {model_path}")


if __name__ == "__main__":
    main()
