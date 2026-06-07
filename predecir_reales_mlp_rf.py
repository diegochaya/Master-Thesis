from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from scipy.stats import skew

from Preprocesado import _elongation_from_vertices, limpiar_df


META_COLS = [
    "real_id",
    "stage",
    "region",
    "csv_name",
    "source_filename",
    "csv_path",
    "n_cells_raw",
    "n_cells_used",
    "n_missing_vertices",
    "n_ambiguous_bonds",
    "excluir_borde",
    "n_center_cells",
    "length_scale",
    "max_elongation",
]


@dataclass
class ModeloCargado:
    tipo: str
    tag: str
    path: Path
    model: object
    feature_cols: list[str]
    stats: dict
    config: dict
    torch_module: object | None = None


def normalizar_token(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        as_float = float(text)
    except ValueError:
        return text
    if as_float.is_integer():
        return str(int(as_float))
    return text


def parse_hash_tokens(value):
    if pd.isna(value):
        return []
    tokens = []
    for token in str(value).split("#"):
        token = normalizar_token(token)
        if token is not None:
            tokens.append(token)
    return tokens


def parse_vertices(value, length_scale=1.0):
    if pd.isna(value):
        return []

    vertices = []
    for token in str(value).split("#"):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            x_text, y_text = token.split(":", 1)
        elif "," in token:
            x_text, y_text = token.split(",", 1)
        else:
            raise ValueError(f"Vertice con formato inesperado: {token!r}")
        vertices.append((float(x_text) * length_scale, float(y_text) * length_scale))
    return vertices


def reconstruir_vecinos(df):
    cell_ids = [normalizar_token(v) for v in df["local_id_cells"]]
    row_bonds = [parse_hash_tokens(v) for v in df["local_id_of_bonds"]]

    bond_to_cells = defaultdict(set)
    for cell_id, bonds in zip(cell_ids, row_bonds):
        if cell_id is None:
            continue
        for bond_id in bonds:
            bond_to_cells[bond_id].add(cell_id)

    vecinos_por_celda = []
    for cell_id, bonds in zip(cell_ids, row_bonds):
        vecinos = set()
        for bond_id in bonds:
            vecinos.update(bond_to_cells[bond_id])
        vecinos.discard(cell_id)
        vecinos_por_celda.append(
            sorted(vecinos, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
        )

    n_ambiguous_bonds = sum(1 for cells in bond_to_cells.values() if len(cells) > 2)
    return vecinos_por_celda, n_ambiguous_bonds


def leer_csv_real(csv_path, csv_root, length_scale=1.0):
    raw = pd.read_csv(csv_path, sep="\t")
    vecinos, n_ambiguous_bonds = reconstruir_vecinos(raw)

    vertices = [parse_vertices(v, length_scale=length_scale) for v in raw["vx_coords_cells"]]
    n_missing_vertices = sum(len(v) == 0 for v in vertices)

    n_vertices = pd.to_numeric(
        raw.get("nb_of_vertices_cut_off", raw.get("nb_of_vertices_no_cut_off")),
        errors="coerce",
    )
    n_vertices = n_vertices.fillna(pd.Series([len(v) for v in vertices], index=raw.index))

    df = pd.DataFrame(
        {
            "frame": pd.to_numeric(raw.get("frame_nb", 0), errors="coerce").fillna(0).astype(int),
            "stage": "",
            "cell_id": [normalizar_token(v) for v in raw["local_id_cells"]],
            "cell_type": 0,
            "area": pd.to_numeric(raw["area_cells"], errors="coerce") * (length_scale**2),
            "center_x": pd.to_numeric(raw["center_x_cells"], errors="coerce") * length_scale,
            "center_y": pd.to_numeric(raw["center_y_cells"], errors="coerce") * length_scale,
            "n_vertices": n_vertices.astype(float),
            "n_neighbors": [len(v) for v in vecinos],
            "is_border": raw["is_border_cell"],
            "perimeter": pd.to_numeric(raw["perimeter_length"], errors="coerce") * length_scale,
            "neighbors": vecinos,
            "vertices": vertices,
        }
    )

    rel = csv_path.relative_to(csv_root)
    parts = rel.parts
    stage = parts[0] if len(parts) >= 3 else ""
    region = parts[1] if len(parts) >= 3 else csv_path.parent.name
    source_filename = ""
    if "filename" in raw.columns and len(raw) > 0:
        source_filename = str(raw["filename"].mode().iloc[0])

    meta = {
        "real_id": str(rel.with_suffix("")),
        "stage": stage,
        "region": region,
        "csv_name": csv_path.name,
        "source_filename": source_filename,
        "csv_path": str(csv_path),
        "n_cells_raw": int(len(raw)),
        "n_missing_vertices": int(n_missing_vertices),
        "n_ambiguous_bonds": int(n_ambiguous_bonds),
    }
    return df, meta


def extraer_features_reales(
    df,
    meta,
    excluir_borde=True,
    n_center_cells=196,
    length_scale=1.0,
    max_elongation=10.0,
):
    d = limpiar_df(
        df,
        excluir_borde=excluir_borde,
        n_center_cells=n_center_cells,
    )

    if len(d) < 5:
        return None

    area = d["area"].to_numpy(dtype=float)
    perimeter = d["perimeter"].to_numpy(dtype=float)
    shape = d["shape_index"].to_numpy(dtype=float)
    elongation = np.array(
        [_elongation_from_vertices(vertices) for vertices in d["vertices"]],
        dtype=float,
    )
    if max_elongation is not None:
        elongation = np.clip(elongation, 1.0, float(max_elongation))
    neighbors = d["n_neighbors"].to_numpy(dtype=float)
    cx = d["center_x"].to_numpy(dtype=float)
    cy = d["center_y"].to_numpy(dtype=float)

    xy_corr = np.corrcoef(cx, cy)[0, 1] if len(cx) > 1 else 0.0

    fila = {
        **meta,
        "n_cells_used": int(len(d)),
        "excluir_borde": bool(excluir_borde),
        "n_center_cells": int(n_center_cells) if n_center_cells is not None else -1,
        "length_scale": float(length_scale),
        "max_elongation": float(max_elongation) if max_elongation is not None else -1.0,
        "area_mean": area.mean(),
        "area_std": area.std(ddof=0),
        "area_cv": area.std(ddof=0) / (area.mean() + 1e-12),
        "area_q10": np.quantile(area, 0.10),
        "area_q50": np.quantile(area, 0.50),
        "area_q90": np.quantile(area, 0.90),
        "area_skew": skew(area, bias=False),
        "perimeter_mean": perimeter.mean(),
        "perimeter_std": perimeter.std(ddof=0),
        "perimeter_cv": perimeter.std(ddof=0) / (perimeter.mean() + 1e-12),
        "perimeter_q10": np.quantile(perimeter, 0.10),
        "perimeter_q50": np.quantile(perimeter, 0.50),
        "perimeter_q90": np.quantile(perimeter, 0.90),
        "perimeter_skew": skew(perimeter, bias=False),
        "shape_mean": shape.mean(),
        "shape_std": shape.std(ddof=0),
        "shape_q10": np.quantile(shape, 0.10),
        "shape_q50": np.quantile(shape, 0.50),
        "shape_q90": np.quantile(shape, 0.90),
        "elongation_mean": elongation.mean(),
        "elongation_std": elongation.std(ddof=0),
        "elongation_q10": np.quantile(elongation, 0.10),
        "elongation_q50": np.quantile(elongation, 0.50),
        "elongation_q90": np.quantile(elongation, 0.90),
        "neighbors_mean": neighbors.mean(),
        "neighbors_std": neighbors.std(ddof=0),
        "center_x_std": cx.std(ddof=0),
        "center_y_std": cy.std(ddof=0),
        "xy_corr": xy_corr,
    }

    for k in [4, 5, 6, 7, 8]:
        fila[f"frac_neighbors_{k}"] = np.mean(neighbors == k)

    return fila


def descubrir_csvs(csv_root):
    return sorted(
        p
        for p in csv_root.rglob("cell_*.csv")
        if p.is_file() and not p.name.startswith(".~lock")
    )


def construir_features(
    csv_root,
    excluir_borde=True,
    n_center_cells=196,
    length_scale=1.0,
    max_elongation=10.0,
):
    filas = []
    for csv_path in descubrir_csvs(csv_root):
        df, meta = leer_csv_real(csv_path, csv_root=csv_root, length_scale=length_scale)
        fila = extraer_features_reales(
            df,
            meta,
            excluir_borde=excluir_borde,
            n_center_cells=n_center_cells,
            length_scale=length_scale,
            max_elongation=max_elongation,
        )
        if fila is not None:
            filas.append(fila)

    if not filas:
        raise ValueError("No se pudo construir ningun vector de features con los CSV reales.")

    return pd.DataFrame(filas).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def desnormalizar_y(y_pred_n, stats):
    return y_pred_n * as_numpy(stats["y_std"]) + as_numpy(stats["y_mean"])


def cargar_random_forest(path):
    import joblib

    ckpt = joblib.load(path)
    return ModeloCargado(
        tipo="random_forest",
        tag=path.parent.name,
        path=path,
        model=ckpt["model"],
        feature_cols=list(ckpt["feature_cols"]),
        stats=ckpt["stats"],
        config=ckpt.get("config", {}),
    )


def construir_mlp(torch, input_dim, hidden_dim, output_dim):
    nn = torch.nn

    class MLPRegresor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    return MLPRegresor()


def torch_load_compatible(torch, path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def cargar_mlp(path, device="cpu"):
    import torch

    ckpt = torch_load_compatible(torch, path, device)
    state_dict = ckpt["model_state_dict"]
    input_dim = int(state_dict["net.0.weight"].shape[1])
    hidden_dim = int(state_dict["net.0.weight"].shape[0])
    output_dim = int(state_dict["net.2.weight"].shape[0])

    model = construir_mlp(torch, input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return ModeloCargado(
        tipo="mlp",
        tag=path.parent.name,
        path=path,
        model=model,
        feature_cols=list(ckpt["feature_cols"]),
        stats=ckpt["stats"],
        config=ckpt.get("config", {}),
        torch_module=torch,
    )


def config_preprocesado(modelo):
    config = modelo.config or {}
    if "excluir_borde" in config:
        excluir_borde = bool(config["excluir_borde"])
    else:
        excluir_borde = not bool(config.get("incluir_borde", False))
    n_center_cells = config.get("n_center_cells", 196)
    return excluir_borde, n_center_cells


def preparar_X(features, feature_cols, model_path):
    missing = [col for col in feature_cols if col not in features.columns]
    if missing:
        raise ValueError(
            f"El modelo {model_path} espera features que no se han construido: {missing}"
        )
    return features[feature_cols].to_numpy(dtype=np.float32)


def predecir_modelo(modelo, features, clip_z=None):
    X = preparar_X(features, modelo.feature_cols, modelo.path)
    X_n = (X - as_numpy(modelo.stats["x_mean"])) / as_numpy(modelo.stats["x_std"])
    abs_z = np.abs(X_n)
    max_abs_z = abs_z.max(axis=1)
    mean_abs_z = abs_z.mean(axis=1)
    frac_abs_z_gt3 = (abs_z > 3.0).mean(axis=1)
    frac_abs_z_gt5 = (abs_z > 5.0).mean(axis=1)
    if clip_z is not None:
        X_n_pred = np.clip(X_n, -float(clip_z), float(clip_z))
    else:
        X_n_pred = X_n

    if modelo.tipo == "random_forest":
        pred_n = modelo.model.predict(X_n_pred)
    elif modelo.tipo == "mlp":
        torch = modelo.torch_module
        device = next(modelo.model.parameters()).device
        X_t = torch.tensor(X_n_pred, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_n = modelo.model(X_t).cpu().numpy()
    else:
        raise ValueError(f"Tipo de modelo no soportado: {modelo.tipo}")

    pred = desnormalizar_y(pred_n, modelo.stats)
    out = features[META_COLS].copy()
    out.insert(0, "model_path", str(modelo.path))
    out.insert(0, "model_tag", modelo.tag)
    out.insert(0, "model_type", modelo.tipo)
    out["lambda_pred"] = pred[:, 0]
    out["gamma_pred"] = pred[:, 1]
    out["trained_on"] = str(modelo.config.get("carpetas", ""))
    out["max_abs_z"] = max_abs_z
    out["mean_abs_z"] = mean_abs_z
    out["frac_abs_z_gt3"] = frac_abs_z_gt3
    out["frac_abs_z_gt5"] = frac_abs_z_gt5
    out["clip_z"] = float(clip_z) if clip_z is not None else -1.0
    return out


def parse_length_scale(value):
    value_text = str(value).strip().lower()
    if value_text == "auto":
        return None
    length_scale = float(value)
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("--length-scale debe ser 'auto' o un numero positivo.")
    return length_scale


def estimar_length_scale(features_unscaled, modelo):
    candidates = []

    area_cols = ["area_mean", "area_q50", "area_q10", "area_q90"]
    length_cols = ["perimeter_mean", "perimeter_q50", "center_x_std", "center_y_std"]

    for col in area_cols:
        if col in modelo.feature_cols and col in features_unscaled.columns:
            idx = modelo.feature_cols.index(col)
            train_ref = float(as_numpy(modelo.stats["x_mean"])[idx])
            real_ref = float(features_unscaled[col].median())
            if train_ref > 0 and real_ref > 0:
                candidates.append(np.sqrt(train_ref / real_ref))

    for col in length_cols:
        if col in modelo.feature_cols and col in features_unscaled.columns:
            idx = modelo.feature_cols.index(col)
            train_ref = float(as_numpy(modelo.stats["x_mean"])[idx])
            real_ref = float(features_unscaled[col].median())
            if train_ref > 0 and real_ref > 0:
                candidates.append(train_ref / real_ref)

    candidates = [float(c) for c in candidates if np.isfinite(c) and c > 0]
    if not candidates:
        raise ValueError(
            f"No se pudo estimar length_scale automaticamente para {modelo.path}."
        )
    return float(np.median(candidates))


def scale_tag(length_scale):
    return f"{length_scale:.8g}".replace("-", "m").replace(".", "p")


def normalizar_carpetas_config(config):
    carpetas = config.get("carpetas")
    if carpetas is None:
        carpetas = config.get("carpeta")
    if carpetas is None:
        return []
    if isinstance(carpetas, (str, Path)):
        return [str(carpetas)]
    return [str(carpeta) for carpeta in carpetas]


def obtener_training_bounds(modelo):
    from dataframes_v4 import leer_dataframes_barrido

    carpetas = normalizar_carpetas_config(modelo.config or {})
    indices = []
    carpetas_usadas = []
    carpetas_faltantes = []
    for carpeta in carpetas:
        try:
            indice = leer_dataframes_barrido(
                carpeta=carpeta,
                status="ok",
                devolver_indice=True,
            )
        except FileNotFoundError:
            carpetas_faltantes.append(carpeta)
            continue
        indices.append(indice[["lambda_cc", "gamma"]])
        carpetas_usadas.append(carpeta)

    if not indices:
        return None

    params = pd.concat(indices, ignore_index=True)
    return {
        "model_type": modelo.tipo,
        "model_tag": modelo.tag,
        "model_path": str(modelo.path),
        "train_lambda_min": float(params["lambda_cc"].min()),
        "train_lambda_max": float(params["lambda_cc"].max()),
        "train_gamma_min": float(params["gamma"].min()),
        "train_gamma_max": float(params["gamma"].max()),
        "training_folders_used": "|".join(carpetas_usadas),
        "training_folders_missing": "|".join(carpetas_faltantes),
        "training_bounds_complete": len(carpetas_faltantes) == 0,
    }


def plot_predicciones_por_tipo(df_pred, df_bounds, output_dir):
    plot_paths = []
    for model_type, df_tipo in df_pred.groupby("model_type", sort=True):
        bounds_tipo = df_bounds.loc[df_bounds["model_type"] == model_type].copy()
        if len(bounds_tipo) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap("tab10")
        tags = list(df_tipo["model_tag"].drop_duplicates())

        for i, tag in enumerate(tags):
            color = cmap(i % 10)
            pred_tag = df_tipo.loc[df_tipo["model_tag"] == tag]
            ax.scatter(
                pred_tag["lambda_pred"],
                pred_tag["gamma_pred"],
                s=45,
                alpha=0.78,
                color=color,
                label=f"{tag} pred",
            )

            bounds_tag = bounds_tipo.loc[bounds_tipo["model_tag"] == tag]
            if len(bounds_tag) == 0:
                continue
            b = bounds_tag.iloc[0]
            xmin = b["train_lambda_min"]
            xmax = b["train_lambda_max"]
            ymin = b["train_gamma_min"]
            ymax = b["train_gamma_max"]
            complete = bool(b.get("training_bounds_complete", True))
            range_label = "train range" if complete else "train range partial"
            ax.axvspan(xmin, xmax, ymin=0, ymax=1, color=color, alpha=0.08)
            ax.axhspan(ymin, ymax, xmin=0, xmax=1, color=color, alpha=0.05)
            ax.axvline(xmin, color=color, linestyle="--", linewidth=1.0, alpha=0.9)
            ax.axvline(xmax, color=color, linestyle="--", linewidth=1.0, alpha=0.9)
            ax.axhline(ymin, color=color, linestyle=":", linewidth=1.2, alpha=0.9)
            ax.axhline(ymax, color=color, linestyle=":", linewidth=1.2, alpha=0.9)
            ax.plot(
                [xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
                color=color,
                linewidth=1.5,
                alpha=0.95,
                label=f"{tag} {range_label}",
            )

        ax.set_xlabel("lambda predicha")
        ax.set_ylabel("gamma predicha")
        ax.set_title(f"Predicciones reales - {model_type}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()

        output_png = output_dir / f"predicciones_{model_type}.png"
        fig.savefig(output_png, dpi=220)
        plt.close(fig)
        plot_paths.append(output_png)

    return plot_paths


def limpiar_outputs_generados(output_dir):
    patrones = [
        "features_reales__*.csv",
        "predicciones_*.png",
        "predicciones_reales_mlp_rf.csv",
        "length_scales_usadas.csv",
        "training_bounds_modelos.csv",
    ]
    for patron in patrones:
        for path in output_dir.glob(patron):
            if path.is_file():
                path.unlink()


def descubrir_modelos(args):
    rf_paths = [Path(p) for p in args.rf_model] if args.rf_model else []
    mlp_paths = [Path(p) for p in args.mlp_model] if args.mlp_model else []

    if not args.no_default_models:
        if not rf_paths:
            rf_paths = [
                Path("outputs/random_forest_dataframesBarrido/modelo_random_forest.joblib")
            ]
        if not mlp_paths:
            mlp_paths = [
                Path("outputs/red_neuronal_dataframesBarrido/modelo_parametros.pt")
            ]

    rf_paths = [path for path in rf_paths if path.exists()]
    mlp_paths = [path for path in mlp_paths if path.exists()]

    return rf_paths, mlp_paths


def cargar_modelos(rf_paths, mlp_paths, device):
    modelos = []
    for path in rf_paths:
        modelos.append(cargar_random_forest(path))
    for path in mlp_paths:
        modelos.append(cargar_mlp(path, device=device))
    if not modelos:
        raise ValueError("No se encontro ningun modelo RF/MLP para cargar.")
    return modelos


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convierte CSV reales de AnalisisDatosReales/csvdata a features y "
            "predice lambda/gamma con los modelos MLP y Random Forest guardados."
        )
    )
    parser.add_argument("--csv-root", type=Path, default=Path("AnalisisDatosReales/csvdata"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predicciones_reales_mlp_rf"))
    parser.add_argument("--rf-model", action="append", default=[])
    parser.add_argument("--mlp-model", action="append", default=[])
    parser.add_argument("--no-default-models", action="store_true")
    parser.add_argument(
        "--length-scale",
        default="auto",
        help=(
            "'auto' estima la escala pixel->simulacion con los stats del modelo. "
            "Tambien puedes pasar un numero, por ejemplo 0.02."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-elongation",
        type=float,
        default=10.0,
        help=(
            "Cota superior para elongation por celula antes de agregar features. "
            "Usa un valor negativo para desactivarla."
        ),
    )
    parser.add_argument(
        "--clip-z",
        type=float,
        default=-1.0,
        help=(
            "Si es positivo, recorta las features normalizadas a [-clip_z, clip_z] "
            "antes de predecir. Por defecto no recorta."
        ),
    )
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Guarda tambien los CSV de features reales usados para predecir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    length_scale_fijo = parse_length_scale(args.length_scale)
    max_elongation = None if args.max_elongation < 0 else args.max_elongation
    clip_z = None if args.clip_z < 0 else args.clip_z
    rf_paths, mlp_paths = descubrir_modelos(args)
    modelos = cargar_modelos(rf_paths, mlp_paths, device=args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    limpiar_outputs_generados(args.output_dir)
    features_cache = {}
    predicciones = []
    bounds_records = []

    for modelo in modelos:
        excluir_borde, n_center_cells = config_preprocesado(modelo)
        bounds = obtener_training_bounds(modelo)
        if bounds is not None:
            bounds_records.append(bounds)

        if length_scale_fijo is None:
            raw_key = (excluir_borde, n_center_cells, 1.0, max_elongation)
            if raw_key not in features_cache:
                features_cache[raw_key] = construir_features(
                    args.csv_root,
                    excluir_borde=excluir_borde,
                    n_center_cells=n_center_cells,
                    length_scale=1.0,
                    max_elongation=max_elongation,
                )
            length_scale = estimar_length_scale(features_cache[raw_key], modelo)
        else:
            length_scale = length_scale_fijo

        cache_key = (excluir_borde, n_center_cells, length_scale, max_elongation)
        if cache_key not in features_cache:
            features = construir_features(
                args.csv_root,
                excluir_borde=excluir_borde,
                n_center_cells=n_center_cells,
                length_scale=length_scale,
                max_elongation=max_elongation,
            )
            features_cache[cache_key] = features
            suffix = (
                f"excluir_borde_{int(excluir_borde)}"
                f"__n_center_{n_center_cells}"
                f"__scale_{scale_tag(length_scale)}"
                f"__maxelong_{scale_tag(max_elongation) if max_elongation is not None else 'none'}"
            )
            if args.save_features:
                features.to_csv(args.output_dir / f"features_reales__{suffix}.csv", index=False)

        predicciones.append(predecir_modelo(modelo, features_cache[cache_key], clip_z=clip_z))

    df_pred = pd.concat(predicciones, ignore_index=True)
    df_bounds = pd.DataFrame(bounds_records)
    pred_path = args.output_dir / "predicciones_reales_mlp_rf.csv"
    bounds_path = args.output_dir / "training_bounds_modelos.csv"
    df_pred.to_csv(pred_path, index=False)
    df_bounds.to_csv(bounds_path, index=False)
    plot_paths = plot_predicciones_por_tipo(df_pred, df_bounds, args.output_dir)

    print(f"Modelos cargados: {len(modelos)}")
    print(f"CSV reales procesados: {df_pred['real_id'].nunique()}")
    print(f"Rangos de entrenamiento guardados en: {bounds_path}")
    print(f"Predicciones guardadas en: {pred_path}")
    for plot_path in plot_paths:
        print(f"Grafica guardada en: {plot_path}")


if __name__ == "__main__":
    main()
