import ast
import re

import numpy as np
import pandas as pd
from scipy.stats import skew



def _to_bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y", "si", "sí"})
    )


def _parse_vertices(v):
    if isinstance(v, str):
        v = ast.literal_eval(v)

    arr = np.asarray(v, dtype=float)

    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise ValueError("No se pueden reinterpretar los vértices como pares (x, y).")
        arr = arr.reshape(-1, 2)

    if arr.shape[-1] != 2:
        arr = arr.reshape(-1, 2)

    return arr


def _elongation_from_vertices(vertices, eps=1e-12):
    verts = _parse_vertices(vertices)
    if len(verts) < 2:
        return 1.0

    centered = verts - verts.mean(axis=0)
    cov = centered.T @ centered / len(centered)
    eigvals = np.linalg.eigvalsh(cov)
    eig_min = max(float(eigvals[0]), eps)
    eig_max = max(float(eigvals[-1]), eps)
    return np.sqrt(eig_max / eig_min)


def limpiar_df(
    df,
    excluir_borde=True,
    n_center_cells=196,
    imprimir_porcentaje=None,
    nombre_df=None,
):
    n_original = len(df)
    d = df.copy()

    if excluir_borde and "is_border" in d.columns:
        d = d.loc[~_to_bool_series(d["is_border"])].copy()

    mask_valid = (
        np.isfinite(d["area"])
        & np.isfinite(d["perimeter"])
        & (d["area"] > 0)
        & np.isfinite(d["center_x"])
        & np.isfinite(d["center_y"])
    )
    d = d.loc[mask_valid].copy()

    n_tras_filtros = len(d)

    if n_center_cells is not None and len(d) > int(n_center_cells):
        cx0 = float(d["center_x"].mean())
        cy0 = float(d["center_y"].mean())
        d = d.copy()
        d["_dist2_center"] = (d["center_x"] - cx0) ** 2 + (d["center_y"] - cy0) ** 2
        d = (
            d.sort_values("_dist2_center", kind="stable")
            .head(int(n_center_cells))
            .drop(columns="_dist2_center")
            .copy()
        )

    n_final = len(d)
    d["shape_index"] = d["perimeter"] / np.sqrt(d["area"])

    if imprimir_porcentaje:
        etiqueta = f"[{nombre_df}] " if nombre_df is not None else ""
        pct_original = 100 * n_final / n_original if n_original > 0 else 0.0
        pct_filtrado = 100 * n_final / n_tras_filtros if n_tras_filtros > 0 else 0.0
        print(
            f"{etiqueta}usando {n_final}/{n_original} células "
            f"({pct_original:.2f}% del total, {pct_filtrado:.2f}% tras filtros)"
        )

    return d


def _tag_a_float(tag):
    return float(str(tag).replace("m", "-").replace("p", "."))


def _extraer_lambda_gamma(clave):
    if isinstance(clave, tuple) and len(clave) == 2:
        return float(clave[0]), float(clave[1])

    m = re.search(r"lam_(?P<lam>.*?)__gam_(?P<gam>.*?)__", str(clave))
    if m is None:
        raise ValueError(
            f"No puedo extraer lambda/gamma de la clave {clave!r}. "
            "Necesito o bien una clave (lambda, gamma) o un sim_id de dataframes_v4."
        )

    return _tag_a_float(m.group("lam")), _tag_a_float(m.group("gam"))


def extraer_features(
    simulaciones,
    excluir_borde=True,
    n_center_cells=196,
):
    """
    Extrae features agregadas por simulación.
    """
    filas = []

    for clave, df in sorted(simulaciones.items(), key=lambda x: str(x[0])):
        lam, gam = _extraer_lambda_gamma(clave)
        d = limpiar_df(
            df,
            excluir_borde=excluir_borde,
            n_center_cells=n_center_cells,
        )

        if len(d) < 5:
            print(len(d))
            print((lam, gam))
            print(df.columns)
            continue

        area = d["area"].to_numpy(dtype=float)
        perimeter = d["perimeter"].to_numpy(dtype=float)
        shape = d["shape_index"].to_numpy(dtype=float)
        elongation = np.array(
            [_elongation_from_vertices(vertices) for vertices in d["vertices"]],
            dtype=float,
        )
        neighbors = d["n_neighbors"].to_numpy(dtype=float)
        cx = d["center_x"].to_numpy(dtype=float)
        cy = d["center_y"].to_numpy(dtype=float)

        fila = {
            "lambda_cc": lam,
            "gamma": gam,

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
            "xy_corr": np.corrcoef(cx, cy)[0, 1] if len(cx) > 1 else 0.0,
        }

        for k in [4, 5, 6, 7, 8]:
            fila[f"frac_neighbors_{k}"] = np.mean(neighbors == k)

        """
        if "cell_type" in d.columns:
            tipos = pd.Series(d["cell_type"]).value_counts(normalize=True)
            for t, frac in tipos.items():
                fila[f"frac_celltype_{t}"] = frac
        """
        filas.append(fila)

    if not filas:
        return pd.DataFrame()

    Xy = pd.DataFrame(filas).fillna(0.0)
    Xy = Xy.sort_values(["lambda_cc", "gamma"]).reset_index(drop=True)
    return Xy
