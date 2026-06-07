import math
from pathlib import Path

import numpy as np


def r2_score_regresion(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.isclose(ss_tot, 0.0):
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse_sobre_rango(errores, y_true):
    errores = np.asarray(errores, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    rango = float(np.max(y_true) - np.min(y_true))
    if np.isclose(rango, 0.0):
        return float("nan")
    return float(np.sqrt(np.mean(errores**2)) / rango)


def calcular_metricas_parametros(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true e y_pred deben tener la misma forma: {y_true.shape} != {y_pred.shape}"
        )
    if y_true.ndim != 2 or y_true.shape[1] < 2:
        raise ValueError("Se esperan arrays 2D con al menos dos columnas: lambda_cc y gamma.")

    errores = y_pred - y_true

    return {
        "mae_lambda": float(np.mean(np.abs(errores[:, 0]))),
        "mae_gamma": float(np.mean(np.abs(errores[:, 1]))),
        "rmse_lambda": float(np.sqrt(np.mean(errores[:, 0] ** 2))),
        "rmse_gamma": float(np.sqrt(np.mean(errores[:, 1] ** 2))),
        "r2_lambda": r2_score_regresion(y_true[:, 0], y_pred[:, 0]),
        "r2_gamma": r2_score_regresion(y_true[:, 1], y_pred[:, 1]),
        "rmse_rango_lambda": rmse_sobre_rango(errores[:, 0], y_true[:, 0]),
        "rmse_rango_gamma": rmse_sobre_rango(errores[:, 1], y_true[:, 1]),
    }


def _normalizar_01(x, xmin, xmax):
    x = np.asarray(x, dtype=float)
    if np.isclose(xmax, xmin):
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def _colores_bivariados(lambda_vals, gamma_vals, lam_min, lam_max, gam_min, gam_max):
    lam_n = _normalizar_01(lambda_vals, lam_min, lam_max)
    gam_n = _normalizar_01(gamma_vals, gam_min, gam_max)

    r = lam_n
    g = 0.25 + 0.5 * (1.0 - np.abs(lam_n - gam_n))
    b = gam_n
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def _chi2_quantile_df2(prob):
    return -2.0 * math.log(max(1.0 - float(prob), 1e-12))


def _ellipse_patch(mu, cov, prob, **kwargs):
    from matplotlib.patches import Ellipse

    q = _chi2_quantile_df2(prob)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2.0 * math.sqrt(q * eigvals[0])
    height = 2.0 * math.sqrt(q * eigvals[1])
    return Ellipse(xy=mu, width=width, height=height, angle=angle, **kwargs)


def plot_reales_vs_predichos_bivariado(
    df_pred,
    output_png,
    modelo,
    titulo=None,
    cov_pred=None,
    n_ellipses=10,
    ellipse_prob=0.90,
    etiqueta_predicho=None,
    mostrar_fondo=True,
    real_size=70,
    pred_size=70,
    line_width=1.2,
    line_alpha=0.85,
    real_linewidth=1.8,
    pred_linewidth=0.6,
    fondo_alpha=0.50,
    axis_label_size=15,
    dpi=300,
):
    import matplotlib.pyplot as plt

    lam_real = df_pred["lambda_real"].to_numpy(dtype=float)
    gam_real = df_pred["gamma_real"].to_numpy(dtype=float)
    lam_pred = df_pred["lambda_pred"].to_numpy(dtype=float)
    gam_pred = df_pred["gamma_pred"].to_numpy(dtype=float)

    lam_min = float(min(lam_real.min(), lam_pred.min()))
    lam_max = float(max(lam_real.max(), lam_pred.max()))
    gam_min = float(min(gam_real.min(), gam_pred.min()))
    gam_max = float(max(gam_real.max(), gam_pred.max()))

    lam_pad = 0.05 * max(lam_max - lam_min, 1e-6)
    gam_pad = 0.05 * max(gam_max - gam_min, 1e-6)
    lam_min -= lam_pad
    lam_max += lam_pad
    gam_min -= gam_pad
    gam_max += gam_pad

    point_colors = _colores_bivariados(
        lam_real,
        gam_real,
        lam_min,
        lam_max,
        gam_min,
        gam_max,
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    if mostrar_fondo:
        grid_n = 200
        lam_grid = np.linspace(lam_min, lam_max, grid_n)
        gam_grid = np.linspace(gam_min, gam_max, grid_n)
        lam_mesh, gam_mesh = np.meshgrid(lam_grid, gam_grid)
        color_mesh = _colores_bivariados(
            lam_mesh.ravel(),
            gam_mesh.ravel(),
            lam_min,
            lam_max,
            gam_min,
            gam_max,
        ).reshape(grid_n, grid_n, 3)
        ax.imshow(
            color_mesh,
            origin="lower",
            extent=[lam_min, lam_max, gam_min, gam_max],
            aspect="auto",
            alpha=fondo_alpha,
        )

    for i in range(len(df_pred)):
        ax.plot(
            [lam_real[i], lam_pred[i]],
            [gam_real[i], gam_pred[i]],
            color=point_colors[i],
            linewidth=line_width,
            alpha=line_alpha,
            zorder=2,
        )

    ax.scatter(
        lam_real,
        gam_real,
        c=point_colors,
        s=real_size,
        marker="x",
        linewidths=real_linewidth,
        label="Real",
        zorder=3,
    )

    if etiqueta_predicho is None:
        etiqueta_predicho = "Predicted mean" if cov_pred is not None else "Predicted"
    ax.scatter(
        lam_pred,
        gam_pred,
        c=point_colors,
        s=pred_size,
        edgecolors="black",
        linewidths=pred_linewidth,
        label=etiqueta_predicho,
        zorder=4,
    )

    if cov_pred is not None and len(df_pred) > 0:
        cov_pred = np.asarray(cov_pred, dtype=float)
        if cov_pred.shape[0] != len(df_pred):
            raise ValueError(
                "cov_pred debe tener una covarianza por fila de df_pred: "
                f"{cov_pred.shape[0]} != {len(df_pred)}"
            )

        ellipse_idx = np.linspace(
            0,
            len(df_pred) - 1,
            min(n_ellipses, len(df_pred)),
            dtype=int,
        )
        for i in ellipse_idx:
            patch = _ellipse_patch(
                np.array([lam_pred[i], gam_pred[i]], dtype=float),
                cov_pred[i],
                prob=ellipse_prob,
                facecolor="none",
                edgecolor="black",
                linewidth=1.2,
                alpha=0.65,
                zorder=5,
            )
            ax.add_patch(patch)

    ax.set_xlabel(r"$\mathbf{\Lambda}$", fontsize=axis_label_size, fontweight="bold")
    ax.set_ylabel(r"$\mathbf{\Gamma}$", fontsize=axis_label_size, fontweight="bold")
    if titulo is None:
        titulo = f"{modelo}: true and predicted parameters"
        if cov_pred is not None:
            titulo = (
                f"{modelo}: true parameters, predicted means "
                f"and {int(ellipse_prob * 100)}% ellipses"
            )
    ax.set_title(titulo)
    ax.set_xlim(lam_min, lam_max)
    ax.set_ylim(gam_min, gam_max)
    ax.legend()
    ax.grid(True, alpha=0.25)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=dpi)
    plt.close(fig)
