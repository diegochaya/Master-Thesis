from __future__ import annotations

from pathlib import Path

from lanzador_tifosi import _plot_dataframe_cells, lanzar_simulacion, leer_final_stage


# Ajusta estos parametros para buscar una geometria inicial que te guste.
NOMBRE_SALIDA = "hexagonal_stage2_3"
N_CELULAS_OBJETIVO = 14 * 14
LAMBDA_CC = 0
GAMMA = 0.05
SPEED = 1
NCELLSX = 14
NCELLSY = 14
STAGE1_DURATION = 1
STAGE1_INTERMEDIATE = 1
STAGE2_DURATION = 1000
STAGE2_INTERMEDIATE = 1000


def seleccionar_subtejido_centrado(df_cells):
    centro_x = float(df_cells["center_x"].mean())
    centro_y = float(df_cells["center_y"].mean())

    df_sel = df_cells.copy()
    df_sel["dist2_centro"] = (
        (df_sel["center_x"] - centro_x) ** 2 + (df_sel["center_y"] - centro_y) ** 2
    )
    df_sel = (
        df_sel.sort_values("dist2_centro", kind="stable")
        .head(N_CELULAS_OBJETIVO)
        .sort_values(["center_y", "center_x"], kind="stable")
        .reset_index(drop=True)
    )
    return df_sel.drop(columns=["dist2_centro"])


def guardar_estado_inicial(df_cells, output_dat: Path) -> None:
    lineas = [f"{len(df_cells)} 1"]

    for nuevo_id, row in df_cells.iterrows():
        vertices = row["vertices"]
        n_vertices = len(vertices)
        vecinos_dummy = " ".join(["-1"] * n_vertices)
        coords = " ".join(f"{x:.10f} {y:.10f}" for x, y in vertices)

        lineas.append(
            f"{nuevo_id} {int(row['cell_type'])} {n_vertices} {float(row['area']):.10f} 0 "
            f"{float(row['center_x']):.10f} {float(row['center_y']):.10f} "
            f"{vecinos_dummy} {coords}"
        )

    output_dat.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    tifosi_dir = repo_dir / "TIFOSI"
    initial_conditions_dir = repo_dir / "initial_conditions"
    plots_dir = initial_conditions_dir / "plots"

    initial_conditions_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    lanzar_simulacion(
        lambda_cc=LAMBDA_CC,
        gamma=GAMMA,
        speed=SPEED,
        ncellsx=NCELLSX,
        ncellsy=NCELLSY,
        stage1_duration=STAGE1_DURATION,
        stage1_intermediate=STAGE1_INTERMEDIATE,
        stage2_duration=STAGE2_DURATION,
        stage2_intermediate=STAGE2_INTERMEDIATE,
        tifosi_dir=tifosi_dir,
        pintar=False,
    )

    destino_dat = initial_conditions_dir / f"{NOMBRE_SALIDA}.dat"
    destino_png = plots_dir / f"{NOMBRE_SALIDA}.png"
    df_stage2 = leer_final_stage(tifosi_dir, 2)
    df_subtejido = seleccionar_subtejido_centrado(df_stage2)

    guardar_estado_inicial(df_subtejido, destino_dat)
    _plot_dataframe_cells(
        df_subtejido,
        destino_png,
        titulo=f"Initial condition candidate: {NOMBRE_SALIDA}",
    )

    print(f"Condicion inicial guardada en: {destino_dat}")
    print(f"Imagen guardada en: {destino_png}")


if __name__ == "__main__":
    main()
