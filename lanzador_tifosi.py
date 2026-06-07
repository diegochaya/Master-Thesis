from __future__ import annotations

import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np


# ----------------------------
# Geometria y lectura de datos
# ----------------------------

def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _perimetro(vertices: List[Tuple[float, float]]) -> float:
    if len(vertices) < 2:
        return 0.0
    return sum(_dist(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices)))


def _bbox_vertices(vertices: List[Tuple[float, float]]):
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return {
        "xmin": min(xs),
        "xmax": max(xs),
        "ymin": min(ys),
        "ymax": max(ys),
    }


def _update_bbox(b1, b2):
    return {
        "xmin": min(b1["xmin"], b2["xmin"]),
        "xmax": max(b1["xmax"], b2["xmax"]),
        "ymin": min(b1["ymin"], b2["ymin"]),
        "ymax": max(b1["ymax"], b2["ymax"]),
    }


def _leer_frames_dcells(ruta_dcells: Path):
    texto = ruta_dcells.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    frames = []
    frame_global = 0

    while i < len(texto):
        cabecera = texto[i].strip()
        i += 1
        if not cabecera:
            continue

        partes = cabecera.split()
        if len(partes) < 2:
            continue

        try:
            n_celulas = int(partes[0])
            stage = int(partes[1])
        except ValueError:
            continue

        frame_global += 1
        filas = []

        for _ in range(n_celulas):
            if i >= len(texto):
                break

            linea = texto[i].strip()
            i += 1
            if not linea:
                continue

            toks = linea.split()
            cell_id = toks[0]
            cell_type = int(toks[1])
            n_vertices = int(toks[2])
            area = float(toks[3])
            n_proteins = int(toks[4])

            idx = 5 + n_proteins
            center_x = float(toks[idx])
            center_y = float(toks[idx + 1])
            idx += 2

            vecinos = [int(float(x)) for x in toks[idx: idx + n_vertices]]
            idx += n_vertices

            coords = [float(x) for x in toks[idx: idx + 2 * n_vertices]]
            vertices = [(coords[j], coords[j + 1]) for j in range(0, len(coords), 2)]

            filas.append(
                {
                    "frame": frame_global,
                    "stage": stage,
                    "cell_id": cell_id,
                    "cell_type": cell_type,
                    "area": area,
                    "center_x": center_x,
                    "center_y": center_y,
                    "n_vertices": n_vertices,
                    "n_neighbors": sum(v >= 0 for v in vecinos),
                    "is_border": any(v < 0 for v in vecinos),
                    "perimeter": _perimetro(vertices),
                    "neighbors": vecinos,
                    "vertices": vertices,
                }
            )

        if filas:
            frames.append(pd.DataFrame(filas))

    return frames


def leer_imagen_final(ruta_dcells: Path) -> pd.DataFrame:
    frames = _leer_frames_dcells(ruta_dcells)
    if not frames:
        raise RuntimeError("No se ha podido leer ningún frame de dcells.dat")
    return frames[-1].copy()


def leer_historia_resumida(ruta_dcells: Path) -> pd.DataFrame:
    frames = _leer_frames_dcells(ruta_dcells)
    resumen = []

    for df in frames:
        interior = df.loc[~df["is_border"]].copy()
        usar = interior if not interior.empty else df

        resumen.append(
            {
                "frame": int(df["frame"].iloc[0]),
                "stage": int(df["stage"].iloc[0]),
                "n_cells_total": int(len(df)),
                "n_cells_interior": int(len(interior)),
                "mean_area": float(usar["area"].mean()),
                "std_area": float(usar["area"].std(ddof=0)) if len(usar) > 1 else 0.0,
                "mean_perimeter": float(usar["perimeter"].mean()),
                "mean_neighbors": float(usar["n_neighbors"].mean()),
            }
        )

    return pd.DataFrame(resumen)


# ----------------------------
# Configuracion XML
# ----------------------------

def actualizar_config_simple(
    template_xml: Path,
    output_xml: Path,
    *,
    lambda_cc: float,
    gamma: float,
    speed: float,
    ncellsx: int,
    ncellsy: int,
    initial_state_file: str | Path | None,
    stage1_duration: int,
    stage1_intermediate: int,
    stage2_duration: int,
    stage2_intermediate: int,
) -> None:
    tree = ET.parse(template_xml)
    root = tree.getroot()

    root.find("./global/itissue/ncellsx").text = str(ncellsx)
    root.find("./global/itissue/ncellsy").text = str(ncellsy)
    file_tag = root.find("./global/itissue/file")
    file_tag.set("f", "" if initial_state_file is None else str(initial_state_file))

    stages = root.findall("./stages/stage")
    stages[0].set("duration", str(stage1_duration))
    stages[0].set("intermediate", str(stage1_intermediate))
    stages[1].set("duration", str(stage2_duration))
    stages[1].set("intermediate", str(stage2_intermediate))

    root.find("./potentials/potential/LAMBDA[@t1='neural'][@t2='neural']").text = str(lambda_cc)
    root.find("./potentials/potential/GAMMA[@t='neural']").text = str(gamma)
    root.find("./cycles/cycle/speed[@t='neural']").text = str(speed)

    tree.write(output_xml, encoding="UTF-8", xml_declaration=True)


# ----------------------------
# Plot de stages finales
# ----------------------------

def _plot_dataframe_cells(
    df_cells: pd.DataFrame,
    output_png: Path,
    *,
    titulo: str,
    facecolor: str = "#9ecae1",
    edgecolor: str = "#ffffff",
    linewidth: float = 1.2,
    dpi: int = 300,
) -> Path:
    if df_cells.empty:
        raise ValueError("El DataFrame de células está vacío")

    bbox = None
    patches = []

    for _, row in df_cells.iterrows():
        vertices = row["vertices"]
        cell_bbox = _bbox_vertices(vertices)
        bbox = cell_bbox if bbox is None else _update_bbox(bbox, cell_bbox)
        patches.append(
            Polygon(
                vertices,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
        )
    #print("bbox:", bbox)

    for i, row in df_cells.iterrows():
        verts = row["vertices"]
        for v in verts:
            if not np.isfinite(v[0]) or not np.isfinite(v[1]):
                print("VERTICE NO FINITO en fila", i, "cell_id=", row.get("cell_id"))
                print("vertices =", verts)
                break
    fig, ax = plt.subplots(figsize=(8, 8))
    pc = PatchCollection(patches, match_original=True)
    ax.add_collection(pc)
    ax.set_xlim(bbox["xmin"], bbox["xmax"])
    ax.set_ylim(bbox["ymin"], bbox["ymax"])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(titulo)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return output_png


def leer_final_stage(tifosi_dir: str | Path, stage: int) -> pd.DataFrame:
    tifosi_dir = Path(tifosi_dir)
    ruta_stage = tifosi_dir / f"dcells_final_s{stage}.dat"

    if ruta_stage.exists():
        frames = _leer_frames_dcells(ruta_stage)
        if not frames:
            raise RuntimeError(f"No se ha podido leer el final del stage {stage}")
        return frames[-1].copy()

    ruta_dcells = tifosi_dir / "dcells.dat"
    frames = _leer_frames_dcells(ruta_dcells)
    frames_stage = [df for df in frames if int(df["stage"].iloc[0]) == stage]
    if not frames_stage:
        raise FileNotFoundError(f"No encuentro datos del stage {stage} en {ruta_dcells}")
    return frames_stage[-1].copy()


def pintar_final_stage(
    tifosi_dir: str | Path,
    stage: int,
    *,
    output_png: str | Path | None = None,
    titulo: str | None = None,
) -> Path:
    tifosi_dir = Path(tifosi_dir)
    df_stage = leer_final_stage(tifosi_dir, stage)

    if output_png is None:
        output_png = tifosi_dir / f"stage_{stage}_final.png"
    else:
        output_png = Path(output_png)

    if titulo is None:
        titulo = f"Stage {stage} final"

    return _plot_dataframe_cells(df_stage, output_png, titulo=titulo)


def pintar_stage_1_y_2(
    tifosi_dir: str | Path,
    *,
    carpeta_salida: str | Path | None = None,
) -> tuple[Path, Path]:
    tifosi_dir = Path(tifosi_dir)
    carpeta_salida = Path(carpeta_salida) if carpeta_salida is not None else tifosi_dir / "plots"
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    png1 = pintar_final_stage(
        tifosi_dir,
        1,
        output_png=carpeta_salida / "stage1_final.png",
        titulo="Final stage 1",
    )
    png2 = pintar_final_stage(
        tifosi_dir,
        2,
        output_png=carpeta_salida / "stage2_final.png",
        titulo="Final stage 2",
    )
    return png1, png2


# ----------------------------
# Ejecucion
# ----------------------------

def ejecutar_comando(comando: list[str], cwd: Path) -> None:
    subprocess.run(comando, cwd=str(cwd), check=True)


def lanzar_simulacion(
    *,
    lambda_cc: float = 0.075,
    gamma: float = 0.040,
    speed: float = 3.08,
    ncellsx: int = 10,
    ncellsy: int = 10,
    stage1_duration: int = 10,
    stage1_intermediate: int = 200,
    stage2_duration: int = 20,
    stage2_intermediate: int = 500,
    initial_state_file: str | Path | None = "initial_conditions/hexagonal_2000_1.dat",
    tifosi_dir: str | Path | None = None,
    template_name: str = "config_simple_neuraltube.xml",
    active_config_name: str = "config.xml",
    compilar: bool = True,
    ejecutar: bool = True,
    pintar: bool = False,
):
    script_dir = Path(__file__).resolve().parent
    tifosi_dir = Path(tifosi_dir) if tifosi_dir is not None else script_dir / "TIFOSI"
    if initial_state_file is not None:
        initial_state_file = Path(initial_state_file)
        if not initial_state_file.is_absolute():
            initial_state_file = (script_dir / initial_state_file).resolve()

    template_xml = tifosi_dir / template_name
    active_xml = tifosi_dir / active_config_name
    dcells_path = tifosi_dir / "dcells.dat"

    if not tifosi_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta TIFOSI: {tifosi_dir}")
    if not template_xml.exists():
        raise FileNotFoundError(f"No existe la plantilla XML: {template_xml}")
    if initial_state_file is not None and not initial_state_file.exists():
        raise FileNotFoundError(f"No existe la condicion inicial: {initial_state_file}")

    actualizar_config_simple(
        template_xml,
        active_xml,
        lambda_cc=lambda_cc,
        gamma=gamma,
        speed=speed,
        ncellsx=ncellsx,
        ncellsy=ncellsy,
        initial_state_file=initial_state_file,
        stage1_duration=stage1_duration,
        stage1_intermediate=stage1_intermediate,
        stage2_duration=stage2_duration,
        stage2_intermediate=stage2_intermediate,
    )

    if compilar:
        ejecutar_comando([sys.executable, "compile.py"], cwd=tifosi_dir)

    if ejecutar:
        ejecutar_comando(["./bin/tifosi"], cwd=tifosi_dir)

    if not dcells_path.exists():
        raise FileNotFoundError(f"No se ha generado dcells.dat en: {dcells_path}")

    df_final = leer_imagen_final(dcells_path)
    df_history = leer_historia_resumida(dcells_path)

    if pintar:
        pintar_stage_1_y_2(tifosi_dir)

    return df_final, df_history

