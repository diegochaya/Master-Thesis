import json
import shutil
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path
import re

import numpy as np
import pandas as pd
from lanzador_tifosi import _plot_dataframe_cells, lanzar_simulacion


def _float_a_tag(x, ndigits=8):
    """
    Convierte un float en un string seguro para nombres de fichero.
    Ejemplo: 0.075 -> '0p075', -1.2 -> 'm1p2'
    """
    s = f"{float(x):.{ndigits}g}"
    return s.replace("-", "m").replace(".", "p")


def _normalizar_id(texto):
    texto = str(texto).strip()
    texto = re.sub(r"\s+", "_", texto)
    texto = re.sub(r"[^A-Za-z0-9_.-]", "_", texto)
    if not texto:
        raise ValueError("El identificador no puede quedar vacío.")
    return texto


def _generar_sim_id(lam, gam, rep_idx):
    marca = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return _normalizar_id(
        f"sim__lam_{_float_a_tag(lam)}__gam_{_float_a_tag(gam)}__rep_{int(rep_idx):03d}__{marca}"
    )


def _nombre_fichero_df_final(lam, gam, sim_id):
    return (
        f"df_final_lambda_{_float_a_tag(lam)}"
        f"_gamma_{_float_a_tag(gam)}"
        f"__sim_{_normalizar_id(sim_id)}.pkl"
    )


def _nombre_fichero_plot_final(lam, gam, sim_id):
    return (
        f"lambda_{lam}_gamma_{gam}"
        f"__sim_{_normalizar_id(sim_id)}.png"
    )


def generar_parameter_pairs_triangulo(
    lambda_values,
    gamma_values,
    *,
    gamma_max_intercept=0.10,
    gamma_max_slope=0.50,
    excluir_lambda_cerca_cero_gamma_baja=True,
    complemento=False,
):
    """
    Devuelve pares (lambda, gamma) de una region triangular.

    La frontera por defecto recorta la esquina de lambda positiva y gamma alta:
    gamma <= gamma_max_intercept - gamma_max_slope * lambda.

    Si complemento=True, devuelve el otro lado de esa frontera:
    gamma > gamma_max_intercept - gamma_max_slope * lambda.
    """
    pares = []
    for lam, gam in product(lambda_values, gamma_values):
        lam = float(lam)
        gam = float(gam)
        gamma_frontera = gamma_max_intercept - gamma_max_slope * lam
        esta_en_triangulo_base = gam <= gamma_frontera

        if complemento:
            if esta_en_triangulo_base:
                continue
        elif not esta_en_triangulo_base:
            continue

        if excluir_lambda_cerca_cero_gamma_baja and -0.05 <= lam <= 0.0 and gam <= 0.035:
            continue
        pares.append((lam, gam))
    return pares


def _safe_json(obj):
    if obj is None:
        return pd.NA
    return json.dumps(obj, sort_keys=True, default=str)


def _columnas_indice_v3():
    return [
        "sim_id",
        "lambda_cc",
        "gamma",
        "replica",
        "init_id",
        "grupo_id",
        "filepath",
        "n_filas",
        "columnas",
        "sim_kwargs",
        "extra_metadata",
        "status",
        "error_msg",
        "traceback",
    ]


def _leer_indice_existente(ruta_indice):
    columnas = _columnas_indice_v3()
    if not ruta_indice.exists():
        return pd.DataFrame(columns=columnas)

    indice = pd.read_csv(ruta_indice)
    for col in columnas:
        if col not in indice.columns:
            indice[col] = pd.NA
    return indice[columnas].copy()


def _preparar_indice_para_guardar(indice):
    columnas = _columnas_indice_v3()
    if indice is None:
        indice = pd.DataFrame(columns=columnas)
    for col in columnas:
        if col not in indice.columns:
            indice[col] = pd.NA
    indice = indice[columnas].copy()
    indice = indice.drop_duplicates(subset=["sim_id"], keep="last")

    if len(indice) > 0:
        indice = indice.sort_values(
            ["lambda_cc", "gamma", "replica", "sim_id"]
        ).reset_index(drop=True)

    return indice


def _guardar_indice_atomico(indice, ruta_indice):
    indice = _preparar_indice_para_guardar(indice)
    ruta_tmp = ruta_indice.with_name(f"{ruta_indice.name}.tmp")
    indice.to_csv(ruta_tmp, index=False)
    ruta_tmp.replace(ruta_indice)
    return indice


def guardar_dataframes_barrido(
    lambda_values=None,
    gamma_values=None,
    carpeta=None,
    lanzar_simulacion=lanzar_simulacion,
    nombre_indice="indice_simulaciones.csv",
    sobrescribir=False,
    replicas=1,
    init_id=None,
    initial_state_files=["hexagonal_2000_1.dat"],
    parameter_pairs=None,
    grupo_id=None,
    continuar_si_error=True,
    extra_metadata=None,
    append_indice=True,
    actualizar_indice_cada=1,
    carpeta_previa="/home/diego/Escritorio/TFM/dataframesCarpetas",
    **sim_kwargs,
):
    """
    Ejecuta simulaciones para cada par (lambda_cc, gamma) y guarda
    cada df_final en un fichero pickle. Si una simulación falla, registra el
    error en el índice y continúa con el resto cuando continuar_si_error=True.

    Parámetros
    ----------
    lambda_values : iterable o None
    gamma_values : iterable o None
        Si parameter_pairs es None, se ejecuta el producto cartesiano de ambas.
    parameter_pairs : iterable[tuple[float, float]] o None
        Lista explicita de pares (lambda_cc, gamma). Si se proporciona, ignora
        lambda_values y gamma_values.
    lanzar_simulacion : callable
        Debe devolver: df_final, df_history
    carpeta : str
        Carpeta donde guardar los ficheros. Si ya existe, se elimina al
        comenzar y se crea de nuevo.
    nombre_indice : str
        Nombre del CSV índice.
    sobrescribir : bool
        Si False, lanza error si el fichero destino ya existe.
    replicas : int
        Número de repeticiones por cada par (lambda_cc, gamma).
    init_id : str o None
        Identificador de la condición inicial. Si es None, se usa el nombre
        del fichero de initial_conditions correspondiente a cada simulación.
    initial_state_files : list[str]
        Lista de ficheros dentro de initial_conditions/ a usar como condición
        inicial. Se ejecuta una simulación por cada elemento de la lista.
    grupo_id : str o None
        Identificador del experimento/grupo.
    continuar_si_error : bool
        Si True, registra el error y sigue. Si False, relanza la excepción.
    extra_metadata : dict o callable o None
        Si es callable, firma esperada:
        extra_metadata(lam, gam, rep_idx, sim_kwargs) -> dict
    append_indice : bool
        Si True, añade los nuevos registros al índice existente en vez de
        sobrescribirlo.
    actualizar_indice_cada : int o None
        Guarda el índice en disco cada N simulaciones registradas. Por defecto
        es 1, es decir, se actualiza tras cada simulación correcta o fallida.
        Usa None o 0 para escribir solo al final.
    **sim_kwargs :
        Resto de argumentos para lanzar_simulacion.

    Devuelve
    --------
    pd.DataFrame
        Índice completo de simulaciones, con filas tanto de simulaciones
        correctas como fallidas.
    """
    if carpeta is None:
        raise ValueError("Debes proporcionar carpeta.")
    if parameter_pairs is None:
        if lambda_values is None or gamma_values is None:
            raise ValueError("Debes proporcionar parameter_pairs o lambda_values y gamma_values.")
        parameter_pairs = list(product(lambda_values, gamma_values))
    parameter_pairs = [(float(lam), float(gam)) for lam, gam in parameter_pairs]
    if actualizar_indice_cada is not None:
        actualizar_indice_cada = int(actualizar_indice_cada)
        if actualizar_indice_cada < 0:
            raise ValueError("actualizar_indice_cada debe ser >= 0, None o 0.")

    carpeta = Path(carpeta_previa) / Path(carpeta)
    if carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_indice = carpeta / nombre_indice

    sim_kwargs = dict(sim_kwargs)
    pintar_resultados = bool(sim_kwargs.pop("pintar", False))
    carpeta_plots = carpeta / "plots"
    if pintar_resultados:
        carpeta_plots.mkdir(parents=True, exist_ok=True)
    initial_state_files = [str(x) for x in initial_state_files]

    registros_nuevos = []
    indice_base = _leer_indice_existente(ruta_indice) if append_indice else pd.DataFrame(columns=_columnas_indice_v3())
    indice_actual = _preparar_indice_para_guardar(indice_base)
    total = 0

    def registrar_y_actualizar_indice(registro, forzar=False):
        nonlocal indice_actual
        registros_nuevos.append(registro)
        indice_actual = pd.concat(
            [indice_actual, pd.DataFrame([registro], columns=_columnas_indice_v3())],
            ignore_index=True,
        )

        debe_guardar = (
            forzar
            or (
                actualizar_indice_cada is not None
                and actualizar_indice_cada > 0
                and len(registros_nuevos) % actualizar_indice_cada == 0
            )
        )
        if debe_guardar:
            indice_actual = _guardar_indice_atomico(indice_actual, ruta_indice)

        return indice_actual

    for initial_state_name in initial_state_files:
        for lam, gam in parameter_pairs:
            for rep_idx in range(int(replicas)):
                total += 1
                sim_id = _generar_sim_id(lam, gam, rep_idx)
                nombre_fichero = _nombre_fichero_df_final(lam, gam, sim_id)
                ruta_fichero = carpeta / nombre_fichero
                init_id_actual = str(init_id) if init_id is not None else str(initial_state_name)
                sim_kwargs_actual = dict(sim_kwargs)
                sim_kwargs_actual["initial_state_file"] = f"initial_conditions/{initial_state_name}"

                if ruta_fichero.exists() and not sobrescribir:
                    raise FileExistsError(
                        f"Ya existe {ruta_fichero}. Usa sobrescribir=True si quieres reemplazarlo."
                    )

                if callable(extra_metadata):
                    metadata_actual = extra_metadata(lam, gam, rep_idx, sim_kwargs_actual)
                else:
                    metadata_actual = extra_metadata

                print(
                    f"[{total}] init={initial_state_name}, lambda={lam}, gamma={gam}, replica={rep_idx}, sim_id={sim_id}"
                )

                registro_base = {
                    "sim_id": sim_id,
                    "lambda_cc": float(lam),
                    "gamma": float(gam),
                    "replica": int(rep_idx),
                    "init_id": init_id_actual,
                    "grupo_id": grupo_id,
                    "filepath": pd.NA,
                    "n_filas": pd.NA,
                    "columnas": pd.NA,
                    "sim_kwargs": _safe_json(sim_kwargs_actual),
                    "extra_metadata": _safe_json(metadata_actual),
                    "status": pd.NA,
                    "error_msg": pd.NA,
                    "traceback": pd.NA,
                }

                try:
                    df_final, df_history = lanzar_simulacion(
                        lambda_cc=lam,
                        gamma=gam,
                        **sim_kwargs_actual,
                    )
                    es_valido, motivo = _df_tifosi_valido(df_final)
                    if not es_valido:
                        raise ValueError(f"salida inválida: {motivo}")

                    df_final.to_pickle(ruta_fichero)
                    if pintar_resultados:
                        try:
                            ruta_png = carpeta_plots / _nombre_fichero_plot_final(lam, gam, sim_id)
                            _plot_dataframe_cells(
                                df_final,
                                ruta_png,
                                titulo=f"lambda={lam}, gamma={gam}, sim={sim_id}",
                            )
                        except Exception as exc_plot:
                            print(f"  WARNING plot en {sim_id}: {exc_plot}")

                    registro = dict(registro_base)
                    registro.update(
                        {
                            "filepath": str(ruta_fichero),
                            "n_filas": int(len(df_final)),
                            "columnas": _safe_json(list(df_final.columns)),
                            "status": "ok",
                        }
                    )
                    registrar_y_actualizar_indice(registro)

                except Exception as exc:
                    tb = traceback.format_exc()
                    print(f"  ERROR en {sim_id}: {exc}")

                    registro = dict(registro_base)
                    registro.update(
                        {
                            "status": "error",
                            "error_msg": str(exc),
                            "traceback": tb,
                        }
                    )
                    registrar_y_actualizar_indice(registro, forzar=not continuar_si_error)

                    if not continuar_si_error:
                        raise

    return _guardar_indice_atomico(indice_actual, ruta_indice)


def leer_dataframes_barrido(
    carpeta,
    nombre_indice="indice_simulaciones.csv",
    lambda_cc=None,
    gamma=None,
    sim_id=None,
    init_id=None,
    grupo_id=None,
    replica=None,
    status=None,
    cargar_todos=False,
    devolver_indice=False,
    atol=1e-12,
    carpeta_previa="/home/diego/Escritorio/TFM/dataframesCarpetas"
):
    """
    Lee los dataframes guardados usando el índice de la versión 3.

    Reglas de uso
    -------------
    - Sin filtros y cargar_todos=False: devuelve el índice completo.
    - Con devolver_indice=True: devuelve el subíndice filtrado.
    - Si la selección deja una sola simulación 'ok': devuelve el dataframe.
    - Si la selección deja varias y cargar_todos=True: devuelve dict {sim_id: df}.
    - Si la selección contiene simulaciones con status='error' y no usas
      devolver_indice=True, esas simulaciones no se pueden cargar como pickle.
    """
    path_carpeta = Path(carpeta_previa) / Path(carpeta)
    ruta_indice = path_carpeta / nombre_indice

    if not ruta_indice.exists():
        raise FileNotFoundError(
            f"No existe el índice {ruta_indice}. Primero ejecuta guardar_dataframes_barrido(...)."
        )

    indice = pd.read_csv(ruta_indice)

    filtros_activos = any(
        x is not None
        for x in [lambda_cc, gamma, sim_id, init_id, grupo_id, replica, status]
    )

    if not filtros_activos and not cargar_todos:
        return indice

    seleccion = indice.copy()

    if sim_id is not None:
        seleccion = seleccion.loc[seleccion["sim_id"].astype(str) == str(sim_id)]

    if lambda_cc is not None:
        seleccion = seleccion.loc[
            np.isclose(
                seleccion["lambda_cc"].to_numpy(), float(lambda_cc), atol=atol, rtol=0
            )
        ]

    if gamma is not None:
        seleccion = seleccion.loc[
            np.isclose(
                seleccion["gamma"].to_numpy(), float(gamma), atol=atol, rtol=0
            )
        ]

    if init_id is not None:
        seleccion = seleccion.loc[seleccion["init_id"].astype(str) == str(init_id)]

    if grupo_id is not None:
        seleccion = seleccion.loc[seleccion["grupo_id"].astype(str) == str(grupo_id)]

    if replica is not None:
        seleccion = seleccion.loc[seleccion["replica"] == int(replica)]

    if status is not None:
        seleccion = seleccion.loc[seleccion["status"].astype(str) == str(status)]

    seleccion = seleccion.reset_index(drop=True)

    if len(seleccion) == 0:
        raise ValueError("No se encontró ninguna simulación con esos filtros.")

    if devolver_indice:
        return seleccion

    seleccion_ok = seleccion.loc[seleccion["status"].astype(str) == "ok"].reset_index(drop=True)

    if len(seleccion_ok) == 0:
        raise ValueError(
            "La selección no contiene simulaciones con status='ok'. "
            "Usa devolver_indice=True para inspeccionar los errores registrados."
        )

    if len(seleccion_ok) == 1 and not cargar_todos:
        ruta_fichero = path_carpeta / Path(seleccion_ok.iloc[0]["filepath"]).name
        return pd.read_pickle(ruta_fichero)

    if cargar_todos:
        dataframes = {}
        for _, fila in seleccion_ok.iterrows():
            ruta_fichero = path_carpeta / Path(fila["filepath"]).name
            dataframes[str(fila["sim_id"])] = pd.read_pickle(ruta_fichero)
        return dataframes

    raise ValueError(
        "La selección devuelve varias simulaciones con status='ok'. "
        "Usa sim_id=..., o bien cargar_todos=True, o devolver_indice=True "
        "para inspeccionar primero el subíndice."
    )

def cargar_barrido_v4(carpeta, nombre_indice="indice_simulaciones.csv", **filtros):
    filtros.setdefault("status", "ok")
    indice = leer_dataframes_barrido(
        carpeta=carpeta,
        nombre_indice=nombre_indice,
        devolver_indice=True,
        **filtros,
    )
    dfs = leer_dataframes_barrido(
        carpeta=carpeta,
        nombre_indice=nombre_indice,
        cargar_todos=True,
        **filtros,
    )
    indice = indice[indice["sim_id"].astype(str).isin(dfs.keys())].reset_index(drop=True)
    return dfs, indice


def _df_tifosi_valido(df):
    if df is None or len(df) == 0:
        return False, "df vacío"

    cols_necesarias = ["area", "center_x", "center_y", "perimeter", "vertices"]
    for c in cols_necesarias:
        if c not in df.columns:
            return False, f"falta columna {c}"

    # columnas numéricas principales
    for c in ["area", "center_x", "center_y", "perimeter"]:
        vals = df[c].to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            return False, f"NaN/Inf en columna {c}"

    # vértices
    for i, verts in enumerate(df["vertices"]):
        if not isinstance(verts, (list, tuple)) or len(verts) == 0:
            return False, f"vertices vacíos en fila {i}"
        for x, y in verts:
            if not np.isfinite(x) or not np.isfinite(y):
                return False, f"NaN/Inf en vertices fila {i}"

    return True, ""
