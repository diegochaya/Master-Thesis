from dataframes_v4 import generar_parameter_pairs_triangulo, guardar_dataframes_barrido
import numpy as np

carpeta="yhjdatafrridos"

parameter_pairs = generar_parameter_pairs_triangulo(
    lambda_values=np.linspace(-0.15, 0.15, 100),
    gamma_values=np.linspace(0.03, 0.14, 100),
    gamma_max_intercept=0.120,
    gamma_max_slope=0.50,
    complemento=False,
)

inndice = guardar_dataframes_barrido(
    parameter_pairs=parameter_pairs,
    carpeta=carpeta,
    replicas=1,
    initial_state_files=["hexagonal_2000_1.dat"],
    pintar=True,
    speed=3.08,
    stage1_duration=1,
    stage1_intermediate=1,
    stage2_duration=500,
    stage2_intermediate=500,
)

"""
carpeta="dataframesComprobacionesIC"
indice = guardar_dataframes_barrido(
    lambda_values=[-0.05],
    gamma_values=[0.07],
    carpeta=carpeta,
    replicas=30,
    initial_state_files=["hexagonal_perfect_14x14.dat","hexagonal_2000_1.dat","hexagonal_1000_1.dat","hexagonal_1000_2.dat"],
    pintar=True,
    speed=3.08,
    stage1_duration=1,
    stage1_intermediate=1,
    stage2_duration=500,
    stage2_intermediate=500,
)


indice = guardar_dataframes_barrido(
    lambda_values=[-0.1,0.0],
    gamma_values=[0.05, 0.08],
    carpeta=carpeta,
    replicas=30,
    initial_state_files=["hexagonal_2000_1.dat"],
    pintar=True,
    speed=3.08,
    stage1_duration=1,
    stage1_intermediate=1,
    stage2_duration=500,
    stage2_intermediate=500,
)

carpeta="dataframesReal2_complemento"
parameter_pairs = generar_parameter_pairs_triangulo(
    lambda_values=np.linspace(-0.15, 0.15, 30),
    gamma_values=np.linspace(0.03, 0.14, 30),
    complemento=True,
)

indice = guardar_dataframes_barrido(
    parameter_pairs=parameter_pairs,
    carpeta=carpeta,
    replicas=1,
    initial_state_files=["hexagonal_2000_1.dat"],
    pintar=True,
    speed=3.08,
    stage1_duration=1,
    stage1_intermediate=1,
    stage2_duration=500,
    stage2_intermediate=500,
)"""
