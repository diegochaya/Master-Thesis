from pathlib import Path

import matplotlib.pyplot as plt

from dataframes_v4 import leer_dataframes_barrido
from Preprocesado import extraer_features


CARPETA = "dataframesBarrido"
EXCLUIR_BORDE = True
N_CENTER_CELLS = 196
CARPETA_SALIDA = Path("PreliminaryExperiments")


dfs = leer_dataframes_barrido(
    carpeta=CARPETA,
    cargar_todos=True,
    status="ok",
)

datos = extraer_features(
    dfs,
    excluir_borde=EXCLUIR_BORDE,
    n_center_cells=N_CENTER_CELLS,
)

if datos.empty:
    raise ValueError(f"No hay datos disponibles en {CARPETA!r}.")

corr = datos.corr(numeric_only=True).fillna(0.0)

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
output_png = CARPETA_SALIDA / f"matriz_correlacion_{CARPETA}.png"

fig, ax = plt.subplots(figsize=(16, 14), constrained_layout=True)
imagen = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_title(f"Matriz de correlacion: {CARPETA}")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticklabels(corr.index, fontsize=7)

colorbar = fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
colorbar.set_label("Correlacion de Pearson")

fig.savefig(output_png, dpi=180, bbox_inches="tight")
plt.close(fig)

print(corr.round(3).to_string())
print(f"\nGuardada matriz de correlacion en: {output_png}")
