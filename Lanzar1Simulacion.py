from lanzador_tifosi import lanzar_simulacion
import matplotlib.pyplot as plt
import numpy as np

df_final, df_history = lanzar_simulacion(
    lambda_cc = 0.04,
    gamma = 0.05,
    speed = 1,
    ncellsx = 14,
    ncellsy = 14,
    initial_state_file = "initial_conditions/hexagonal_2000_1.dat",
    stage1_duration = 100,
    stage1_intermediate = 100,
    stage2_duration= 500,
    stage2_intermediate = 500,pintar=True)
print("Columnas df_final:", df_final.columns.tolist())
df_history["mean_shapeindex"]=df_history["mean_perimeter"] / (np.sqrt(df_history["mean_area"]))
"""


print("Columnas df_history:", df_history.columns.tolist())

# =========================================================
# 1) Evolución temporal de medias usando df_history
# =========================================================

plt.figure(figsize=(8, 5))
#plt.plot(df_history["frame"], df_history["mean_area"], marker="o", label="mean_area")
#plt.plot(df_history["frame"], df_history["mean_perimeter"], marker="o", label="mean_perimeter")
plt.plot(df_history["frame"], df_history["mean_shapeindex"], marker="o", label="mean_shapeindex")
plt.xlabel("frame_global")
plt.ylabel("valor")
plt.title("Evolución temporal de área y perímetro medios")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("TIFOSI/plots/media_area_y_perimeter_vs_tiempo.png", dpi=300)
plt.close()

# =========================================================
# 2) Distribución final de áreas usando df_final
# =========================================================

plt.figure(figsize=(8, 5))
plt.hist(df_final["area"], bins=30, edgecolor="black")
plt.xlabel("Área")
plt.ylabel("Frecuencia")
plt.title("Distribución final de áreas")
plt.tight_layout()
plt.savefig("TIFOSI/plots/distribucion_areas_final.png", dpi=300)
plt.close()

# =========================================================
# 3) Distribución final de perímetros usando df_final
# =========================================================

plt.figure(figsize=(8, 5))
plt.hist(df_final["perimeter"], bins=30, edgecolor="black")
plt.xlabel("Perímetro")
plt.ylabel("Frecuencia")
plt.title("Distribución final de perímetros")
plt.tight_layout()
plt.savefig("TIFOSI/plots/distribucion_perimetros_final.png", dpi=300)
plt.close()

print("Figuras guardadas en la carpeta plots/")"""
