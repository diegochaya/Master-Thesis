# TFM-public

Research code for an inverse problem on epithelial vertex-model simulations: infer the mechanical parameters `lambda_cc` and `gamma` from the final tissue morphology produced by TiFoSi.

This GitHub version contains the code for:
- launching and parsing TiFoSi simulations
- building sweep datasets from simulation outputs
- preprocessing and extracting morphological features
- running preliminary experiments on feature-space structure
- training predictive models for parameter inference

## Repository Structure

### Simulation launch
- `Lanzar1Simulacion.py`: minimal example that launches one TiFoSi simulation and plots basic summaries.
- `LanzarGuardarSimulaciones.py`: launches a parameter sweep and stores one dataframe per simulation.
- `Lcrear1PosicionInicial.py`: generates a centered initial condition from a larger simulated tissue.
- `lanzador_tifosi.py`: interface layer to TiFoSi, XML update logic, simulation execution, output parsing, and plotting.

### Dataset building
- `dataframes_v4.py`: utilities to generate parameter grids, validate simulation outputs, save sweep dataframes, and load them later through an index CSV.

### Preprocessing and features
- `Preprocesado.py`: cleaning rules, border-cell handling, centered-cell selection, elongation computation, and aggregated feature extraction.
- `metricas_resultados.py`: regression metrics and plotting helpers shared by the predictive models.

### Preliminary experiments
- `experimento1_repetitividad.py`: repeatability under identical simulation conditions.
- `experimento2_separabilidad.py`: separability between parameter groups using PCA, silhouette score, and nearest-centroid diagnostics.
- `experimento3_condiciones_iniciales.py`: effect of initial condition on the final feature vectors.
- `correlaciones_features.py`: correlation matrix for the aggregated feature dataset.
- `pca.py`: PCA of the main sweep feature matrix.

### Predictive models
- `random_forest_parametros.py`: Random Forest regressor for joint prediction of `lambda_cc` and `gamma`.
- `red_neuronal_parametros.py`: MLP regressor on aggregated feature vectors.
- `red_neuronal_classification.py`: maximum-likelihood MLP that predicts both mean and covariance over the two target parameters.
- `gnn_parametros.py`: GraphSAGE-based regressor operating on cell-neighbour graphs.

### Assets
- `initial_conditions/`: `.dat` files used as initial tissues, plus reference plots.
- `PreliminaryExperiments/`: exploratory outputs and intermediate analysis artifacts.
- `outputs/`: model outputs and prediction artifacts when they are kept in the public version.

## Dependency Setup

### Python
This repository is written as a research-code layout, not as a packaged Python module. Run the scripts from the repository root.

Recommended:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### TiFoSi
TiFoSi is not pushed in this GitHub version.

Important:
- `lanzador_tifosi.py` assumes a local `TIFOSI/` directory by default.
- `Lcrear1PosicionInicial.py` also assumes a local `TIFOSI/` directory inside the repo root.
- The TiFoSi copy used during the project had a user-made change in `TIFOSI/src/poblacio.cpp` for the repeatability experiments so the seed was not fixed:

```cpp
llavor = -(long)clock();
ran3(&llavor);
```

- That change was already made by the project author in the local TiFoSi setup used for the experiments.
- If you want to reproduce the simulation-launch scripts, you need your own TiFoSi copy and you should apply the same change.
- If you place TiFoSi elsewhere, adjust the corresponding path assumptions before running simulation-launch scripts.

Scripts that only consume precomputed dataframes can still be read and adapted without a local TiFoSi installation.

## Recommended Reading Order

1. `Lanzar1Simulacion.py`
2. `lanzador_tifosi.py`
3. `dataframes_v4.py`
4. `Preprocesado.py`
5. `experimento1_repetitividad.py`
6. `experimento2_separabilidad.py`
7. `experimento3_condiciones_iniciales.py`
8. `correlaciones_features.py`
9. `pca.py`
10. `random_forest_parametros.py`
11. `red_neuronal_parametros.py`
12. `red_neuronal_classification.py`
13. `gnn_parametros.py`

## What Each Main Script Does

### Simulation and dataset scripts
- `Lanzar1Simulacion.py`
  - Input: one fixed parameter pair and one initial condition.
  - Output: one final tissue dataframe, a time-history dataframe, and basic plots.
  - Writes plots under `TIFOSI/plots/`.

- `LanzarGuardarSimulaciones.py`
  - Input: a sweep definition over `lambda_cc` and `gamma`.
  - Output: many final-tissue dataframes and one index CSV.
  - Writes under `dataframesCarpetas/<carpeta>/`.

- `Lcrear1PosicionInicial.py`
  - Input: one TiFoSi run used to build a reusable centered tissue.
  - Output: a `.dat` initial condition file and a reference plot.
  - Writes under `initial_conditions/` and `initial_conditions/plots/`.

- `dataframes_v4.py`
  - Provides the main persistence layer for simulation campaigns.
  - `guardar_dataframes_barrido(...)` stores one pickle per simulation plus an index CSV.
  - `leer_dataframes_barrido(...)` reloads one simulation, many simulations, or only the index metadata.

### Feature and exploratory scripts
- `Preprocesado.py`
  - Input: TiFoSi-style per-cell dataframes.
  - Output: cleaned cell tables and one aggregated feature vector per simulation.

- `experimento1_repetitividad.py`
  - Input: repeated simulations for four fixed parameter groups.
  - Output: feature summaries, group centroids, distances, and variability tables.
  - Writes under `PreliminaryExperiments/experimento1_repetitividad/`.

- `experimento2_separabilidad.py`
  - Input: the feature table produced by `experimento1_repetitividad.py`.
  - Output: PCA projections, centroid-distance ratios, silhouette summaries, and leave-one-out nearest-centroid classification diagnostics.
  - Writes under `PreliminaryExperiments/experimento2_separabilidad/`.

- `experimento3_condiciones_iniciales.py`
  - Input: repeated simulations at fixed parameters with different initial conditions.
  - Output: PCA projections, centroid-distance ratios, silhouette summaries, and leave-one-out nearest-centroid classification diagnostics.
  - Writes under `PreliminaryExperiments/experimento3_condiciones_iniciales/`.

- `correlaciones_features.py`
  - Input: aggregated feature matrix from the main sweep.
  - Output: correlation matrix plot.
  - Writes under `PreliminaryExperiments/`.

- `pca.py`
  - Input: aggregated feature matrix from the main sweep.
  - Output: PCA loadings table, explained-variance table, and 2D PCA plot.
  - Writes under `PreliminaryExperiments/`.

### Predictive-model scripts
- `random_forest_parametros.py`
  - Input: aggregated feature vectors from a sweep dataset.
  - Output: fitted Random Forest, test predictions, metrics, and feature importances.
  - Writes under `outputs/random_forest_<carpetas>/`.

- `red_neuronal_parametros.py`
  - Input: aggregated feature vectors from a sweep dataset.
  - Output: fitted MLP, training history, test predictions, and metrics.
  - Writes under `outputs/red_neuronal_<carpetas>/`.

- `red_neuronal_classification.py`
  - Input: aggregated feature vectors from a sweep dataset.
  - Output: fitted probabilistic MLP, predictive means and covariances, ellipse-based plots, and uncertainty metrics.
  - Writes under `outputs/red_neuronal_classfication_<carpeta>/`.

- `gnn_parametros.py`
  - Input: graph representation of each tissue, built from cell-neighbour relations.
  - Output: fitted GraphSAGE regressor, training history, and test predictions.
  - Writes under `outputs/gnn_<carpetas>/`.

## Data Expectations

Large generated datasets under `dataframesCarpetas/` are not part of the GitHub repository and should not be expected in a clean clone.

Several scripts expect precomputed simulation outputs stored in folders such as:

- `dataframesCarpetas/dataframesBarrido/`
- `dataframesCarpetas/dataframesComprobaciones3/`
- `dataframesCarpetas/dataframesComprobacionesIC/`

These folders are produced by the simulation-launch pipeline and are intentionally excluded from version control.

If someone needs `dataframesCarpetas/`, they should contact the project author directly. The folder is large, around 6.7 GB, so it is not distributed through GitHub.

As a consequence:
- preliminary experiment scripts require the corresponding precomputed sweep or repetition datasets
- predictive-model scripts require precomputed simulation dataframes before training
- only a subset of the repository is runnable from a clean checkout without first generating or restoring those datasets

In practice:
- do not try to fetch `dataframesCarpetas/` from the repository, because it is not distributed here
- do not expect `TIFOSI/` to be present in the GitHub version
- for full simulation reproducibility, obtain TiFoSi separately and apply the local seed change described above

## Public-Repo Limitations

- No simulation datasets are included.
- The repository keeps many original research-code assumptions about local relative paths and expected folder names.

