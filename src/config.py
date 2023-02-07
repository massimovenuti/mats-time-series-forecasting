EPOCHS = {
    "electricity": {
        "multivariate":[1000, 500],
        "univariate":[1000, 500]
    },
    "exchange": {
        "multivariate":[1000, 500],
        "univariate":[1000, 500]
    },
    "ett_h1": {
        "multivariate":[1000, 500],
        "univariate":[1000, 500]
    },
    "ett_h2": {
        "multivariate":[1000, 100],
        "univariate":[1000, 100]
    },
    "ett_m1": {
        "multivariate":[1000, 500],
        "univariate":[1000, 500]
    },
    "ett_m2": {
        "multivariate":[1000, 100],
        "univariate":[1000, 100]
    },
    "traffic": {
        "multivariate":[1000, 200],
        "univariate":[1000, 200]
    },
    "weather": {
        "multivariate":[1000, 100],
        "univariate":[1000, 100]
    },
    "ili": {
        "multivariate":[1000, 500],
        "univariate":[1000, 1000]
    },
}

DIM_T_1 = 192  # Longeur d'une serie chronologique stage 1
DIM_T_2 = 96  # Longeur d'une serie chronologique stage 2
DIMS_H = [96, 192, 336, 720]  # Nombre de valeur à prédire pour une serie chronologique

DIM_T_1_ILI = 60  # Longeur d'une serie chronologique stage 1 (ILI)
DIM_T_2_ILI = 36  # Longeur d'une serie chronologique stage 2 (ILI)
DIMS_H_ILI = [24, 36, 48, 60]

BATCH_SIZE = 64
DIM_E = 64  # Nombre de variable d'une serie chronologique apres encodeur ( taille couche sortie encodeur)
SIZE_M = 16  # Taille de la banque de mémoire

RUNS_DIR = "/tempory/mv/mats/runs/"
STATES_DIR = "/tempory/mv/mats/states/"
RESULTS_DIR = "results/"
DATA_DIR = "/tempory/mv/mats/data/"
