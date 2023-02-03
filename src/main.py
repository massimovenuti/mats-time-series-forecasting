import datasets
import torch
import mats
import pandas as pd
from pathlib import Path
from torch.utils import tensorboard as tb
import time

# torch.cuda.set_per_process_memory_fraction(1., 0)

DIM_T_1 = 192  # Longeur d'une serie chronologique stage 1
DIM_T_2 = 96  # Longeur d'une serie chronologique stage 2
DIMS_H = [96, 192, 336, 720]  # Nombre de valeur à prédire pour une serie chronologique

DIM_T_1_ILI = 60  # Longeur d'une serie chronologique stage 1 (ILI)
DIM_T_2_ILI = 36  # Longeur d'une serie chronologique stage 2 (ILI)
DIMS_H_ILI = [24, 36, 48, 60]

BATCH_SIZE = 64
DIM_E = 64  # Nombre de variable d'une serie chronologique apres encodeur ( taille couche sortie encodeur)
SIZE_M = 16  # Taille de la banque de mémoire

RUNS_DIR = "runs/"
STATES_DIR = "states/"
RESULTS_DIR = "results/"
DATA_DIR = "data/"


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_paths = {
        "electricity": Path(DATA_DIR, "electricity.txt"),
        "exchange": Path(DATA_DIR, "exchange.txt"),
        "ett": Path(DATA_DIR, "ett_h1.txt"),
        "ett": Path(DATA_DIR, "ett_h2.txt"),
        "ett": Path(DATA_DIR, "ett_m1.txt"),
        "ett": Path(DATA_DIR, "ett_m2.txt"),
        "traffic": Path(DATA_DIR, "traffic.txt"),
        "weather": Path(DATA_DIR, "weather.csv"),
        "ili": Path(DATA_DIR, "ili.csv"),
    }
    
    all_mse_train = {}
    all_mae_train = {}
    all_mse_test = {}
    all_mae_test = {}

    for dataset, path in dataset_paths.items():
        print(f"##### Dataset : {dataset} #####")

        all_mse_train[dataset] = {}
        all_mae_train[dataset] = {}
        all_mse_test[dataset] = {}
        all_mae_test[dataset] = {}

        for i in range(len(DIMS_H)):
            dim_h = DIMS_H[i] if dataset != "ili" else DIMS_H_ILI[i]
            print(f"### H = {dim_h} ###")

            writer = tb.SummaryWriter(Path(RUNS_DIR, f"mats_{dataset}_h_{dim_h}_{time.asctime()}"))
            save_path = Path(STATES_DIR, dataset, f"mats_h_{dim_h}.pkl")
            save_path.parent.mkdir(exist_ok=True, parents=True)

            train_loader_1, test_loader_1 = datasets.get_loaders(
                dataset=dataset,
                path=path,
                dim_t=DIM_T_1 if dataset != "ili" else DIM_T_1_ILI,
            )

            train_loader_2, test_loader_2 = datasets.get_loaders(
                dataset=dataset,
                path=path,
                dim_h=dim_h,
                dim_t=DIM_T_2 if dataset != "ili" else DIM_T_2_ILI,
            )

            if save_path.is_file():
                with save_path.open("rb") as fp:
                    model = torch.load(fp)
            else:
                dim_c = train_loader_1.dataset.data.shape[1]
                model = mats.MATS(dim_c, SIZE_M, DIM_E)

            model = model.to(device)

            model.fit(
                train_loader_1=train_loader_1,
                val_loader_1=test_loader_1,
                train_loader_2=train_loader_2,
                val_loader_2=test_loader_2,
                epochs_1=1000,
                epochs_2=500,
                save_path=save_path,
                writer=writer,
            )

            mse, mae = model.evaluate(train_loader_2)
            print(f"[train] MSE={mse:.2f}, MAE={mae:.2f}")

            all_mse_train[dataset][dim_h] = mse
            all_mae_train[dataset][dim_h] = mae

            mse, mae = model.evaluate(test_loader_2)
            print(f"[test] MSE={mse:.2f}, MAE={mae:.2f}")

            all_mse_test[dataset][dim_h] = mse
            all_mae_test[dataset][dim_h] = mae

            Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
            df_mse_train = pd.DataFrame.from_dict(all_mse_train).to_csv(
                Path(RESULTS_DIR, "mse_train.csv", index_label="H")
            )
            df_mae_train = pd.DataFrame.from_dict(all_mae_train).to_csv(
                Path(RESULTS_DIR, "mae_train.csv", index_label="H")
            )
            df_mse_test = pd.DataFrame.from_dict(all_mse_test).to_csv(
                Path(RESULTS_DIR, "mse_test.csv", index_label="H")
            )
            df_mae_test = pd.DataFrame.from_dict(all_mae_test).to_csv(
                Path(RESULTS_DIR, "mae_test.csv", index_label="H")
            )
