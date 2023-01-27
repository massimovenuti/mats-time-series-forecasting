import datasets
from torch.utils.data import DataLoader
import torch
import mats
from tqdm import tqdm
from torch import nn
import numpy as np
from pathlib import Path
from torch.utils import tensorboard as tb
import time
from tqdm import tqdm

# torch.cuda.set_per_process_memory_fraction(1., 0)


BATCH_SIZE = 64
DIM_T_1 = 192  # Longeur d'une serie chronologique stage 1
DIM_T_2 = 96  # Longeur d'une serie chronologique stage 2
# DIM_H = [96, 192, 336, 720]  # Nombre de valeur à prédire pour une serie chronologique
DIM_H = 96  # Nombre de valeur à prédire pour une serie chronologique
DIM_E = 64  # Nombre de variable d'une serie chronologique apres encodeur ( taille couche sortie encodeur)
SIZE_M = 16  # Taille de la banque de mémoire

STATES_DIR = "states/"


if __name__ == "__main__":
    writer = tb.SummaryWriter(f"runs/mats-{time.asctime()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = Path(STATES_DIR, "mats.pkl")
    save_path.parent.mkdir(exist_ok=True)

    train_loader_1, val_loader_1, _ = datasets.get_loaders(
        dataset="exchange",
        path="data/Exchange/exchange_rate.txt",
        dim_t=DIM_T_1,
    )

    train_loader_2, val_loader_2, test_loader_2 = datasets.get_loaders(
        dataset="exchange",
        path="data/Exchange/exchange_rate.txt",
        dim_t=DIM_T_2,
        dim_h=DIM_H,
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
        val_loader_1=val_loader_1,
        train_loader_2=train_loader_2,
        val_loader_2=val_loader_2,
        epochs_1=1000,
        epochs_2=500,
        save_path=save_path,
        writer=writer,
        device=device,
    )

    list_mse, list_mae = model.evaluate(train_loader_2, device)
    mse = np.array(list_mse).mean()
    mae = np.array(list_mae).mean()
    print(f"[TRAIN] \t MSE : {mse:.2f}")
    print(f"[TRAIN] \t MAE : {mae:.2f}")
    print("=======")

    list_mse, list_mae = model.evaluate(val_loader_2, device)
    mse = np.array(list_mse).mean()
    mae = np.array(list_mae).mean()
    print(f"[VAL] \t MSE : {mse:.2f}")
    print(f"[VAL] \t MAE : {mae:.2f}")
    print("=======")

    list_mse, list_mae = model.evaluate(test_loader_2, device)
    mse = np.array(list_mse).mean()
    mae = np.array(list_mae).mean()
    print(f"[TEST] \t MSE : {mse:.2f}")
    print(f"[TEST] \t MAE : {mae:.2f}")
