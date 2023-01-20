import datasets
from torch.utils.data import DataLoader
import torch
import mats
from tqdm import tqdm
from torch import nn
import numpy as np
from pathlib import Path


def train_stage_1(dataloader, state, epochs, device, save_path):

    state.encoder.train()
    state.decoder.train()
    state.memory_bank.train()
    state.discriminator.train()
    state.predictor.train()

    criterion_edm = mats.EDMLoss(state.decoder)
    criterion_discriminator = mats.DiscriminatorLoss()

    iteration = state.stage_1_iteration

    for epoch in range(state.stage_1_epoch, epochs):
        for e, (X, _) in enumerate(dataloader):
            state.optim_edm.zero_grad()
            state.optim_discriminator.zero_grad()

            X = torch.movedim(X, 1, 2).to(device)  # BATCH_SIZE * DIM_C * DIM_T
            H = state.encoder(X).to(device)  # BATCH_SIZE * DIM_D * DIM_T2
            C = state.memory_bank(H).to(device)  # BATCH_SIZE * DIM_M * DIM_T2

            # BATCH_SIZE X DIM_D X DIM_T2
            Hhat = state.memory_bank.reconstruct(C).to(device)

            Xhat = state.decoder(Hhat).to(device)  # BATCH_SIZE * DIM_C * DIM_T

            if e % 2 == 0:
                # (4)
                Dhat = state.discriminator(Xhat).to(device)  #  BATCH_SIZE * 1 * DIM_T2
                loss = criterion_edm(Xhat, X, H, state.memory_bank.units, Dhat)
                loss.backward()
                state.optim_edm.step()
            else:
                # (3)
                D = state.discriminator(X).to(device)  # BATCH_SIZE * 1 * DIM_T2
                Dhat = state.discriminator(Xhat).to(device)  #  BATCH_SIZE * 1 * DIM_T2
                loss = criterion_discriminator(Dhat, D)
                loss.backward()
                state.optim_discriminator.step()

            # TODO : use tensorboard
            if e % 20 == 0 or e % 20 == 1:
                print(
                    f"[STAGE 1]"
                    f"[{epoch}/{epochs}][{e}/{len(dataloader)}]"
                    f"[{'EDM' if e%2 == 0 else 'D'}]\t"
                    f"Loss : {loss:.2f}"
                )

            iteration = iteration + 1

        state.stage_1_iteration = iteration
        state.stage_1_epoch = epoch + 1

        with save_path.open("wb") as fp:
            torch.save(state, fp)


def train_stage_2(dataloader, state, dim_h, epochs, device, save_path):
    state.encoder.train()
    state.decoder.train()
    state.memory_bank.train()
    state.discriminator.train()
    state.predictor.train()

    criterion_predictor = nn.BCELoss()

    iteration = state.stage_2_iteration

    for epoch in range(state.stage_2_epoch, epochs):
        for e, (X, y) in enumerate(dataloader):
            state.optim_predictor.zero_grad()

            # (1)
            # CNN waits dim N * C_in * L
            X = torch.movedim(X, 1, 2).to(device)  # BATCH_SIZE * DIM_C * DIM_T
            H = state.encoder(X).to(device)  # BATCH_SIZE * DIM_D * DIM_T2
            C = state.memory_bank(H).to(device)  # BATCH_SIZE * DIM_M * DIM_T2

            # (5)
            dim_t = X.shape[2]
            dim_t2 = C.shape[2]
            dim_h2 = np.ceil(dim_t2 * dim_h / dim_t).astype(int)

            # TODO : should we do teacher forcing only ?
            # See Curriculum Learning
            # LSTM waits dim L * N * H_in
            C = C.movedim((0, 1, 2), (1, 2, 0))  # DIM_T2 * BATCH_SIZE * DIM_M
            # DIM_T2 * BATCH_SIZE * DIM_M
            pred_output, (last_hidden, last_cell) = state.predictor(C)
            prediction = state.predictor.decode(
                pred_output
            )  # DIM_T2 * BATCH_SIZE * DIM_M

            all_predictions = [prediction]
            for _ in range(dim_h2):
                pred_output, (last_hidden, last_cell) = state.predictor(
                    prediction[-1].unsqueeze(0), (last_hidden, last_cell)
                )
                prediction = state.predictor.decode(pred_output)
                all_predictions.append(prediction)

            # (DIM_T2 + DIM_H2) * BATCH_SIZE * DIM_M
            Chat = torch.vstack(all_predictions)

            # (6)
            y = torch.movedim(y, 1, 2).to(device)  # BATCH_SIZE * DIM_C * DIM_T
            X_gt = torch.cat((X, y), dim=2)  # BATCH_SIZE * DIM_C * (DIM_T + DIM_H)
            H_gt = state.encoder(X_gt).to(
                device
            )  # BATCH_SIZE * DIM_D * (DIM_T2 + DIM_H2)
            # BATCH_SIZE * DIM_M * (DIM_T2 + DIM_H2)
            C_gt = state.memory_bank(H_gt).to(device)
            # (DIM_T2 + DIM_H2) * BATCH_SIZE * DIM_M
            C_gt = C_gt.movedim((0, 1, 2), (1, 2, 0)).to(device)

            # (7)
            loss = criterion_predictor(Chat, C_gt)
            loss.backward()
            state.optim_predictor.step()

            # TODO : use tensorboard
            if e % 20 == 0:
                print(f"[{epoch}/{epochs}][{e}/{len(dataloader)}]\t Loss : {loss:.2f}")

            iteration = iteration + 1

        state.stage_2_iteration = iteration
        state.stage_2_epoch = epoch + 1

        with save_path.open("wb") as fp:
            torch.save(state, fp)


def inference(X, state, dim_h, device):
    state.encoder.eval()
    state.decoder.eval()
    state.memory_bank.eval()
    state.discriminator.eval()
    state.predictor.eval()

    with torch.no_grad():

        # (1)
        # CNN waits dim N * C_in * L
        X = torch.movedim(X, 1, 2).to(device)  # BATCH_SIZE * DIM_C * DIM_T
        H = state.encoder(X).to(device)  # BATCH_SIZE * DIM_D * DIM_T2
        C = state.memory_bank(H).to(device)  # BATCH_SIZE * DIM_M * DIM_T2

        # (5)
        dim_t = X.shape[2]
        dim_t2 = C.shape[2]
        dim_h2 = np.ceil(dim_t2 * dim_h / dim_t).astype(int)

        # LSTM waits dim L * N * H_in
        C = C.movedim((0, 1, 2), (1, 2, 0))  # DIM_T2 * BATCH_SIZE * DIM_M
        # DIM_T2 * BATCH_SIZE * DIM_M
        pred_output, (last_hidden, last_cell) = state.predictor(C)
        prediction = state.predictor.decode(pred_output).to(
            device
        )  # DIM_T2 * BATCH_SIZE * DIM_M

        all_predictions = [prediction]
        for _ in range(dim_h2):
            pred_output, (last_hidden, last_cell) = state.predictor(
                prediction[-1].unsqueeze(0), (last_hidden, last_cell)
            )
            prediction = state.predictor.decode(pred_output).to(device)
            all_predictions.append(prediction)

        # (DIM_T2 + DIM_H2) * BATCH_SIZE * DIM_M
        Chat = torch.vstack(all_predictions).to(device)
        Chat = Chat.movedim(
            (0, 1, 2), (2, 0, 1)
        )  # BATCH_SIZE * DIM_M * (DIM_T2 + DIM_H2)

        # (2)
        Hhat = state.memory_bank.reconstruct(Chat).to(
            device
        )  # BATCH_SIZE X DIM_D X (DIM_T2 + DIM_H2)
        Xhat = state.decoder(Hhat).to(device)  # BATCH_SIZE X DIM_C X (DIM_T2 + DIM_H2)
        Xpred = Xhat[:, :, dim_t:]

    return Xpred.movedim((1, 2), (2, 1))  # BATCH_SIZE  X  DIM_H X DIM_C


def test(loader, state, dim_h, device):
    list_mse = []
    list_mae = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        Xpred = inference(X, state, dim_h, device)
        mse = torch.nn.functional.mse_loss(Xpred, y, reduction="mean")
        mae = torch.nn.functional.l1_loss(Xpred, y, reduction="mean")
        list_mse.append(mse.cpu())
        list_mae.append(mae.cpu())

    return list_mse, list_mae


# torch.cuda.set_per_process_memory_fraction(1., 0)

BATCH_SIZE = 64
# BATCH_SIZE = 99  # Just for tests to distinguish
DIM_T_1 = 192  # Longeur d'une serie chronologique stage 1
DIM_T_2 = 96  # Longeur d'une serie chronologique stage 2
# DIM_H = [96, 192, 336, 720]  # Nombre de valeur à prédire pour une serie chronologique
DIM_H = 96  # Nombre de valeur à prédire pour une serie chronologique
DIM_E = 64  # Nombre de variable d'une serie chronologique apres encodeur ( taille couche sortie encodeur)
SIZE_M = 16  # Taille de la banque de mémoire ( voir papier taille 16)
# SIZE_M = 33  # Just for tests to distinguish

STATES_DIR = "states/"


if __name__ == "__main__":
    # stage 1
    train_dataset_1, _, _ = datasets.load_ld_dataset(
        "data/LD2011_2014/LD2011_2014.txt", dim_t=DIM_T_1
    )

    train_loader_1 = DataLoader(
        train_dataset_1,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # stage 2
    train_dataset_2, val_dataset_2, test_dataset_2 = datasets.load_ld_dataset(
        "data/LD2011_2014/LD2011_2014.txt", dim_t=DIM_T_2
    )

    train_loader_2 = DataLoader(
        train_dataset_2,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader_2 = DataLoader(
        val_dataset_2,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader_2 = DataLoader(
        test_dataset_2,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = Path(STATES_DIR, "mats.pkl")
    save_path.parent.mkdir(exist_ok=True)

    if save_path.is_file():
        with save_path.open("rb") as fp:
            mats_state = torch.load(fp)
    else:
        # Nombre de variables
        dim_c = train_dataset_1.data.shape[1]

        encoder = mats.Encoder(dim_c).to(device)
        decoder = mats.Decoder(dim_c).to(device)
        discriminator = mats.Discriminator(dim_c).to(device)
        memory_bank = mats.MemoryBank(SIZE_M, DIM_E).to(device)
        predictor = mats.Predictor(SIZE_M).to(device)

        optim_edm = torch.optim.Adam(
            list(encoder.parameters())
            + list(decoder.parameters())
            + list(memory_bank.parameters()),
            lr=0.0001,
        )
        optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
        optim_predictor = torch.optim.AdamW(predictor.parameters(), lr=0.0001)

        mats_state = mats.State(
            encoder,
            decoder,
            memory_bank,
            discriminator,
            predictor,
            optim_edm,
            optim_discriminator,
            optim_predictor,
        )

    train_stage_1(
        train_loader_1,
        mats_state,
        epochs=1000,
        device=device,
        save_path=save_path,
    )

    # freeze stage 1 models
    list_models = [
        mats_state.encoder,
        mats_state.decoder,
        mats_state.discriminator,
        mats_state.memory_bank,
    ]
    for model in list_models:
        for param in model.parameters():
            param.requires_grad = False

    train_stage_2(
        train_loader_2,
        mats_state,
        DIM_H,
        epochs=500,
        device=device,
        save_path=save_path,
    )

    list_mse, list_mae = test(train_loader_2, mats_state, DIM_H, device)
    mse = np.array(list_mse).mean()
    mae = np.array(list_mae).mean()
    print(f"[TRAIN] \t MSE : {mse:.2f}")
    print(f"[TRAIN] \t MAE : {mae:.2f}")
    print("=======")

    list_mse, list_mae = test(val_loader_2, mats_state, DIM_H, device)
    mse = np.array(list_mse).mean()
    mae = np.array(list_mae).mean()
    print(f"[VAL] \t MSE : {mse:.2f}")
    print(f"[VAL] \t MAE : {mae:.2f}")
    print("=======")

    list_mse, list_mae = test(test_loader_2, mats_state, DIM_H, device)
    mse = np.array(list_mse).mean()
    mae = np.array(list_mae).mean()
    print(f"[TEST] \t MSE : {mse:.2f}")
    print(f"[TEST] \t MAE : {mae:.2f}")
    print("=======")
