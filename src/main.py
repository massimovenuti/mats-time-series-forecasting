import datasets
from torch.utils.data import DataLoader
import torch
import model
from tqdm import tqdm
from torch import nn
import numpy as np


def train_stage_1(
    dataloader,
    encoder,
    decoder,
    discriminator,
    memory_bank,
    memory_coef,
    dhat_coef,
    epochs,
    device,
):
    optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    optim_edm = torch.optim.Adam(
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(memory_bank.parameters()),
        lr=0.0001,
    )

    criterion_edm = model.EDMLoss(memory_coef, dhat_coef)
    criterion_discriminator = model.DiscriminatorLoss()

    for epoch in tqdm(range(epochs)):
        for e, (X, _) in enumerate(dataloader):
            optim_edm.zero_grad()
            optim_discriminator.zero_grad()

            X = torch.movedim(X, 1, 2).to(device)  # BATCH_SIZE X DIM_C X DIM_T
            H = encoder(X).to(device)  # BATCH_SIZE X DIM_D X DIM_T2
            C = memory_bank(H).to(device)  # BATCH_SIZE * DIM_M * DIM_T2

            Hhat = memory_bank.reconstruct(C).to(device)  # BATCH_SIZE X DIM_D X DIM_T2

            Xhat = decoder(Hhat).to(device)  # BATCH_SIZE X DIM_C X DIM_T

            D = discriminator(X).to(device)  # BATCH_SIZE X DIM_D X DIM_T2
            Dhat = discriminator(Xhat).to(device)  #  BATCH_SIZE X DIM_D X DIM_T2

            if e % 2 == 0:
                # (4)
                loss = criterion_edm(Xhat, X, H, memory_bank.units, Dhat)
                loss.backward()
                optim_edm.step()
            else:
                # (3)
                loss = criterion_discriminator(Dhat, D)
                loss.backward()
                optim_discriminator.step()

            # TODO : use tensorboard
            if e % 20 == 0 or e % 20 == 1:
                print(
                    f"[STAGE 1]"
                    f"[{epoch}/{epochs}][{e}/{len(dataloader)}]"
                    f"[{'EDM' if e%2 == 0 else 'D'}]\t"
                    f"Loss : {loss:.2f}"
                )


def train_stage_2(dataloader, encoder, memory_bank, predictor, dim_h, epochs, device):
    optim_predictor = torch.optim.AdamW(predictor.parameters(), lr=0.0001)
    criterion_predictor = nn.BCELoss()

    for epoch in range(epochs):
        for e, (X, y) in enumerate(dataloader):
            optim_predictor.zero_grad()

            # (1)
            # CNN waits dim N * C_in * L
            X = torch.movedim(X, 1, 2).to(device)  # BATCH_SIZE * DIM_C * DIM_T
            H = encoder(X).to(device)  # BATCH_SIZE * DIM_D * DIM_T2
            C = memory_bank(H).to(device)  # BATCH_SIZE * DIM_M * DIM_T2

            # (5)
            dim_t = X.shape[2]
            dim_t2 = C.shape[2]
            dim_h2 = np.ceil(dim_t2 * dim_h / dim_t).astype(int)

            # LSTM waits dim L * N * H_in
            C = C.movedim((0, 1, 2), (1, 2, 0))  # DIM_T2 * BATCH_SIZE * DIM_M
            # DIM_T2 * BATCH_SIZE * DIM_M
            pred_output, (last_hidden, last_cell) = predictor(C)
            prediction = predictor.decode(pred_output)  # DIM_T2 * BATCH_SIZE * DIM_M

            all_predictions = [prediction]
            for _ in range(dim_h2):
                pred_output, (last_hidden, last_cell) = predictor(
                    prediction[-1].unsqueeze(0), (last_hidden, last_cell)
                )
                prediction = predictor.decode(pred_output)
                all_predictions.append(prediction)

            # (DIM_T2 + DIM_H2) * BATCH_SIZE * DIM_M
            Chat = torch.vstack(all_predictions)

            # (6)
            y = torch.movedim(y, 1, 2).to(device)  # BATCH_SIZE * DIM_C * DIM_T
            X_gt = torch.cat((X, y), dim=2)  # BATCH_SIZE * DIM_C * (DIM_T + DIM_H)
            H_gt = encoder(X_gt).to(device)  # BATCH_SIZE * DIM_D * (DIM_T2 + DIM_H2)
            # BATCH_SIZE * DIM_M * (DIM_T2 + DIM_H2)
            C_gt = memory_bank(H_gt).to(device)
            # (DIM_T2 + DIM_H2) * BATCH_SIZE * DIM_M
            C_gt = C_gt.movedim((0, 1, 2), (1, 2, 0)).to(device)

            # (7)
            loss = criterion_predictor(Chat, C_gt)
            loss.backward()
            optim_predictor.step()

            # TODO : use tensorboard
            if e % 20 == 0:
                print(f"[{epoch}/{epochs}][{e}/{len(dataloader)}]\t Loss : {loss:.2f}")


BATCH_SIZE = 64
# BATCH_SIZE = 99  # Just for tests to distinguish
DIM_T = 192  # Longeur d'une serie chronologique stage 1
DIM_TT = 96  # Longeur d'une serie chronologique stage 2
# DIM_H = [96, 192, 336, 720]  # Nombre de valeur à prédire pour une serie chronologique
DIM_H = 96  # Nombre de valeur à prédire pour une serie chronologique
DIM_E = 64  # Nombre de variable d'une serie chronologique apres encodeur ( taille couche sortie encodeur)
SIZE_M = 16  # Taille de la banque de mémoire ( voir papier taille 16)
# SIZE_M = 33  # Just for tests to distinguish
MEMORY_COEF = 0.5
DHAT_COEF = 0.5

if __name__ == "__main__":
    # stage 1
    train_dataset, val_dataset, test_dataset = datasets.load_ld_dataset(
        "data/LD2011_2014/LD2011_2014.txt"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim_c = train_dataset.data.shape[1]  # Nombre de variables d'une serie chronologique

    encoder = model.Encoder(dim_c).to(device)
    decoder = model.Decoder(dim_c).to(device)
    discriminator = model.Discriminator(dim_c).to(device)
    memory_bank = model.MemoryBank(SIZE_M, DIM_E).to(device)

    train_stage_1(
        train_loader,
        encoder,
        decoder,
        discriminator,
        memory_bank,
        MEMORY_COEF,
        DHAT_COEF,
        epochs=5,
        device=device,
    )

    # freeze stage 1 networks
    list_networks = [encoder, decoder, discriminator, memory_bank]
    for network in list_networks:
        for param in network.parameters():
            param.requires_grad = False

    predictor = model.Predictor(SIZE_M)

    train_stage_2(
        train_loader, encoder, memory_bank, predictor, DIM_H, epochs=5, device=device
    )
