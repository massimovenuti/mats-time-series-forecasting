import datasets
from torch.utils.data import DataLoader
import torch
from math import ceil
import model


def train_stage_1(dataloader, epochs):

    DIM_C = dataloader.dataset.data.shape[
        1
    ]  # Nombre de variable d'une serie chronologique
    DIM_D = 64  # Nombre de variable d'une serie chronologique apres encodeur ( taille couche sortie encodeur)
    DIM_N = (
        len(dataloader.dataset.data)
        - dataloader.dataset.DIM_T
        - dataloader.dataset.DIM_H
        + 1
    )  # Nombre de segment de longeur T
    DIM_M = 16  # Taille de la banque de mémoire ( voir papier taille 16)
    lambd = 0.5
    gamma = 0.5

    encoder = model.Encoder(DIM_C)
    decoder = model.Decoder(DIM_C)
    discriminator = model.Discriminator(DIM_C)

    opt_encoder = torch.optim.Adam(encoder.parameters(), lr=0.0001)
    opt_decoder = torch.optim.Adam(decoder.parameters(), lr=0.0001)
    opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    DATA_M = torch.rand(
        (DIM_D, DIM_M), dtype=torch.float32
    )  # Banque de mémoire :  DIM_D X DIM_M

    for epoch in epochs:
        for e, (X, Y) in enumerate(dataloader):

            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            opt_discriminator.zero_grad()

            # (1)
            DATA_X = torch.movedim(X, 1, 2)  # Data : BATCH_SIZE X DIM_C X DIM_T
            DATA_H = encoder(DATA_X)  # Data Encoder : BATCH_SIZE X DIM_D X DIM_T2
            DATA_C = model.measure_similarity(
                DATA_H, DATA_M
            )  # Matrice similarity : BATCH_SIZE * DIM_M * DIM_T2
            DIM_T2 = DATA_H.shape[
                2
            ]  # Longeur d'une serie chronologique apres convolution T2 < T à cause convolution et stride suprérieur à 1

            DATA_H_reconstruit = (
                DATA_M @ DATA_C
            )  # Data Encoder : BATCH_SIZE X DIM_D X DIM_T2
            DATA_X_reconstruit = decoder(
                DATA_H_reconstruit
            )  # Data reconstruit : BATCH_SIZE X DIM_C X DIM_T
            loss_REC = model.calcule_loss_reconstruction(
                DIM_N,
                DIM_T,
                DIM_C,
                DATA_X.clone().detach(),
                DATA_X_reconstruit.clone().detach(),
            )  # cout loss reconstruction
            loss_M = model.calcule_loss_m(
                DIM_N, DIM_T2, DIM_D, DATA_X, DATA_H, DATA_M
            )  # cout loss m
            DATA_D = discriminator(DATA_X)  # : BATCH_SIZE X DIM_D X DIM_T2
            DATA_D_reconstruit = discriminator(
                DATA_X_reconstruit
            )  #  : BATCH_SIZE X DIM_D X DIM_T2

            if e % 2 == 0:
                # (4)
                loss = model.calcule_loss(
                    loss_REC, loss_M, DATA_D_reconstruit, DIM_N, DIM_T2, lambd, gamma
                )
                loss.backward()
                opt_encoder.step()
                opt_decoder.step()
                print("loss : ", loss)
            else:
                # (3)
                loss_D = model.calcule_loss_d(DIM_N, DIM_T2, DATA_D, DATA_D_reconstruit)
                loss_D.backward()
                print("loss_D : ", loss_D)
                opt_discriminator.step()

            #             print('loss_REC : ',loss_REC)
            #             print('loss_M : ', loss_M)
            #             print('loss_D : ', loss_D)

    return encoder, decoder, DATA_M


def train_stage_2(encoder, decoder, DATA_M, dataloader, epochs):

    DIM_C = dataloader.dataset.data.shape[
        1
    ]  # Nombre de variable d'une serie chronologique
    DIM_D = 64  # Nombre de variable d'une serie chronologique apres encodeur ( taille couche sortie encodeur)
    DIM_N = (
        len(dataloader.dataset.data)
        - dataloader.dataset.DIM_T
        - dataloader.dataset.DIM_H
        + 1
    )  # Nombre de segment de longeur T
    DIM_M = 16  # Taille de la banque de mémoire ( voir papier taille 16)

    predictor = model.Predictor(DIM_M)

    opt_predictor = torch.optim.AdamW(predictor.parameters(), lr=0.0001)

    for epoch in epochs:
        for X, Y in dataloader:

            opt_predictor.zero_grad()

            # (1)
            DATA_X = torch.movedim(X, 1, 2)  # Data : BATCH_SIZE X DIM_C X DIM_T
            DATA_H = encoder(DATA_X)  # Data Encoder : BATCH_SIZE X DIM_D X DIM_T2
            DATA_C = model.measure_similarity(
                DATA_H, DATA_M
            )  # Matrice similarity : BATCH_SIZE * DIM_M * DIM_T2
            DIM_T2 = DATA_H.shape[
                2
            ]  # Longeur d'une serie chronologique apres convolution T2 < T à cause convolution et stride suprérieur à 1

            # (5)
            DIM_CC = ceil(
                DIM_T2
                * (dataloader.dataset.DIM_T + dataloader.dataset.DIM_H)
                / dataloader.dataset.DIM_T
            )
            DIM_HH = ceil(DIM_T2 * dataloader.dataset.DIM_H / dataloader.dataset.DIM_T)

            DATA_C = torch.movedim(DATA_C, 1, 2)  # Data : BATCH_SIZE X DIM_T2 X DIM_M
            DATA_C_reconstruit = predictor(DATA_C)

            # (6)
            DATA_X_Y = torch.cat((X, Y), dim=0)
            DATA_X_Y = encoder(DATA_X_Y)
            DATA_CC = DATA_M @ DATA_X_Y

            # (7)
            loss_P = model.calcule_loss_pred(
                DIM_T2, DIM_HH, DATA_C_reconstruit, DATA_CC
            )
            loss_P.backward()

            print("loss_P : ", loss_P)
            print()

            opt_predictor.step()

    return predictor


BATCH_SIZE = 64
DIM_T = 192  # Longeur d'une serie chronologique stage 1
DIM_TT = 96  # Longeur d'une serie chronologique stage 2
DIM_H = [96, 192, 336, 720]  # Nombre de valeur à prédire pour une serie chronologique

if __name__ == "__main__":
    # stage 1
    dataset1 = datasets.DatasetLd(
        path="data/LD2011_2014/LD2011_2014.pkl", DIM_T=DIM_T, DIM_H=DIM_H[0]
    )

    dataloader1 = DataLoader(
        dataset1,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dataset1.__collate_fn__,
        drop_last=True,
    )

    encoder, decoder, DATA_M = train_stage_1(dataloader1, epochs=range(1))

    # stage 2
    dataset2 = datasets.DatasetLd(
        path="../data/LD2011_2014/LD2011_2014.pkl", DIM_T=DIM_TT, DIM_H=DIM_H[0]
    )

    dataloader2 = DataLoader(
        dataset2,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dataset2.__collate_fn__,
    )

    DIM_C = dataloader2.dataset.data.shape[
        1
    ]  # Nombre de variable d'une serie chronologique
    encoder = model.Encoder(DIM_C)
    decoder = model.Decoder(DIM_C)
    discriminator = model.Discriminator(DIM_C)
    DATA_M = torch.rand(
        (64, 16), dtype=torch.float32
    )  # Banque de mémoire :  DIM_D X DIM_M

    list_model = [encoder, decoder, discriminator]
    for model in list_model:
        for param in model.parameters():
            param.requires_grad = False

    train_stage_2(encoder, decoder, DATA_M, dataloader2, epochs=range(1))
