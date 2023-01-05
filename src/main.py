import datasets
from torch.utils.data import DataLoader
import torch
from math import ceil
import model


def train_stage_1(
    dataloader,
    encoder,
    decoder,
    discriminator,
    memory_bank,
    memory_coef,
    dhat_coef,
    epochs,
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

    for epoch in range(epochs):
        for e, (X, _) in enumerate(dataloader):
            optim_edm.zero_grad()
            optim_discriminator.zero_grad()

            X = torch.movedim(X, 1, 2)  # BATCH_SIZE X DIM_C X DIM_T
            H = encoder(X)  # BATCH_SIZE X DIM_D X DIM_T2
            C = memory_bank(H)  # BATCH_SIZE * DIM_M * DIM_T2

            Hhat = memory_bank.reconstruct(C)  # BATCH_SIZE X DIM_D X DIM_T2

            Xhat = decoder(Hhat)  # BATCH_SIZE X DIM_C X DIM_T

            D = discriminator(X)  # BATCH_SIZE X DIM_D X DIM_T2
            Dhat = discriminator(Xhat)  #  BATCH_SIZE X DIM_D X DIM_T2

            # ? : freeze ok ?
            if e % 2 == 0:
                # (4)
                # rq : .detach() peut ne pas être obligatoire ?
                loss = criterion_edm(Xhat, X, H, memory_bank.units, Dhat.detach())
                loss.backward()
                optim_edm.step()
                print("loss_edm : ", loss)  # TODO : use tensorboard
            else:
                # (3)
                loss = criterion_discriminator(Dhat, D)
                loss.backward()
                optim_discriminator.step()
                print("loss_d : ", loss)  # TODO : use tensorboard


# def train_stage_2(encoder, decoder, DATA_M, dataloader, epochs):

#     DIM_C = dataloader.dataset.data.shape[
#         1
#     ]  # Nombre de variable d'une serie chronologique
#     DIM_D = 64  # Nombre de variable d'une serie chronologique apres encodeur ( taille couche sortie encodeur)
#     DIM_N = (
#         len(dataloader.dataset.data)
#         - dataloader.dataset.DIM_T
#         - dataloader.dataset.DIM_H
#         + 1
#     )  # Nombre de segment de longeur T
#     DIM_M = 16  # Taille de la banque de mémoire ( voir papier taille 16)

#     predictor = model.Predictor(DIM_M)

#     opt_predictor = torch.optim.AdamW(predictor.parameters(), lr=0.0001)

#     for epoch in epochs:
#         for X, Y in dataloader:

#             opt_predictor.zero_grad()

#             # (1)
#             DATA_X = torch.movedim(X, 1, 2)  # Data : BATCH_SIZE X DIM_C X DIM_T
#             DATA_H = encoder(DATA_X)  # Data Encoder : BATCH_SIZE X DIM_D X DIM_T2
#             DATA_C = model.measure_similarity(
#                 DATA_H, DATA_M
#             )  # Matrice similarity : BATCH_SIZE * DIM_M * DIM_T2
#             DIM_T2 = DATA_H.shape[
#                 2
#             ]  # Longeur d'une serie chronologique apres convolution T2 < T à cause convolution et stride suprérieur à 1

#             # (5)
#             DIM_CC = ceil(
#                 DIM_T2
#                 * (dataloader.dataset.DIM_T + dataloader.dataset.DIM_H)
#                 / dataloader.dataset.DIM_T
#             )
#             DIM_HH = ceil(DIM_T2 * dataloader.dataset.DIM_H / dataloader.dataset.DIM_T)

#             DATA_C = torch.movedim(DATA_C, 1, 2)  # Data : BATCH_SIZE X DIM_T2 X DIM_M
#             DATA_C_reconstruit = predictor(DATA_C)

#             # (6)
#             DATA_X_Y = torch.cat((X, Y), dim=0)
#             DATA_X_Y = encoder(DATA_X_Y)
#             DATA_CC = DATA_M @ DATA_X_Y

#             # (7)
#             loss_P = model.calcule_loss_pred(
#                 DIM_T2, DIM_HH, DATA_C_reconstruit, DATA_CC
#             )
#             loss_P.backward()

#             print("loss_P : ", loss_P)
#             print()

#             opt_predictor.step()

#     return predictor


# BATCH_SIZE = 64
BATCH_SIZE = 99  # Just for tests to distinguish
DIM_T = 192  # Longeur d'une serie chronologique stage 1
DIM_TT = 96  # Longeur d'une serie chronologique stage 2
# DIM_H = [96, 192, 336, 720]  # Nombre de valeur à prédire pour une serie chronologique
DIM_H = 96  # Nombre de valeur à prédire pour une serie chronologique
DIM_E = 64  # Nombre de variable d'une serie chronologique apres encodeur ( taille couche sortie encodeur)
# SIZE_M = 16  # Taille de la banque de mémoire ( voir papier taille 16)
SIZE_M = 33  # Just for tests to distinguish
MEMORY_COEF = 0.5
DHAT_COEF = 0.5

if __name__ == "__main__":
    # stage 1
    dataset1 = datasets.DatasetLd(
        path="data/LD2011_2014/LD2011_2014.pkl", DIM_T=DIM_T, DIM_H=DIM_H
    )

    dataloader1 = DataLoader(
        dataset1,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dataset1.__collate_fn__,
        drop_last=True,
    )

    dim_c = dataset1.data.shape[1]  # Nombre de variables d'une serie chronologique

    encoder = model.Encoder(dim_c)
    decoder = model.Decoder(dim_c)
    discriminator = model.Discriminator(dim_c)
    memory_bank = model.MemoryBank(SIZE_M, DIM_E)

    train_stage_1(
        dataloader1,
        encoder,
        decoder,
        discriminator,
        memory_bank,
        MEMORY_COEF,
        DHAT_COEF,
        epochs=5,
    )

    # # stage 2
    # dataset2 = datasets.DatasetLd(
    #     path="../data/LD2011_2014/LD2011_2014.pkl", DIM_T=DIM_TT, DIM_H=DIM_H[0]
    # )

    # dataloader2 = DataLoader(
    #     dataset2,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     collate_fn=dataset2.__collate_fn__,
    # )

    # DIM_C = dataloader2.dataset.data.shape[
    #     1
    # ]  # Nombre de variable d'une serie chronologique
    # encoder = model.Encoder(DIM_C)
    # decoder = model.Decoder(DIM_C)
    # discriminator = model.Discriminator(DIM_C)
    # DATA_M = torch.rand(
    #     (64, 16), dtype=torch.float32
    # )  # Banque de mémoire :  DIM_D X DIM_M

    # list_model = [encoder, decoder, discriminator]
    # for model in list_model:
    #     for param in model.parameters():
    #         param.requires_grad = False

    # train_stage_2(encoder, decoder, DATA_M, dataloader2, epochs=range(1))
