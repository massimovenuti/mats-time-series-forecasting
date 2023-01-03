from torch import nn
import torch
import numpy as np


class Encoder(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                in_channels=dim_in, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=True),
            nn.Conv1d(
                in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=True),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=5, stride=3, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=64, out_channels=64, kernel_size=5, stride=3, padding=1
            ),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=True),
            nn.ConvTranspose1d(
                in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=True),
            nn.ConvTranspose1d(
                in_channels=128,
                out_channels=dim_out,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


class Discriminator(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                in_channels=dim_in, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=True),
            nn.Conv1d(
                in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=True),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=5, stride=3, padding=1
            ),
            nn.Tanh(),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=64), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.network(x)
        x = torch.movedim(x, 1, 2)
        x = self.fc(x)
        x = torch.movedim(x, 1, 2)
        x = np.where(x >= 0.5, 1.0, 0.0)
        return torch.tensor(x, dtype=torch.float32)


class Predictor(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.network = nn.Sequential(
            torch.nn.LSTM(input_size=dim_in, hidden_size=128, num_layers=8, dropout=0.5)
        )

    def forward(self, x):
        x, _ = self.network(x)
        x = torch.nn.Softmax()(x)
        return torch.tensor(x, dtype=torch.float32)


def measure_similarity(H, M):
    H = H.detach().numpy()
    M = M.detach().numpy()

    tmp = np.exp(
        -1
        * (
            -2 * (M.T @ H)
            + np.sum(M**2, axis=0)[:, np.newaxis]
            + np.sum(H**2, axis=1)[:, np.newaxis]
        )
    )
    sum_tmp = np.sum(tmp, axis=0)
    C = tmp / sum_tmp
    C = np.where(C.T == np.max(C.T, axis=0), 1.0, 0.0).T
    return torch.tensor(C, dtype=torch.float32)  # DIM_M * DIM_T2


def calcule_loss_reconstruction(DIM_N, DIM_T, DIM_C, X, X_reconstruit):
    loss = (1 / (DIM_N * DIM_T * DIM_C)) * torch.sum(
        torch.pow(X_reconstruit - X, 2)
    )  # cout loss reconstruction
    return (
        loss.clone().detach().requires_grad_(True)
    )  # torch.tensor(loss,requires_grad=True)


def calcule_loss_m(DIM_N, DIM_T2, DIM_D, X, H, M):
    H2 = H.clone().detach()  # avec stop gradient
    H2.requires_grad = False
    Z = np.argmin(H.detach().numpy() - M.detach().numpy(), axis=1)
    Z2 = np.argmin(H2 - M, axis=1)  # avec stop gradient
    Z2.requires_grad = False
    loss = (1 / (DIM_N * DIM_T2 * DIM_D)) * torch.sum(
        torch.sum(torch.pow(H2 - Z, 2) + torch.pow(H - Z2, 2))
    )
    return (
        loss.clone().detach().requires_grad_(True)
    )  # torch.tensor(loss,requires_grad=True)


def calcule_loss_d(DIM_N, DIM_T2, DATA_D, DATA_D_reconstruit):
    torch_zero = torch.zeros_like(DATA_D)
    loss = (1 / (DIM_N * DIM_T2)) * torch.sum(
        torch.sum(
            torch.maximum(torch_zero, 1 - DATA_D)
            + torch.maximum(torch_zero, 1 + DATA_D_reconstruit)
        )
    )
    return (
        loss.clone().detach().requires_grad_(True)
    )  # torch.tensor(loss,requires_grad=True)


def calcule_loss(loss_REC, loss_M, DATA_D_reconstruit, DIM_N, DIM_T2, lambd, gamma):
    loss = (
        loss_REC
        + lambd * loss_M
        - (gamma / (DIM_N * DIM_T2)) * torch.sum(DATA_D_reconstruit)
    )
    return (
        loss.clone().detach().requires_grad_(True)
    )  # torch.tensor(loss,requires_grad=True)


def calcule_loss_pred(DIM_T2, DIM_HH, DATA_C_reconstruit, DATA_CC):
    loss = (1 / (DIM_T2 + DIM_HH)) * torch.nn.functional.binary_cross_entropy(
        DATA_CC, DATA_C_reconstruit
    )
    return loss


def calcule_mse(DIM_N, DIM_H, DIM_C, y, y_pred):
    return (1 / DIM_N * DIM_H * DIM_C) * torch.nn.functional.mse_loss(y_pred, y)


def calcule_mae(DIM_N, DIM_H, DIM_C, y, y_pred):
    return (1 / DIM_N * DIM_H * DIM_C) * torch.nn.functional.l1_loss(y_pred, y)
