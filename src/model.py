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
            nn.Dropout(p=0.1),
            nn.Conv1d(
                in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=5, stride=3, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, X):
        return self.network(X)


class Decoder(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=64, out_channels=64, kernel_size=5, stride=3, padding=1
            ),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.ConvTranspose1d(
                in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.ConvTranspose1d(
                in_channels=128,
                out_channels=dim_out,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, X):
        return self.network(X)


class Discriminator(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.backbone = Encoder(dim_in)
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=64), nn.Sigmoid()
        )

    def forward(self, X):
        backbone_out = self.backbone(X)
        backbone_out = backbone_out.transpose(1, 2)  # N * L * d
        output = self.fc(backbone_out)
        output = output.transpose(1, 2)  # N * d * L
        return output


class Predictor(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.network = nn.LSTM(
            input_size=dim_in, hidden_size=1024, num_layers=2, dropout=0.5
        )

    def forward(self, x):
        x, _ = self.network(x)
        x = torch.softmax()(x)  # TODO : utiliser torch.softmax
        return torch.tensor(x, dtype=torch.float16)


class MemoryBank(nn.Module):
    def __init__(self, size, dim) -> None:
        super().__init__()
        units = torch.zeros((dim, size))
        nn.init.uniform_(units)  # rq: ce n'est pas précisé dans le papier
        self.units = nn.Parameter(units)

    def forward(self, H):
        """
        Measures similarity for each h in H with each m in M.
        """
        numerator = torch.stack(
            [
                torch.exp(
                    -(torch.linalg.norm(H.transpose(1, 2) - m, dim=2).pow(2))
                )  # TODO : norm 1 ?
                for m in self.units.T
            ],
            dim=2,
        )
        denominator = torch.sum(numerator, dim=2).unsqueeze(2)
        C = torch.transpose(numerator / denominator, 1, 2)
        return C

    def reconstruct(self, C):
        return self.units @ C


class DiscriminatorLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, Dhat, D):
        max_D = torch.maximum(torch.zeros_like(D), 1 - D)
        max_Dhat = torch.maximum(torch.zeros_like(Dhat), 1 + Dhat)
        loss = torch.mean(max_D + max_Dhat)
        return loss


class EDMLoss(nn.Module):
    def __init__(self, memory_coef, dhat_coef) -> None:
        super().__init__()
        self.reconstruction_loss = nn.MSELoss()
        self.memory_coef = memory_coef
        self.dhat_coef = dhat_coef

    def memory_loss(self, H, M):
        norms = torch.stack(
            [torch.linalg.norm(H.transpose(1, 2) - m, dim=2) for m in M.T],
            dim=2,
        )
        Z = M.T[torch.argmin(norms, dim=2)].transpose(1, 2)
        diffs = torch.linalg.norm(H.detach() - Z, dim=1).pow(2) + torch.linalg.norm(
            H - Z.detach(), dim=1
        ).pow(2)
        loss = diffs.sum() / np.prod(H.shape)
        return loss

    def forward(self, Xhat, X, H, M, Dhat):
        return (
            self.reconstruction_loss(Xhat, X)
            + self.memory_coef * self.memory_loss(H, M)
            - self.dhat_coef * Dhat.mean()
        )
