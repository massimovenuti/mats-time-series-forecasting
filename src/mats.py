from torch import nn
from torch import autograd
from torch import linalg
import torch
import numpy as np
from torch.nn import functional as F


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
        # self.fc = nn.Sequential(nn.Linear(in_features=64, out_features=1), nn.Sigmoid())
        self.fc = nn.Linear(in_features=64, out_features=1)  # removed sigmoid

    def forward(self, X):
        backbone_out = self.backbone(X)
        backbone_out = backbone_out.transpose(1, 2)  # N * L * d
        output = self.fc(backbone_out)  # N * L * 1
        output = output.transpose(1, 2)  # N * 1 * L
        return output


class Predictor(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim_in,
            hidden_size=1024,
            num_layers=2,
            dropout=0.5,
        )
        self.decoder = nn.Sequential(nn.Linear(1024, dim_in), nn.Softmax(dim=2))

    def forward(self, X, initial_state=None):
        return self.lstm(X, initial_state)

    def decode(self, H):
        return self.decoder(H)


class MemoryBank(nn.Module):
    def __init__(self, size, dim) -> None:
        super().__init__()
        self.units = nn.Parameter(torch.randn(dim, size))
        # rq: Initialisation non précisée dans le papier
        # units = torch.zeros((dim, size))
        # nn.init.uniform_(units, -1, 1)
        # self.units = nn.Parameter(units)

    def forward(self, H):
        """
        Measures similarity for each h in H with each m in M.
        """
        diffs = [
            torch.exp(-(linalg.norm(H.transpose(1, 2) - m, dim=2).pow(2)))
            for m in self.units.T
        ]
        numerator = torch.stack(diffs, dim=2)
        denominator = torch.sum(numerator, dim=2).unsqueeze(2)
        C = torch.transpose(numerator / denominator, 1, 2)
        return C

    def reconstruct(self, C):
        return self.units @ C


class DiscriminatorLoss(nn.Module):
    def __init__(self, weight=0.8) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, Dhat, D):
        loss_D = torch.mean(F.relu(1.0 - D))
        loss_Dhat = torch.mean(F.relu(1.0 + Dhat))
        loss = self.weight * 0.5 * (loss_D + loss_Dhat)
        return loss

    # def forward(self, Dhat, D):
    #     # TODO : je ne comprends pas l'intérêt des max() vu que D et Dhat sont dans [0,1]
    #     # TODO: peut être qu'il faut utiliser Tanh en sortie du discrim au lieu de sigmoid
    #     max_D = torch.maximum(torch.zeros_like(D), 1 - D)
    #     max_Dhat = torch.maximum(torch.zeros_like(Dhat), 1 + Dhat)
    #     loss = torch.mean(max_D + max_Dhat)
    #     return loss


class EDMLoss(nn.Module):
    def __init__(
        self, decoder, alpha=1.0, gamma=1e-4, discriminator_weight=0.8
    ) -> None:
        super().__init__()
        # TODO : gamma = 1e-4 ? C.f github de VQGAN
        self.reconstruction_loss = nn.MSELoss()
        self.decoder_last_layer = decoder.network[-1].weight
        self.discriminator_weight = discriminator_weight
        self.alpha = alpha
        self.gamma = gamma

    def memory_loss(self, H, M):
        norms = torch.stack(
            [linalg.norm(H.transpose(1, 2) - m, dim=2, ord=1) for m in M.T],
            dim=2,
        )
        Z = M.T[torch.argmin(norms, dim=2)].transpose(1, 2)
        diffs = linalg.norm(H.detach() - Z, dim=2).pow(2) + linalg.norm(
            H - Z.detach(), dim=2
        ).pow(2)
        loss = diffs.sum() / np.prod(H.shape)
        return loss

    def calc_adaptive_weight(self, loss_rec, loss_d, last_layer):
        # VQGAN recommands to set lambda = 0 for at least 1 epoch
        # They set lambda to 0 in an initial warm-up phase
        # They found that longer warm-ups generally lead to better reconstructions
        rec_grads = autograd.grad(loss_rec, last_layer, retain_graph=True)[0]
        d_grads = autograd.grad(loss_d, last_layer, retain_graph=True)[0]

        weight = linalg.norm(rec_grads) / (linalg.norm(d_grads) + self.gamma)

        weight = torch.clamp(weight, 0.0, 1e4)

        return weight.detach()

    def forward(self, Xhat, X, H, M, Dhat, lmbda=None):
        loss_rec = self.reconstruction_loss(Xhat, X)
        loss_m = self.memory_loss(H, M)
        loss_d = -Dhat.mean()

        if lmbda is None:
            lmbda = self.calc_adaptive_weight(loss_rec, loss_d, self.decoder_last_layer)

        loss = (
            loss_rec + self.alpha * loss_m + self.discriminator_weight * lmbda * loss_d
        )

        return loss, (loss_rec, loss_m, loss_d, lmbda)


class State:
    def __init__(
        self,
        encoder,
        decoder,
        memory_bank,
        discriminator,
        predictor,
        optim_edm,
        optim_discriminator,
        optim_predictor,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.memory_bank = memory_bank
        self.discriminator = discriminator
        self.predictor = predictor
        self.optim_edm = optim_edm
        self.optim_discriminator = optim_discriminator
        self.optim_predictor = optim_predictor
        self.stage_1_epoch = 0
        self.stage_1_iteration = 0
        self.stage_2_epoch = 0
        self.stage_2_iteration = 0
