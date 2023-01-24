from torch import nn
from torch import autograd
from torch import linalg
import torch
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm


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
        self.fc = nn.Sequential(nn.Linear(in_features=64, out_features=1), nn.Sigmoid())
        # self.fc = nn.Linear(in_features=64, out_features=1)  # removed sigmoid

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
        self.decode = nn.Sequential(nn.Linear(1024, dim_in), nn.Softmax(dim=2))

    def forward(self, X, horizon):
        # X : BATCH_SIZE * DIM_M * DIM_T2
        # LSTM waits dim L * N * H_in
        X = X.movedim((0, 1, 2), (1, 2, 0))  # DIM_T2 * BATCH_SIZE * DIM_M

        # TODO : should we do teacher forcing only ?
        # See Curriculum Learning
        pred_output, (last_hidden, last_cell) = self.lstm(X)
        prediction = self.decode(pred_output)  # DIM_T2 * BATCH_SIZE * DIM_M

        all_predictions = [prediction]

        for _ in range(horizon):
            pred_output, (last_hidden, last_cell) = self.lstm(
                prediction[-1].unsqueeze(0), (last_hidden, last_cell)
            )
            prediction = self.decode(pred_output)
            all_predictions.append(prediction)

        # (DIM_T2 + DIM_H2) * BATCH_SIZE * DIM_M
        preds = torch.vstack(all_predictions)

        # BATCH_SIZE * DIM_M * (DIM_T2 + DIM_H2)
        preds = preds.movedim((0, 1, 2), (2, 0, 1))

        return preds


class MemoryBank(nn.Module):
    def __init__(self, size, dim) -> None:
        super().__init__()
        # self.units = nn.Parameter(torch.randn(dim, size))
        # rq: Initialisation non précisée dans le papier
        units = torch.zeros((dim, size))
        nn.init.uniform_(units, -1, 1)
        self.units = nn.Parameter(units)

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
        return loss, (loss_D, loss_Dhat)

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
    def __init__(self) -> None:
        self.epoch = 0
        self.iteration = 0


class MATS(nn.Module):
    def __init__(self, dim_c, size_m, dim_e) -> None:
        super().__init__()
        self.encoder = Encoder(dim_c)
        self.decoder = Decoder(dim_c)
        self.memory_bank = MemoryBank(size_m, dim_e)
        self.discriminator = Discriminator(dim_c)
        self.predictor = Predictor(size_m)

        self.optim_edm = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.memory_bank.parameters()),
            lr=0.0001,
        )
        self.optim_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0001
        )
        self.optim_predictor = torch.optim.AdamW(self.predictor.parameters(), lr=0.0001)

        self.state_1 = State()
        self.state_2 = State()

    def encode(self, X):
        X = torch.movedim(X, 1, 2)  # BATCH_SIZE * DIM_C * DIM_T
        H = self.encoder(X)  # BATCH_SIZE * DIM_D * DIM_T2
        C = self.memory_bank(H)  # BATCH_SIZE * DIM_M * DIM_T2
        return C, H

    def decode(self, C):
        Hhat = self.memory_bank.reconstruct(C)  # BATCH_SIZE * DIM_D * DIM_T2
        Xhat = self.decoder(Hhat)  # BATCH_SIZE * DIM_C * DIM_T
        return Xhat

    def get_dim_h2(self, dim_t, dim_t2, horizon):
        return np.ceil(dim_t2 * horizon / dim_t).astype(int)

    def predict(self, X, horizon):
        C, _ = self.encode(X)  # BATCH_SIZE * DIM_M * DIM_T2
        dim_h2 = self.get_dim_h2(X.shape[1], C.shape[2], horizon)
        Chat = self.predictor(C, dim_h2)  # BATCH_SIZE * DIM_M * DIM_T2
        Xhat = self.decode(Chat).movedim(1, 2)  # BATCH_SIZE * DIM_T * DIM_C
        Xpred = Xhat[:, -horizon:, :]  # BATCH_SIZE * DIM_H * DIM_C
        return Xpred

    def step_stage_1(self, X, optim_index, lmbda=None):
        criterion_edm = EDMLoss(self.decoder)
        criterion_discriminator = DiscriminatorLoss()

        self.optim_edm.zero_grad()
        self.optim_discriminator.zero_grad()

        C, H = self.encode(X)  # C : BATCH_SIZE * DIM_M * DIM_T2
        Xhat = self.decode(C)  # BATCH_SIZE * DIM_C * DIM_T
        X = X.movedim(1, 2)  # BATCH_SIZE * DIM_C * DIM_T

        if optim_index == 0:
            # (4)
            Dhat = self.discriminator(Xhat)  #  BATCH_SIZE * 1 * DIM_T2
            loss, partial_losses = criterion_edm(
                Xhat, X, H, self.memory_bank.units, Dhat, lmbda
            )
            loss.backward()
            self.optim_edm.step()
        else:
            # (3)
            D = self.discriminator(X.detach())  # BATCH_SIZE * 1 * DIM_T2
            Dhat = self.discriminator(Xhat.detach())  #  BATCH_SIZE * 1 * DIM_T2
            loss, partial_losses = criterion_discriminator(Dhat, D)
            loss.backward()
            self.optim_discriminator.step()

        return loss, partial_losses

    def step_stage_2(self, X, y):
        criterion_predictor = nn.BCELoss()
        self.optim_predictor.zero_grad()

        C, _ = self.encode(X)  # BATCH_SIZE * DIM_M * DIM_T2
        dim_h2 = self.get_dim_h2(X.shape[1], C.shape[2], y.shape[1])
        # BATCH_SIZE * DIM_M * (DIM_T2 + DIM_H2)
        Chat = self.predictor(C, dim_h2)  # BATCH_SIZE * DIM_M * DIM_T2

        # (6)
        X_gt = torch.cat((X, y), dim=1)  # BATCH_SIZE * (DIM_T + DIM_H) * DIM_C
        C_gt, _ = self.encode(X_gt)  # BATCH_SIZE * DIM_M * (DIM_T2 + DIM_H2)

        # (7)
        loss = criterion_predictor(Chat, C_gt)
        loss.backward()
        self.optim_predictor.step()

        return loss

    def train_stage_1(
        self, dataloader, epochs, save_path, writer, device="cpu", disc_start=10000
    ):
        # stage 1
        iteration = self.state_1.iteration
        pbar = tqdm(range(self.state_1.epoch, epochs), leave=False)
        for epoch in pbar:
            for X, _ in dataloader:
                X = X.to(device)
                optim_index = 0 if iteration < disc_start or iteration % 2 == 0 else 1
                lmbda = None if iteration > disc_start else 0

                loss, partial_losses = self.step_stage_1(X, optim_index, lmbda)

                if optim_index == 0:
                    loss_rec, loss_m, loss_d, lmbda = partial_losses
                    writer.add_scalars("Loss_S1", {"edm": loss}, iteration)
                    writer.add_scalars(
                        "EDM_partial_losses",
                        {"loss_rec": loss_rec, "loss_m": loss_m, "loss_d": loss_d},
                        iteration,
                    )
                    writer.add_scalar("Lambda", lmbda, iteration)
                else:
                    writer.add_scalars("Loss_S1", {"discriminator": loss}, iteration)

                iteration = iteration + 1

            self.state_1.iteration = iteration
            self.state_1.epoch = epoch + 1

            with save_path.open("wb") as fp:
                torch.save(self, fp)

    def train_stage_2(self, dataloader, epochs, save_path, writer, device="cpu"):
        iteration = self.state_2.iteration
        pbar = tqdm(range(self.state_2.epoch, epochs), leave=False)
        for epoch in pbar:
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)
                loss = self.step_stage_2(X, y)
                writer.add_scalar("Loss_S2", loss, iteration)
                iteration = iteration + 1

            self.state_2.iteration = iteration
            self.state_2.epoch = epoch + 1

            with save_path.open("wb") as fp:
                torch.save(self, fp)

    def fit(
        self,
        dataloader,
        epochs_s1,
        epochs_s2,
        save_path,
        writer,
        device="cpu",
    ):
        self.train()
        self.train_stage_1(dataloader, epochs_s1, save_path, writer, device)

        # freeze stage 1
        list_models = [
            self.encoder,
            self.decoder,
            self.discriminator,
            self.memory_bank,
        ]
        for model in list_models:
            for param in model.parameters():
                param.requires_grad = False

        self.train_stage_2(dataloader, epochs_s2, save_path, writer, device)

    @torch.no_grad()
    def test(self, dataloader, device="cpu"):
        self.eval()
        list_mse = []
        list_mae = []

        for X, y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)
            Xpred = self.predict(X, horizon=y.shape[1])
            mse = F.mse_loss(Xpred, y, reduction="mean")
            mae = F.l1_loss(Xpred, y, reduction="mean")
            list_mse.append(mse.cpu())
            list_mae.append(mae.cpu())

        return list_mse, list_mae

    def forward(self, X):
        return self.predict(X)
