from torch import nn
from torch import autograd
from torch import linalg
import torch
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
import math


# TODO : should we use ?
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


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

    def forward(self, H):
        return self.network(H)


class Discriminator(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.backbone = Encoder(dim_in)
        self.backbone.apply(weights_init)
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

    def forward(self, X, horizon=None, teacher_prob=0, X_gt=None):
        # Curriculum Learning
        output, (last_hidden, last_cell) = self.lstm(X)
        prediction = self.decode(output)  # DIM_T2 * BATCH_SIZE * DIM_M
        all_predictions = [prediction]
        
        if teacher_prob > 0 and X_gt is not None:
            horizon = len(X_gt) - len(X)
        else:
            assert horizon is not None
        
        for i in range(len(X), len(X) + horizon):
            if torch.rand(1) < teacher_prob:
                # Teacher forcing
                current_input = X_gt[i].unsqueeze(0)
            else:
                # Use previous prediction
                current_input = prediction[-1].unsqueeze(0)
            output, (last_hidden, last_cell) = self.lstm(current_input, (last_hidden, last_cell))
            prediction = self.decode(output)
            all_predictions.append(prediction)

        # (DIM_T2 + DIM_H2) * BATCH_SIZE * DIM_M
        preds = torch.vstack(all_predictions)
        return preds


class MemoryBank(nn.Module):
    def __init__(self, size, dim) -> None:
        super().__init__()
        # self.units = nn.Parameter(torch.randn(dim, size))
        # rq: Initialisation non précisée dans le papier
        units = torch.zeros((dim, size))
        nn.init.uniform_(units, -1.0 / size, 1.0 / size)
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

    # def forward(self, Dhat, D):
    #     loss_D = torch.mean(F.relu(1.0 - D))
    #     loss_Dhat = torch.mean(F.relu(1.0 + Dhat))
    #     loss = self.weight * 0.5 * (loss_D + loss_Dhat)
    #     return loss, (loss_D, loss_Dhat)

    def forward(self, Dhat, D):
        # TODO : je ne comprends pas l'intérêt des max() vu que D et Dhat sont dans [0,1]
        max_D = torch.maximum(torch.zeros_like(D), 1 - D)
        max_Dhat = torch.maximum(torch.zeros_like(Dhat), 1 + Dhat)
        loss = self.weight * torch.mean(max_D + max_Dhat)
        return loss, (max_D, max_Dhat)


class EDMLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1e-4, discriminator_weight=0.8) -> None:
        super().__init__()
        # TODO : gamma = 1e-4 ? C.f github de VQGAN
        self.reconstruction_loss = nn.MSELoss()
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

    def forward(self, Xhat, X, H, M, Dhat, decoder, lmbda=None):
        loss_rec = self.reconstruction_loss(Xhat, X)
        loss_m = self.memory_loss(H, M)
        loss_d = -Dhat.mean()

        decoder_last_layer = decoder.network[-1].weight

        if lmbda is None:
            lmbda = self.calc_adaptive_weight(loss_rec, loss_d, decoder_last_layer)

        loss = (
            loss_rec + self.alpha * loss_m + self.discriminator_weight * lmbda * loss_d
        )

        return loss, (loss_rec, loss_m, loss_d, lmbda)


class State:
    def __init__(self) -> None:
        self.epoch = 0
        self.iteration = 0


class MATS(nn.Module):
    def __init__(self, dim_c, size_m, dim_e, disc_start=800) -> None:
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

        self.disc_start = disc_start

    def encode(self, X):
        X = torch.movedim(X, 1, 2)  # BATCH_SIZE * DIM_C * DIM_T
        H = self.encoder(X)  # BATCH_SIZE * DIM_D * DIM_T2
        C = self.memory_bank(H)  # BATCH_SIZE * DIM_M * DIM_T2
        return C, H

    def decode(self, C):
        Hhat = self.memory_bank.reconstruct(C)  # BATCH_SIZE * DIM_D * DIM_T2
        Xhat = self.decoder(Hhat)  # BATCH_SIZE * DIM_C * DIM_T
        return Xhat

    def forward_stage_1(self, X):
        C, H = self.encode(X)  # C : BATCH_SIZE * DIM_M * DIM_T2
        Xhat = self.decode(C).movedim(1, 2)  # BATCH_SIZE * DIM_T * DIM_C
        return Xhat, C, H

    def training_step_stage_1(self, X, optim_index, lmbda=None):
        criterion_edm = EDMLoss()
        criterion_discriminator = DiscriminatorLoss()

        self.optim_edm.zero_grad()
        self.optim_discriminator.zero_grad()

        Xhat, _, H = self.forward_stage_1(X)

        if optim_index == 0:
            # (4)
            Dhat = self.discriminator(Xhat.movedim(1, 2))  #  BATCH_SIZE * 1 * DIM_T2
            loss, partial_losses = criterion_edm(
                Xhat, X, H, self.memory_bank.units, Dhat, self.decoder, lmbda
            )
            loss.backward()
            self.optim_edm.step()
        else:
            # (3)
            D = self.discriminator(X.movedim(1, 2).detach())  # BATCH_SIZE * 1 * DIM_T2
            #  BATCH_SIZE * 1 * DIM_T2
            Dhat = self.discriminator(Xhat.movedim(1, 2).detach())
            loss, partial_losses = criterion_discriminator(Dhat, D)
            loss.backward()
            self.optim_discriminator.step()

        return loss, partial_losses

    @torch.no_grad()
    def evaluate_stage_1(self, dataloader):
        device = next(self.parameters()).device
        criterion_edm = EDMLoss()

        tot_loss, tot_loss_rec, tot_loss_m = 0, 0, 0

        for X, _ in dataloader:
            X = X.to(device)
            Xhat, _, H = self.forward_stage_1(X)
            Dhat = self.discriminator(Xhat.movedim(1, 2))
            loss, (loss_rec, loss_m, loss_d, lmbda) = criterion_edm(
                Xhat, X, H, self.memory_bank.units, Dhat, self.decoder, lmbda=0
            )
            tot_loss += loss
            tot_loss_rec += loss_rec
            tot_loss_m += loss_m

        n = len(dataloader)

        return tot_loss / n, (tot_loss_rec / n, tot_loss_m / n)

    # @torch.no_grad()
    # def log_rec_stage_1(self, X, writer):
    #     X, _ = next(iter(val_loader))
    #     X = X[0].to(device).unsqueeze(0)
    #     Xhat, _, _ = self.forward_stage_1(X)
    #     plt.plot(range(Xhat.shape[2]), Xhat[0, :, 0].cpu(), label="Xhat")
    #     plt.plot(range(X.shape[2]), X[0, :, 0].cpu(), label="X")
    #     plt.legend()
    #     writer.add_figure("S1_Rec", plt.gcf(), self.state_1.iteration)

    def train_stage_1(
        self,
        train_loader,
        val_loader,
        epochs,
        save_path,
        writer,
        patience=5,
        check_every=10
    ):
        device = next(self.parameters()).device
        iteration = self.state_1.iteration
        pbar = tqdm(range(self.state_1.epoch, epochs), leave=False)

        for epoch in pbar:
            for X, _ in train_loader:
                X = X.to(device)
                optim_index = (
                    0 if iteration < self.disc_start or iteration % 2 == 0 else 1
                )
                lmbda = 0 if iteration < self.disc_start else None

                loss, partial_losses = self.training_step_stage_1(X, optim_index, lmbda)

                if optim_index == 0:
                    loss_rec, loss_m, loss_d, lmbda = partial_losses
                    writer.add_scalar("S1_Loss/EDM", loss, iteration)
                    writer.add_scalars(
                        "S1_Partial_losses",
                        {"rec": loss_rec, "mem": loss_m, "disc": loss_d},
                        iteration,
                    )
                    writer.add_scalar("Lambda", lmbda, iteration)
                else:
                    writer.add_scalar("S1_Loss/Disc", loss, iteration)

                iteration = iteration + 1

            self.state_1.iteration = iteration
            self.state_1.epoch = epoch + 1

            # Model save - Early stopping
            if epoch % check_every == 0 :
                loss_train, _ = self.evaluate_stage_1(train_loader)
                loss_val, _ = self.evaluate_stage_1(val_loader)

                # Saving losses info
                writer.add_scalars(
                    "S1_Loss",
                    {"train": loss_train, "val": loss_val},
                    iteration,
                )

                # Increment patience
                if loss_val >= best_loss_val and best_loss_val != -1:
                    no_improvement_checks += 1

                # Save new best loss val and reset patience counter
                else :
                    # We save the new model as it is better than the last one
                    with save_path.open("wb") as fp:
                        torch.save(self, fp)

                    # We reset the values and save the new best loss val
                    no_improvement_checks = 0
                    best_loss_val = loss_val

                # Early stopping
                if no_improvement_checks == patience :
                    loss_train, _ = self.evaluate_stage_1(train_loader)
                    loss_val, _ = self.evaluate_stage_1(val_loader)

                    writer.add_scalars(
                        "S1_Loss",
                        {"train": loss_train, "val": loss_val},
                        iteration,
                    )
                    
                    return

    def get_dim_h2(self, dim_t, dim_t2, horizon):
        return np.ceil(dim_t2 * horizon / dim_t).astype(int)

    def forward_stage_2(self, X, horizon=None, teacher_prob=0, C_gt=None):
        C, _ = self.encode(X)  # BATCH_SIZE * DIM_M * DIM_T2
        C = C.movedim((0, 1, 2), (1, 2, 0))  # DIM_T2 * BATCH_SIZE * DIM_M
        
        if teacher_prob > 0 and C_gt is not None:
            C_gt = C_gt.movedim((0, 1, 2), (1, 2, 0))  # (DIM_T2+DIM_H2) * BATCH_SIZE * DIM_M
            Chat = self.predictor(C, teacher_prob=teacher_prob, X_gt=C_gt)  # (DIM_T2 + DIM_H2) * BATCH_SIZE * DIM_M
        else:
            assert horizon is not None
            dim_h2 = self.get_dim_h2(X.shape[1], C.shape[0], horizon)
            Chat = self.predictor(C, horizon=dim_h2)  # (DIM_T2 + DIM_H2) * BATCH_SIZE * DIM_M

        Chat = Chat.movedim((0, 1, 2), (2, 0, 1)) # BATCH_SIZE * DIM_M * (DIM_T2 + DIM_H2)
        return Chat

    def training_step_stage_2(self, X, y, teacher_prob):
        criterion_predictor = nn.BCELoss()
        self.optim_predictor.zero_grad()

        # (6)
        X_gt = torch.cat((X, y), dim=1)  # BATCH_SIZE * (DIM_T + DIM_H) * DIM_C
        C_gt, _ = self.encode(X_gt)  # BATCH_SIZE * DIM_M * (DIM_T2 + DIM_H2)
        
        Chat = self.forward_stage_2(X, teacher_prob=teacher_prob, C_gt=C_gt)

        # (7)
        loss = criterion_predictor(Chat, C_gt)
        loss.backward()
        self.optim_predictor.step()

        return loss

    @torch.no_grad()
    def evaluate_stage_2(self, dataloader):
        device = next(self.parameters()).device
        criterion_predictor = nn.BCELoss(reduction="sum")
        tot_mse, tot_loss, n = 0, 0, 0

        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            X_gt = torch.cat((X, y), dim=1)  # BATCH_SIZE * (DIM_T + DIM_H) * DIM_C
            C_gt, _ = self.encode(X_gt)  # BATCH_SIZE * DIM_M * (DIM_T2 + DIM_H2)
            
            Chat = self.forward_stage_2(X, horizon=y.shape[1])

            loss = criterion_predictor(Chat, C_gt)
            tot_loss += loss

            Xhat = self.decode(Chat).movedim(1, 2)  # BATCH_SIZE * DIM_T * DIM_C
            Xpred = Xhat[:, -y.shape[1] :, :]  # BATCH_SIZE * DIM_H * DIM_C

            mse = F.mse_loss(Xpred, y, reduction="sum")
            tot_mse += mse
            n += np.prod(X.shape)

        return tot_mse / n, tot_loss / n

    def train_stage_2(self,
        train_loader,
        val_loader,
        epochs,
        save_path,
        writer,
        patience=5,
        check_every=10
    ):
        device = next(self.parameters()).device
        iteration = self.state_2.iteration
        pbar = tqdm(range(self.state_2.epoch, epochs), leave=False)
        
        k = 1
        steps = torch.linspace(-6, 6, epochs * len(train_loader))
        teacher_prob_schedule = k / (k + np.exp(steps / k))
        
        for epoch in pbar:
            for X, y in train_loader:
                X = X.to(device)
                y = y.to(device)
                teacher_prob = teacher_prob_schedule[iteration]
                loss = self.training_step_stage_2(X, y, teacher_prob=teacher_prob)
                iteration = iteration + 1

            self.state_2.iteration = iteration
            self.state_2.epoch = epoch + 1

            # Model save - Early stopping
            if epoch % check_every == 0 :
                mse_train, loss_train = self.evaluate_stage_2(train_loader)
                mse_val, loss_val = self.evaluate_stage_2(val_loader)

                # Saving losses - results info
                writer.add_scalars(
                    "S2_MSE", {"train": mse_train, "val": mse_val}, iteration
                )
                writer.add_scalars(
                    "S2_Loss", {"train": loss_train, "val": loss_val}, iteration
                )

                # Increment patience
                if loss_val >= best_loss_val and best_loss_val != -1 :
                    no_improvement_checks += 1

                # Save new best loss val and reset patience counter
                else :
                    # We save the new model as it is better than the last one
                    with save_path.open("wb") as fp:
                        torch.save(self, fp)

                    # We reset the values and save the new best loss val
                    no_improvement_checks = 0
                    best_loss_val = loss_val

                # Early stopping
                if no_improvement_checks == patience :
                    print("Early stopping - Stage 2 (epoch : {epoch})")
                    
                    return

    def freeze_stage_1(self):
        list_models = [
            self.encoder,
            self.decoder,
            self.discriminator,
            self.memory_bank,
        ]
        for model in list_models:
            for param in model.parameters():
                param.requires_grad = False

    def fit(
        self,
        train_loader_1,
        val_loader_1,
        train_loader_2,
        val_loader_2,
        epochs_1,
        epochs_2,
        save_path_1,
        save_path_2,
        writer,
        patience=5,
        check_every=10
    ):
        self.train()
        self.train_stage_1(train_loader_1, val_loader_1, epochs_1, save_path_1, writer, 
                            patience=patience, check_every=check_every)
        self.freeze_stage_1()
        self.train_stage_2(train_loader_2, val_loader_2, epochs_2, save_path_2, writer,
                           patience=patience, check_every=check_every)

    @torch.no_grad()
    def predict(self, X, horizon):
        Chat = self.forward_stage_2(X, horizon)
        Xhat = self.decode(Chat).movedim(1, 2)  # BATCH_SIZE * DIM_T * DIM_C
        Xpred = Xhat[:, -horizon:, :]  # BATCH_SIZE * DIM_H * DIM_C
        return Xpred

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        device = next(self.parameters()).device
        tot_mse = 0
        tot_mae = 0
        n = 0

        for X, y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)
            Xpred = self.predict(X, horizon=y.shape[1])
            mse = F.mse_loss(Xpred, y, reduction="sum")
            mae = F.l1_loss(Xpred, y, reduction="sum")
            tot_mse += mse.item()
            tot_mae += mae.item()
            n += np.prod(X.shape)

        return tot_mse / n, tot_mae / n

    def forward(self, X, horizon):
        return self.predict(X, horizon)
