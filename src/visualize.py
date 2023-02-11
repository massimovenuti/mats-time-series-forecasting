import datasets
import torch
import mats
from pathlib import Path
import config
from matplotlib import pyplot as plt


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = "weather"
    path = Path(config.DATA_DIR, "weather.csv")
    dim_h = 720

    # Load dataset
    train_loader, test_loader = datasets.get_loaders(
        dataset=dataset,
        path=path,
        dim_h=dim_h,
        dim_t=config.DIM_T_2,
        univariate=True,
        shuffle=False,
    )

    # Load model
    save_path = Path(
        config.STATES_DIR, dataset, f"mats_{dataset}_univariate_h_{dim_h}.pkl"
    )
    with save_path.open("rb") as fp:
        model = torch.load(fp).to(device)

    X, y = next(iter(test_loader))
    X = X[:1, :].to(device)
    y = y[:1, :].to(device)

    Xpred = model(X, dim_h)

    X = X.cpu().squeeze()
    y = y.cpu().squeeze()
    Xpred = Xpred.cpu().squeeze()

    fig = plt.figure(figsize=(10, 3))
    plt.plot(range(len(X)), X, label="History")
    plt.plot(range(len(X), len(X) + dim_h), y, label="Ground truth")
    plt.plot(range(len(X), len(X) + dim_h), Xpred[-dim_h:], label="Prediction")
    plt.axvline(len(X), linestyle="--", color="black")
    plt.legend()
    # fig = plt.gcf()
    fig.tight_layout()

    fig.savefig(Path(config.RESULTS_DIR, f"{dataset}_h_{dim_h}.pdf"))
