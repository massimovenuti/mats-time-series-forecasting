import datasets
import torch
import mats
import pandas as pd
from pathlib import Path
from torch.utils import tensorboard as tb
import time
import config


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_paths = {
        # "electricity": Path(config.DATA_DIR, "electricity.txt"),
        # "exchange": Path(config.DATA_DIR, "exchange.txt"),
        # "ett_h1": Path(config.DATA_DIR, "ett_h1.txt"),
        # "ett_h2": Path(config.DATA_DIR, "ett_h2.txt"),
        "ett_m1": Path(config.DATA_DIR, "ett_m1.txt"),
        # "ett_m2": Path(config.DATA_DIR, "ett_m2.txt"),
        # "traffic": Path(config.DATA_DIR, "traffic.txt"),
        # "weather": Path(config.DATA_DIR, "weather.csv"),
        # "ili": Path(config.DATA_DIR, "ili.csv"),
    }
    
    all_mse_train = {}
    all_mae_train = {}
    all_mse_test = {}
    all_mae_test = {}

    for i in range(len(config.DIMS_H)):
        
        for dataset, path in dataset_paths.items():
            dim_h = config.DIMS_H[i] if dataset != "ili" else config.DIMS_H_ILI[i]
            
            all_mse_train[dim_h] = {}
            all_mae_train[dim_h] = {}
            all_mse_test[dim_h] = {}
            all_mae_test[dim_h] = {}
        
            for variate in ["multivariate", "univariate"]:           
                writer = tb.SummaryWriter(Path(config.RUNS_DIR, dataset, f"mats_{dataset}_{variate}_h_{dim_h}"))
                
                save_path_1 = Path(config.STATES_DIR, dataset, f"mats_{dataset}_{variate}_stage_1.pkl")
                save_path_2 = Path(config.STATES_DIR, dataset, f"mats_{dataset}_{variate}_h_{dim_h}.pkl")
                save_path_2.parent.mkdir(exist_ok=True, parents=True)

                train_loader_1, test_loader_1 = datasets.get_loaders(
                    dataset=dataset,
                    path=path,
                    dim_t=config.DIM_T_1 if dataset != "ili" else config.DIM_T_1_ILI,
                    univariate=variate == "univariate",
                )

                train_loader_2, test_loader_2 = datasets.get_loaders(
                    dataset=dataset,
                    path=path,
                    dim_h=dim_h,
                    dim_t=config.DIM_T_2 if dataset != "ili" else config.DIM_T_2_ILI,
                    univariate=variate == "univariate",
                )

                if save_path_2.is_file():
                    with save_path_2.open("rb") as fp:
                        model = torch.load(fp)
                elif save_path_1.is_file():
                    with save_path.open("rb") as fp:
                        model = torch.load(fp)
                else:
                    dim_c = train_loader_1.dataset.data.shape[1]
                    model = mats.MATS(dim_c, config.SIZE_M, config.DIM_E)

                model = model.to(device)
                
                model.fit(
                    train_loader_1=train_loader_1,
                    val_loader_1=test_loader_1,
                    train_loader_2=train_loader_2,
                    val_loader_2=test_loader_2,
                    epochs_1=config.EPOCHS[dataset][variate][0],
                    epochs_2=config.EPOCHS[dataset][variate][1],
                    save_path_1=save_path_1,
                    save_path_2=save_path_2,
                    writer=writer,
                )

                mse, mae = model.evaluate(train_loader_2)
                print(f"[{dataset}][{variate}][{dim_h}]  MSE_train={mse:.2f}, MAE_train={mae:.2f}")

                all_mse_train[dim_h][dataset] = mse
                all_mae_train[dim_h][dataset] = mae

                mse, mae = model.evaluate(test_loader_2)
                print(f"[{dataset}][{variate}][{dim_h}] MSE_test={mse:.2f}, MAE_test={mae:.2f}")

                all_mse_test[dim_h][dataset] = mse
                all_mae_test[dim_h][dataset] = mae

                Path(config.RESULTS_DIR, "train/").mkdir(parents=True, exist_ok=True)
                Path(config.RESULTS_DIR, "test/").mkdir(parents=True, exist_ok=True)
                df_mse_train = pd.DataFrame.from_dict(all_mse_train).to_csv(
                    Path(config.RESULTS_DIR, "train/", f"mse_train_{variate}.csv", index_label="dataset")
                )
                df_mae_train = pd.DataFrame.from_dict(all_mae_train).to_csv(
                    Path(config.RESULTS_DIR, "train/", f"mae_train_{variate}.csv", index_label="dataset")
                )
                df_mse_test = pd.DataFrame.from_dict(all_mse_test).to_csv(
                    Path(config.RESULTS_DIR, "test/", f"mse_test_{variate}.csv", index_label="dataset")
                )
                df_mae_test = pd.DataFrame.from_dict(all_mae_test).to_csv(
                    Path(config.RESULTS_DIR, "test/", f"mae_test_{variate}.csv", index_label="dataset")
                )
