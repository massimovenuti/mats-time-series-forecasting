from torch.utils import data
import pandas as pd
import numpy as np
from sklearn import preprocessing


def train_val_test_split(data, train_proportion, val_proportion, normalize):
    train_split_idx = round(len(data) * train_proportion)
    val_split_idx = train_split_idx + round(len(data) * val_proportion)

    data_train = data[:train_split_idx]
    data_val = data[train_split_idx:val_split_idx]
    data_test = data[val_split_idx:]

    if normalize:
        scaler = preprocessing.StandardScaler()

        scaler.fit(data_train)

        data_train = scaler.transform(data_train)
        data_val = scaler.transform(data_val)
        data_test = scaler.transform(data_test)

    return data_train, data_val, data_test


def load_ld_dataset(
    path, train_proportion=0.7, val_proportion=0.1, normalize=True, dim_t=192, dim_h=96
):
    data = pd.read_csv(path, decimal=",", sep=";")
    data = data.drop(columns=data.columns[0], axis=1)

    data_train, data_val, data_test = train_val_test_split(
        data, train_proportion, val_proportion, normalize
    )

    train_dataset = TimeSeriesDataset(data_train, dim_t, dim_h)
    val_dataset = TimeSeriesDataset(data_val, dim_t, dim_h)
    test_dataset = TimeSeriesDataset(data_test, dim_t, dim_h)

    return train_dataset, val_dataset, test_dataset


class TimeSeriesDataset(data.Dataset):
    def __init__(self, data, dim_t=192, dim_h=96):
        self.data = data.astype(np.float32)
        self.dim_t = dim_t
        self.dim_h = dim_h

    def __len__(self):
        return len(self.data) - self.dim_t - self.dim_h + 1

    def __getitem__(self, indice):
        X = self.data[indice : indice + self.dim_t]
        y = self.data[indice + self.dim_t : indice + self.dim_t + self.dim_h]
        return X, y
