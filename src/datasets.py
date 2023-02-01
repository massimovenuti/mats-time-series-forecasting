from torch.utils import data
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pathlib
import torch


def train_test_split(data, train_size, val_size=0, normalize=True):
    if val_size > 0:
        data_train, data_val, data_test = np.split(
            data,
            [int(train_size * len(data)), int((train_size + val_size) * len(data))],
        )
    else:
        data_train, data_test = np.split(data, [int(train_size * len(data))])

    if normalize:
        scaler = preprocessing.StandardScaler()
        scaler.fit(data_train)

        data_train = scaler.transform(data_train)
        data_test = scaler.transform(data_test)
        if val_size > 0:
            data_val = scaler.transform(data_val)

    if val_size == 0:
        return data_train, data_test

    return data_train, data_val, data_test


def get_loaders(
    dataset,
    path,
    batch_size=64,
    train_size=0.8,
    val_size=0,
    dim_t=192,
    dim_h=96,
    normalize=True,
    univariate=False,
):
    datasets = ["electricity", "exchange", "ett", "ili", "traffic", "weather"]
    assert dataset in datasets

    load_fonctions = [
        load_elec_dataset,
        load_exchange_dataset,
        load_ETT_dataset,
        load_ILI_dataset,
        load_traffic_dataset,
        load_weather_dataset,
    ]

    dataset_index = datasets.index(dataset)
    load_fn = load_fonctions[dataset_index]

    df = load_fn(path)
    df = df.select_dtypes([np.number])

    if univariate:
        df = df.iloc[:, -1]

    # Dataset split
    if val_size > 0:
        data_train, data_val, data_test = train_test_split(
            df, train_size, val_size, normalize
        )
    else:
        data_train, data_test = train_test_split(df, train_size, val_size, normalize)

    train_loader = data.DataLoader(
        TimeSeriesDataset(data_train, dim_t, dim_h),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = data.DataLoader(
        TimeSeriesDataset(data_test, dim_t, dim_h),
        batch_size=batch_size,
        shuffle=True,
    )

    if val_size == 0:
        return train_loader, test_loader

    val_loader = data.DataLoader(
        TimeSeriesDataset(data_val, dim_t, dim_h),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, val_loader, test_loader


# Electricity dataset management
def load_elec_dataset(path):
    """path: path to the file containing the Electricity data (by default "../data/Electricity/LD2011_2014.txt")
    \ntrain_proportion: proportion of values for the training set (by default 0.7)
    \nval_proportion: proportion of values for the validation set (by default 0.1)
    \ndim_t: (by default 192 in Stage 1)
    \ndim_h: (by default 96, choose value in {96, 192, 336, 720})
    \nshowShape: will stop the execution after the dataset creation to show it's size, no returns
    \nAll default values except 'path' are from the paper, section 4. Experiments
    """
    df = pd.read_csv(path, decimal=".", sep=",", header=None)
    return df


# ETT dataset management
def load_ETT_dataset(path):
    """path: path to the file containing the ETT data (by default "../data/ETT/ETT*")
    \nWarning - Do not specify the file ending in the path ('*h1.txt' for example)
    \nchoice: data to be used (by default "h1", choose between {'h1', 'h2', 'm1', 'm2'})
    \ntrain_proportion: proportion of values for the training set (by default 0.7)
    \nval_proportion: proportion of values for the validation set (by default 0.1)
    \ndim_t: (by default 192 in Stage 1)
    \ndim_h: (by default 96, choose value in {96, 192, 336, 720})
    \nshowShape: will stop the execution after the dataset creation to show it's size, no returns
    \nAll default values except 'path' are from the paper, section 4. Experiments
    """
    # CSV file into pandas conversion and cleaning
    df = pd.read_csv(path)
    df = df.select_dtypes([np.number])  # Removing non-numeric columns
    return df


# Exchange dataset management
def load_exchange_dataset(path):
    """path: path to the file containing the Exchange data (by default "../data/Exchange/exchange_rate.txt")
    \ntrain_proportion: proportion of values for the training set (by default 0.7)
    \nval_proportion: proportion of values for the validation set (by default 0.1)
    \ndim_t: (by default 192 in Stage 1)
    \ndim_h: (by default 96, choose value in {96, 192, 336, 720})
    \nshowShape: will stop the execution after the dataset creation to show it's size, no returns
    \nAll default values except 'path' are from the paper, section 4. Experiments
    """
    # CSV file into pandas conversion and cleaning
    df = pd.read_csv(path, decimal=".", sep=",", header=None)
    return df


# ILI dataset management
def load_ILI_dataset(path):
    """path: path to the file containing the ILI data (by default "../data/ILI/ILINet.csv")
    \ntrain_proportion: proportion of values for the training set (by default 0.6)
    \nval_proportion: proportion of values for the validation set (by default 0.2)
    \ndim_t: (by default 60)
    \ndim_h: (by default 24, choose value in {24, 36, 48, 60})
    \nshowShape: will stop the execution after the dataset creation to show it's size, no returns
    \nAll default values except 'path' are from the paper, section 4. Experiments
    """

    # CSV file into pandas conversion and cleaning
    df = pd.read_csv(path)
    df.drop(
        ["YEAR", "WEEK", "REGION TYPE", "REGION", "% WEIGHTED ILI", "%UNWEIGHTED ILI"],
        inplace=True,
        axis=1,
    )  # Removing percentage data and timestamps
    # Need to compute missing AGE 25-64 from AGE 25-49 and AGE 50-64
    # Missing values are represented by 'X'
    def compute_total(d):
        if d["AGE 25-64"] == "X":
            return int(d["AGE 25-49"]) + int(d["AGE 50-64"])
        return int(d["AGE 25-64"])

    df["AGE 25-64"] = df.apply(compute_total, axis=1)
    # We remove the useless columns to only keep 7 like shown in the paper
    # AGE 25-49 and AGE 50-64 are useless if we have AGE 25-64
    df.drop(["AGE 25-49", "AGE 50-64"], inplace=True, axis=1)
    # We remove the last 26 lines of the dataset to match the expected number of lines from the paper
    df.drop(df.tail(26).index, inplace=True)

    return df


# Traffic dataset management
def load_traffic_dataset(path):
    """path: path to the file containing the Traffic data (by default "../data/Traffic/traffic-5-years.txt")
    \ntrain_proportion: proportion of values for the training set (by default 0.7)
    \nval_proportion: proportion of values for the validation set (by default 0.1)
    \ndim_t: (by default 192 in Stage 1)
    \ndim_h: (by default 96, choose value in {96, 192, 336, 720})
    \nshowShape: will stop the execution after the dataset creation to show it's size, no returns
    \nAll default values except 'path' are from the paper, section 4. Experiments
    """

    # CSV file into pandas conversion and cleaning
    df = pd.read_csv(path, decimal=".", sep=",", header=None)
    return df


# Weather dataset management
def load_weather_dataset(path):
    """path: path to the file containing the Weather data (by default "../data/Weather/mpi_roof_2020.csv")
    \ntrain_proportion: proportion of values for the training set (by default 0.7)
    \nval_proportion: proportion of values for the validation set (by default 0.1)
    \ndim_t: (by default 192 in Stage 1)
    \ndim_h: (by default 96, choose value in {96, 192, 336, 720})
    \nshowShape: will stop the execution after the dataset creation to show it's size, no returns
    \nAll default values except 'path' are from the paper, section 4. Experiments
    """

    # CSV file into pandas conversion and cleaning
    df = pd.read_csv(path)
    return df


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
