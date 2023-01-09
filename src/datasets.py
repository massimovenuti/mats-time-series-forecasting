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



def data_to_hour(df,begin=4*24*365,frequency=4) :
    out = []
    data = df.to_numpy()
    n = len(data)

    for i in range(begin,n,frequency) :
        date = data[i,0].split(':')[0]
        data_houre = data[i:i+frequency,1:].mean(axis=0)
        out.append( [date] + list(data_houre) ) 

    return pd.DataFrame(out)

def data_filter_client(df,number_client=320):
    set_index_client = {0}
    index = 0

    while len(set_index_client) < number_client :
        
        for i,value in enumerate(df.iloc[index,1:]) :
            if value != 0:
                set_index_client.add(i+1)
        index += 1
        
    return df.iloc[:,list(set_index_client)]




def load_ld_dataset(
    path, train_proportion=0.7, val_proportion=0.1, normalize=True, dim_t=192, dim_h=96
):
    data = pd.read_csv(path, decimal=".", sep=",")
    data = data_to_hour(data)
    data = data_filter_client(data)
    data = data.drop(columns=data.columns[0], axis=1)

    data_train, data_val, data_test = train_val_test_split(
        data, train_proportion, val_proportion, normalize
    )

    train_dataset = TimeSeriesDataset(data_train, dim_t, dim_h)
    val_dataset = TimeSeriesDataset(data_val, dim_t, dim_h)
    test_dataset = TimeSeriesDataset(data_test, dim_t, dim_h)

    return train_dataset, val_dataset, test_dataset




# ETT dataset management
def load_ETT_dataset(
    path="../data/ETT/ETT", choice="h1", train_proportion=0.6, val_proportion=0.2, 
    normalize=True, dim_t=60, dim_h=24
):
    """ path: path to the file containing the ETT data (by default "../data/ETT/ETT*")
    \nWarning - Do not specify the file ending in the path ('*h1.txt' for example)
    \nchoice: data to be used (by default "h1", choose between {'h1', 'h2', 'm1', 'm2'})
    \ntrain_proportion: proportion of values for the training set (by default 0.7)
    \nval_proportion: proportion of values for the validation set (by default 0.1)
    \ndim_t: (by default 96 in Stage 1)
    \ndim_h: (by default 96, choose value in {96, 192, 336, 720})
    \nAll default values except 'path' are from the paper, section 4. Experiments
    """
    # File selection
    if choice not in ["h1", "h2", "m1", "m2"]:
        path += "h1.txt" # In case there's an error in choice, h1 is the default choice
    else :
        path += choice + ".txt"

    # CSV file into pandas conversion and cleaning
    data = pd.read_csv(path)
    data = data.select_dtypes([np.number]) # Removing non-numeric columns

    # Dataset split
    data_train, data_val, data_test = train_val_test_split(
        data, train_proportion, val_proportion, normalize
    )
    train_dataset = TimeSeriesDataset(data_train, dim_t, dim_h)
    val_dataset = TimeSeriesDataset(data_val, dim_t, dim_h)
    test_dataset = TimeSeriesDataset(data_test, dim_t, dim_h)

    # Returning the split
    return train_dataset, val_dataset, test_dataset




# Exchange dataset management
def load_exchange_dataset(
    path="../data/Exchange/exchange_rate.txt", train_proportion=0.7, val_proportion=0.1, 
    normalize=True, dim_t=192, dim_h=96
):
    """ path: path to the file containing the Exchange data (by default "../data/Exchange/exchange_rate.txt")
    \ntrain_proportion: proportion of values for the training set (by default 0.7)
    \nval_proportion: proportion of values for the validation set (by default 0.1)
    \ndim_t: (by default 96 in Stage 1)
    \ndim_h: (by default 96, choose value in {96, 192, 336, 720})
    \nAll default values except 'path' are from the paper, section 4. Experiments
    """

    # CSV file into pandas conversion and cleaning
    data = pd.read_csv(path, decimal=".", sep=",")
    data = data.select_dtypes([np.number]) # Removing non-numeric columns

    # Dataset split
    data_train, data_val, data_test = train_val_test_split(
        data, train_proportion, val_proportion, normalize
    )
    train_dataset = TimeSeriesDataset(data_train, dim_t, dim_h)
    val_dataset = TimeSeriesDataset(data_val, dim_t, dim_h)
    test_dataset = TimeSeriesDataset(data_test, dim_t, dim_h)

    # Returning the split
    return train_dataset, val_dataset, test_dataset



# ILI dataset management
def load_ILI_dataset(
    path="../data/ILI/ILINet.csv", train_proportion=0.6, val_proportion=0.2, 
    normalize=True, dim_t=60, dim_h=24
):
    """ path: path to the file containing the ILI data (by default "../data/ILI/ILINet.csv")
    \ntrain_proportion: proportion of values for the training set (by default 0.6)
    \nval_proportion: proportion of values for the validation set (by default 0.2)
    \ndim_t: (by default 60)
    \ndim_h: (by default 24, choose value in {24, 36, 48, 60})
    \nAll default values except 'path' are from the paper, section 4. Experiments
    """

    # CSV file into pandas conversion and cleaning
    data = pd.read_csv(path)
    data = data.select_dtypes([np.number]) # Removing non-numeric columns
    # data = data.drop(['AGE 0-4','AGE 5-24','AGE 65'], axis=1) # Can be discussed

    # Dataset split
    data_train, data_val, data_test = train_val_test_split(
        data, train_proportion, val_proportion, normalize
    )
    train_dataset = TimeSeriesDataset(data_train, dim_t, dim_h)
    val_dataset = TimeSeriesDataset(data_val, dim_t, dim_h)
    test_dataset = TimeSeriesDataset(data_test, dim_t, dim_h)

    # Returning the split
    return train_dataset, val_dataset, test_dataset

# TODO: Fix this ILI error when used in Stage 2
# ValueError: Using a target size (torch.Size([7, 64, 16])) that is different to the input size 
# (torch.Size([13, 64, 16])) is deprecated. Please ensure they have the same size.




# Traffic dataset management
def load_traffic_dataset(
    path="../data/Traffic/traffic-5-years.txt", train_proportion=0.7, val_proportion=0.1, 
    normalize=True, dim_t=192, dim_h=96
):
    """ path: path to the file containing the Traffic data (by default "../data/Traffic/traffic-5-years.txt")
    \ntrain_proportion: proportion of values for the training set (by default 0.7)
    \nval_proportion: proportion of values for the validation set (by default 0.1)
    \ndim_t: (by default 96 in Stage 1)
    \ndim_h: (by default 96, choose value in {96, 192, 336, 720})
    \nAll default values except 'path' are from the paper, section 4. Experiments
    """

    # CSV file into pandas conversion and cleaning
    data = pd.read_csv(path, header=None, skiprows=1)
    # Every entry is actually one big String, need to split multiple times, rename and drop
    data[['Date', 'VMT']] = data[0].str.split("\t", expand=True)
    data[['Miles', '# Lane Points', '% Observed']] = data[2].str.split("\t", expand=True)
    data.rename(columns={1: "Veh"}, inplace=True)
    data.drop([0, 2, 'Date'], inplace=True, axis=1)
    # Transform all columns into numeric columns
    data[data.columns] = data[data.columns].apply(pd.to_numeric, errors='coerce') 

    # Dataset split
    data_train, data_val, data_test = train_val_test_split(
        data, train_proportion, val_proportion, normalize
    )
    train_dataset = TimeSeriesDataset(data_train, dim_t, dim_h)
    val_dataset = TimeSeriesDataset(data_val, dim_t, dim_h)
    test_dataset = TimeSeriesDataset(data_test, dim_t, dim_h)

    # Returning the split
    return train_dataset, val_dataset, test_dataset


# Weather dataset management
def load_weather_dataset(
    path="../data/Weather/mpi_roof_2020.csv", train_proportion=0.7, val_proportion=0.1, 
    normalize=True, dim_t=192, dim_h=96
):
    """ path: path to the file containing the Weather data (by default "../data/Weather/mpi_roof_2020.csv")
    \ntrain_proportion: proportion of values for the training set (by default 0.7)
    \nval_proportion: proportion of values for the validation set (by default 0.1)
    \ndim_t: (by default 96 in Stage 1)
    \ndim_h: (by default 96, choose value in {96, 192, 336, 720})
    \nAll default values except 'path' are from the paper, section 4. Experiments
    """

    # CSV file into pandas conversion and cleaning
    data = pd.read_csv(path)
    data = data.select_dtypes([np.number])

    # Dataset split
    data_train, data_val, data_test = train_val_test_split(
        data, train_proportion, val_proportion, normalize
    )
    train_dataset = TimeSeriesDataset(data_train, dim_t, dim_h)
    val_dataset = TimeSeriesDataset(data_val, dim_t, dim_h)
    test_dataset = TimeSeriesDataset(data_test, dim_t, dim_h)

    # Returning the split
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