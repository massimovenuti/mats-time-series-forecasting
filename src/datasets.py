import torch
from torch.utils.data import Dataset
import utils


class DatasetLd(Dataset):
    def __init__(self, path="../data/LD2011_2014/LD2011_2014.pkl", DIM_T=192, DIM_H=96):
        self.data = utils.lectureFichier(path)
        self.DIM_T = DIM_T
        self.DIM_H = DIM_H

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indice):
        X = self.data[indice : indice + self.DIM_T, :]
        y = self.data[indice + self.DIM_T : indice + self.DIM_T + self.DIM_H, :]
        return (X, y)

    def __collate_fn__(self, batch):
        X = torch.stack([torch.tensor(item[0], dtype=torch.float32) for item in batch])
        y = torch.stack([torch.tensor(item[1], dtype=torch.float32) for item in batch])
        return X, y
