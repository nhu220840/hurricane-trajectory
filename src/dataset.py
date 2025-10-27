# src/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class StormSeqDataset(Dataset):
    def __init__(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.X = torch.tensor(X, dtype=torch.float32)

        # Lấy từ file REbuildLSTM.ipynb (robust hơn)
        if y.ndim >= 3 and y.shape[1] == 1:
            y = np.squeeze(y, axis=1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]