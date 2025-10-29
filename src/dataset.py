# src/dataset.py

import torch
from torch.utils.data import Dataset

class StormSeqDataset(Dataset):
    """
    Dataset cho one-step delta forecasting:
      X: (B, N_IN, d)
      Y: (B, 2)  # Δlat, Δlon (đã scale)
      last_obs_latlon: (B, 2)  # toạ độ gốc tại t (cuối cửa sổ), dùng để tái dựng tọa độ dự đoán ở evaluate
    """
    def __init__(self, X, Y, last_obs_latlon):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Nếu Y là (B,1,2) thì nắn về (B,2). Trong data_processing đã là (B,2).
        if Y.ndim == 3 and Y.shape[1] == 1:
            Y = Y[:, 0, :]
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.last_obs = torch.tensor(last_obs_latlon, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.last_obs[idx]
