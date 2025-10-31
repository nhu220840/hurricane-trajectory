# src/dataset.py

import torch
from torch.utils.data import Dataset

class StormSeqDataset(Dataset):
    """
    Dataset for one-step delta forecasting:
      X: (B, N_IN, d)
      Y: (B, 2)  # Δlat, Δlon (scaled)
      last_obs_latlon: (B, 2)  # original coordinates at t (end of window), used to reconstruct predicted coordinates in evaluate
    """
    def __init__(self, X, Y, last_obs_latlon):
        self.X = torch.tensor(X, dtype=torch.float32)
        # If Y is (B,1,2), reshape to (B,2). In data_processing it's already (B,2).
        if Y.ndim == 3 and Y.shape[1] == 1:
            Y = Y[:, 0, :]
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.last_obs = torch.tensor(last_obs_latlon, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.last_obs[idx]