# src/train.py

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    PROCESSED_NPZ,
    EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, PATIENCE, FACTOR, CLIP_NORM,
    LSTM_TORCH, LSTM_SCRATCH, DEVICE, SEED,
    CHECKPOINT_LSTM_TORCH, CHECKPOINT_LSTM_SCRATCH
)
from .dataset import StormSeqDataset
from .models import LSTMForecaster, LSTMFromScratchForecaster


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_by_sid(window_sid_idx, train_ratio=0.7, val_ratio=0.15, seed=SEED):
    uniq = np.unique(window_sid_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_sids = set(uniq[:n_train])
    val_sids = set(uniq[n_train:n_train+n_val])
    test_sids = set(uniq[n_train+n_val:])
    return train_sids, val_sids, test_sids


def _filter_by_sid_idx(arr, sid_idx, keep_sids):
    mask = np.isin(sid_idx, list(keep_sids))
    return arr[mask], mask


def _finite_batch(Xb, Yb):
    x_ok = torch.isfinite(Xb).all()
    y_ok = torch.isfinite(Yb).all()
    return bool(x_ok and y_ok)


def train_one_model(model_name: str):
    set_seed(SEED)
    use_device = "cuda" if (torch.cuda.is_available() and DEVICE == "cuda") else "cpu"

    data = np.load(PROCESSED_NPZ, allow_pickle=True)
    X = data["X"]                          # (B, N_IN, d)
    Y = data["Y"]                          # (B, 2) delta (scaled)
    last_obs = data["last_obs_latlon"]     # (B, 2)
    sid_idx = data["window_sid_idx"]       # (B,)

    # guard: if still bad
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("X/Y vẫn có NaN/Inf sau tiền xử lý. Kiểm tra lại data_processing.py.")

    train_sids, val_sids, test_sids = _split_by_sid(sid_idx, seed=SEED)

    X_train, m_tr = _filter_by_sid_idx(X, sid_idx, train_sids)
    Y_train = Y[m_tr]
    last_tr = last_obs[m_tr]

    X_val, m_val = _filter_by_sid_idx(X, sid_idx, val_sids)
    Y_val = Y[m_val]
    last_val = last_obs[m_val]

    X_test, m_te = _filter_by_sid_idx(X, sid_idx, test_sids)
    Y_test = Y[m_te]
    last_test = last_obs[m_te]

    ds_tr = StormSeqDataset(X_train, Y_train, last_tr)
    ds_val = StormSeqDataset(X_val, Y_val, last_val)
    ds_te = StormSeqDataset(X_test, Y_test, last_test)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    input_size = X.shape[-1]

    if model_name == "pytorch":
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=LSTM_TORCH["hidden_size"],
            num_layers=LSTM_TORCH["num_layers"],
            dropout=LSTM_TORCH["dropout"]
        )
        ckpt_path = CHECKPOINT_LSTM_TORCH
    elif model_name == "scratch":
        model = LSTMFromScratchForecaster(
            input_size=input_size,
            hidden_size=LSTM_SCRATCH["hidden_size"],
            num_layers=LSTM_SCRATCH["num_layers"],
            dropout=LSTM_SCRATCH["dropout"]
        )
        ckpt_path = CHECKPOINT_LSTM_SCRATCH
    else:
        raise ValueError("model_name phải là 'pytorch' hoặc 'scratch'.")

    model = model.to(use_device)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=FACTOR, patience=max(1, PATIENCE//2))
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    wait = 0
    for epoch in range(1, EPOCHS+1):
        # ---- train ----
        model.train()
        loss_sum = 0.0
        n_seen = 0
        skipped = 0
        for Xb, Yb, _ in dl_tr:
            Xb = Xb.to(use_device)
            Yb = Yb.to(use_device)
            if not _finite_batch(Xb, Yb):
                skipped += Xb.size(0)
                continue
            pred = model(Xb)        # (B,2)
            if not torch.isfinite(pred).all():
                skipped += Xb.size(0)
                continue
            loss = criterion(pred, Yb)
            if not torch.isfinite(loss):
                skipped += Xb.size(0)
                continue
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optim.step()
            loss_sum += loss.item() * Xb.size(0)
            n_seen += Xb.size(0)

        loss_tr = float("nan") if n_seen == 0 else (loss_sum / n_seen)

        # ---- val ----
        model.eval()
        val_sum = 0.0
        n_val_seen = 0
        with torch.no_grad():
            for Xb, Yb, _ in dl_val:
                Xb = Xb.to(use_device); Yb = Yb.to(use_device)
                if not _finite_batch(Xb, Yb):
                    continue
                pred = model(Xb)
                if not torch.isfinite(pred).all():
                    continue
                loss = criterion(pred, Yb)
                if not torch.isfinite(loss):
                    continue
                val_sum += loss.item() * Xb.size(0)
                n_val_seen += Xb.size(0)
        loss_va = float("nan") if n_val_seen == 0 else (val_sum / n_val_seen)

        if np.isfinite(loss_va):
            sched.step(loss_va)

        info_skipped = f" | skipped_train={skipped}" if skipped else ""
        print(f"[{model_name}] Epoch {epoch:03d}/{EPOCHS} | train={loss_tr:.6f} | val={loss_va:.6f}{info_skipped}")

        if np.isfinite(loss_va) and (loss_va < best_val):
            best_val = loss_va
            wait = 0
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[{model_name}] Early stopping at epoch {epoch}. Best val={best_val:.6f}")
                break

    print(f"[{model_name}] Best checkpoint saved: {ckpt_path}")
    return ckpt_path, (X_test, Y_test, last_test)


def train(model_choice: str):
    if model_choice == "all":
        train_one_model("pytorch")
        train_one_model("scratch")
    else:
        train_one_model(model_choice)
