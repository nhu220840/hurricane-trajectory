# scripts/train_model.py
import numpy as np, torch, pickle
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchmetrics.regression import MeanAbsoluteError

# ===== CẤU HÌNH (giữ nguyên) =====
ROOT = Path(__file__).resolve().parents[1]
DATA_NPZ = ROOT / "data" / "processed_splits_delta.npz"
SCALER_PKL = ROOT / "data" / "scaler_delta.pkl"
# Lưu model RNN với tên mới để phân biệt
CKPT = ROOT / "models" / "best_rnn_delta.pt"
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 7


# ========================

class StormSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).squeeze(1)

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.y[i]


# --- THAY ĐỔI DUY NHẤT LÀ Ở ĐÂY ---
class RNNForecaster(nn.Module):  # Đổi tên class cho rõ ràng
    def __init__(self, in_dim=6, hidden=128, num_layers=2, out_dim=2, dropout=0.3):
        super().__init__()
        # Thay thế nn.LSTM bằng nn.RNN
        self.rnn = nn.RNN(in_dim, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        # Dùng self.rnn thay cho self.lstm
        out, _ = self.rnn(x)
        last_h = out[:, -1, :]
        y = self.head(last_h)
        return y


# --- KẾT THÚC THAY ĐỔI ---

def main():
    npz = np.load(DATA_NPZ, allow_pickle=True)
    X_train, y_train = npz["X_train"], npz["y_train"]
    X_valid, y_valid = npz["X_valid"], npz["y_valid"]
    X_test, y_test = npz["X_test"], npz["y_test"]
    INPUT_FEATURES = list(npz["INPUT_FEATURES"])
    TARGET_FEATURES = list(npz["TARGET_FEATURES"])

    print("Loaded splits:",
          X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)

    train_loader = DataLoader(StormSeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(StormSeqDataset(X_valid, y_valid), batch_size=BATCH_SIZE)
    test_loader = DataLoader(StormSeqDataset(X_test, y_test), batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sử dụng class RNNForecaster mới
    model = RNNForecaster(in_dim=len(INPUT_FEATURES),
                          out_dim=y_train.shape[2]).to(device)

    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3, verbose=True)

    mae_metric = MeanAbsoluteError().to(device)

    def run_epoch(loader, train_mode):
        model.train() if train_mode else model.eval()
        tot_loss, n = 0.0, 0
        mae_metric.reset()

        with torch.set_grad_enabled(train_mode):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = crit(pred, yb)
                if train_mode:
                    opt.zero_grad();
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                mae_metric.update(pred, yb)
                bs = xb.size(0)
                tot_loss += loss.item() * bs
                n += bs

        epoch_mae = mae_metric.compute()
        return tot_loss / max(n, 1), epoch_mae

    best, bad = float('inf'), 0
    CKPT.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_mae = run_epoch(train_loader, True)
        va_loss, va_mae = run_epoch(valid_loader, False)
        sched.step(va_loss)
        print(f"Epoch {ep:02d} | train loss {tr_loss:.6f} | valid loss {va_loss:.6f} | valid MAE {va_mae:.6f}")

        if va_loss < best:
            best, bad = va_loss, 0
            torch.save({"model": model.state_dict(),
                        "INPUT_FEATURES": INPUT_FEATURES,
                        "TARGET_FEATURES": TARGET_FEATURES}, CKPT)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping.");
                break

    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model"])
    test_loss, test_mae = run_epoch(test_loader, False)
    print(f"[TEST] MSE: {test_loss:.6f} | MAE: {test_mae:.6f}")


if __name__ == "__main__":
    main()