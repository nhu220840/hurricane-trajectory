# scripts/2_train_model.py
import numpy as np
import torch
import pickle
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchmetrics.regression import MeanAbsoluteError

# ===== CẤU HÌNH =====
ROOT = Path(__file__).resolve().parents[1]
DATA_NPZ = ROOT / "data/processed_data.npz"
CKPT_DIR = ROOT / "models"
CKPT_PATH = CKPT_DIR / "best_lstm_model.pt"
BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 10

class StormSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).squeeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class LSTMForecaster(nn.Module):
    def __init__(self, in_dim, hidden=20, num_layers=1, out_dim=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        last_h = out[:, -1, :]
        return self.head(last_h)

def run_epoch(loader, model, crit, opt, device, train_mode, mae_metric):
    model.train() if train_mode else model.eval()
    total_loss, n_samples = 0.0, 0
    mae_metric.reset()
    with torch.set_grad_enabled(train_mode):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            if train_mode:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            mae_metric.update(pred, yb)
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
    return total_loss / max(n_samples, 1), mae_metric.compute()

def main():
    npz = np.load(DATA_NPZ, allow_pickle=True)
    X_train, y_train = npz["X_train"], npz["y_train"]
    X_valid, y_valid = npz["X_valid"], npz["y_valid"]
    X_test, y_test = npz["X_test"], npz["y_test"]
    INPUT_FEATURES = list(npz["INPUT_FEATURES"])
    train_loader = DataLoader(StormSeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(StormSeqDataset(X_valid, y_valid), batch_size=BATCH_SIZE)
    test_loader = DataLoader(StormSeqDataset(X_test, y_test), batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    model = LSTMForecaster(in_dim=X_train.shape[2], out_dim=y_train.shape[2]).to(device)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3, verbose=True)
    mae_metric = MeanAbsoluteError().to(device)
    best_loss, bad_epochs = float('inf'), 0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nBắt đầu huấn luyện...")
    for ep in range(1, EPOCHS + 1):
        tr_loss, _ = run_epoch(train_loader, model, crit, opt, device, True, mae_metric)
        va_loss, va_mae = run_epoch(valid_loader, model, crit, None, device, False, mae_metric)
        sched.step(va_loss)
        print(f"Epoch {ep:02d} | Train Loss {tr_loss:.6f} | Valid Loss {va_loss:.6f} | Valid MAE {va_mae:.6f}")
        if va_loss < best_loss:
            best_loss, bad_epochs = va_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "input_features": INPUT_FEATURES, "target_features": list(npz["TARGET_FEATURES"])}, CKPT_PATH)
            print(f" -> Đã lưu model tốt nhất.")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE: print("Early stopping."); break
    print("\nĐánh giá trên tập test...")
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_mae = run_epoch(test_loader, model, crit, None, device, False, mae_metric)
    print(f"[KẾT QUẢ TEST] MSE: {test_loss:.6f} | MAE: {test_mae:.6f}")

if __name__ == "__main__":
    main()