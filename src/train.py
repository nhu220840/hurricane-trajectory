# src/train.py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import MeanAbsoluteError

# Thay đổi import model
from src import config
from src.model import LSTMAttention # <-- THAY ĐỔI

class StormSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # y có thể có shape (n, 1, 2), squeeze(1) để thành (n, 2)
        self.y = torch.tensor(y, dtype=torch.float32).squeeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def run_epoch(loader, model, crit, opt, device, train_mode, mae_metric):
    # (Hàm này giữ nguyên, không cần thay đổi)
    model.train() if train_mode else model.eval()
    total_loss, n_samples = 0.0, 0
    mae_metric.reset()
    with torch.set_grad_enabled(train_mode):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            if train_mode:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            mae_metric.update(pred, yb)
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
    return total_loss / max(n_samples, 1), mae_metric.compute()

def run_training():
    """
    Hàm chính để chạy toàn bộ pipeline huấn luyện model.
    """
    npz = np.load(config.PROCESSED_DATA_NPZ, allow_pickle=True)
    X_train, y_train = npz["X_train"], npz["y_train"]
    X_valid, y_valid = npz["X_valid"], npz["y_valid"]
    X_test, y_test = npz["X_test"], npz["y_test"]
    INPUT_FEATURES = list(npz["INPUT_FEATURES"])
    TARGET_FEATURES = list(npz["TARGET_FEATURES"])

    # y_train đã được squeeze trong data_processing
    train_loader = DataLoader(StormSeqDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(StormSeqDataset(X_valid, y_valid), batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(StormSeqDataset(X_test, y_test), batch_size=config.BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # --- THAY ĐỔI: Đảm bảo out_dim = len(TARGET_FEATURES) ---
    model = LSTMAttention(
        in_dim=X_train.shape[2],         # ~22 features
        out_dim=len(TARGET_FEATURES),  # 5 features
        **config.MODEL_PARAMS
    ).to(device)
    # ----------------------------------------------------

    crit = nn.HuberLoss() # Dùng HuberLoss tốt cho delta
    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3, verbose=True)
    mae_metric = MeanAbsoluteError().to(device)

    best_loss, bad_epochs = float('inf'), 0
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("\nBắt đầu huấn luyện model LSTM + Attention (với 5 delta targets)...")
    for ep in range(1, config.EPOCHS + 1):
        tr_loss, _ = run_epoch(train_loader, model, crit, opt, device, True, mae_metric)
        va_loss, va_mae = run_epoch(valid_loader, model, crit, None, device, False, mae_metric)
        sched.step(va_loss)
        print(f"Epoch {ep:02d} | Train Loss {tr_loss:.6f} | Valid Loss {va_loss:.6f} | Valid MAE {va_mae:.6f}")

        if va_loss < best_loss:
            best_loss, bad_epochs = va_loss, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_features": INPUT_FEATURES,
                "target_features": TARGET_FEATURES, # Lưu 5 target features
                "model_params": config.MODEL_PARAMS
            }, config.MODEL_CKPT_PATH)
            print(f" -> Đã lưu model tốt nhất.")
        else:
            bad_epochs += 1
            if bad_epochs >= config.PATIENCE:
                print("Early stopping.")
                break

    print("\nĐánh giá trên tập test...")
    checkpoint = torch.load(config.MODEL_CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_mae = run_epoch(test_loader, model, crit, None, device, False, mae_metric)
    print(f"[KẾT QUẢ TEST] Loss: {test_loss:.6f} | MAE: {test_mae:.6f}")