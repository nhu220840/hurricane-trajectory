# src/train.py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanAbsoluteError
from src import config, models, dataset


# --- Logic huấn luyện (Lấy từ REbuildLSTM.ipynb vì rõ ràng hơn) ---

@torch.no_grad()
def _eval_epoch(loader, model, crit, device, mae_metric):
    model.eval()
    total_loss, n_samples = 0.0, 0
    mae_metric.reset()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = crit(pred, yb)
        mae_metric.update(pred, yb)
        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
    return total_loss / max(n_samples, 1), mae_metric.compute()


def _train_epoch(loader, model, crit, opt, device, mae_metric):
    model.train()
    total_loss, n_samples = 0.0, 0
    mae_metric.reset()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = crit(pred, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        mae_metric.update(pred, yb)
        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
    return total_loss / max(n_samples, 1), mae_metric.compute()


def run_epoch(loader, model, crit, opt, device, train_mode, mae_metric):
    if train_mode:
        return _train_epoch(loader, model, crit, opt, device, mae_metric)
    else:
        return _eval_epoch(loader, model, crit, device, mae_metric)


# --- Hàm huấn luyện chính ---

def run_training(model_type: str):
    """
    Hàm chính để huấn luyện một model.
    model_type: 'pytorch' hoặc 'scratch'
    """
    if model_type not in ['pytorch', 'scratch']:
        raise ValueError("model_type phải là 'pytorch' hoặc 'scratch'")

    print(f"\n--- Bắt đầu huấn luyện model: {model_type.upper()} ---")

    # 1. Tải dữ liệu
    try:
        npz = np.load(config.PROCESSED_NPZ_PATH, allow_pickle=True)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {config.PROCESSED_NPZ_PATH}.")
        print("Vui lòng chạy bước '--process-data' trước.")
        return

    X_train, y_train = npz["X_train"], npz["y_train"]
    X_valid, y_valid = npz["X_valid"], npz["y_valid"]
    INPUT_FEATURES = list(npz["INPUT_FEATURES"])
    TARGET_FEATURES = list(npz["TARGET_FEATURES"])

    in_dim = X_train.shape[2]
    out_dim = y_train.shape[-1] if y_train.ndim >= 2 else 1

    # 2. Tạo DataLoaders
    train_loader = DataLoader(dataset.StormSeqDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset.StormSeqDataset(X_valid, y_valid), batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 4. Khởi tạo Model, Loss, Optimizer
    params = config.MODEL_PARAMS[model_type]

    if model_type == 'pytorch':
        model = models.LSTMForecaster(
            in_dim=in_dim,
            hidden=params["hidden"],
            num_layers=params["num_layers"],
            out_dim=out_dim,
            dropout=params["dropout"]
        ).to(device)
        checkpoint_path = config.CKPT_PATH_PYTORCH
    else:  # 'scratch'
        model = models.LSTMFromScratchForecaster(
            in_dim=in_dim,
            hidden=params["hidden"],
            num_layers=params["num_layers"],
            out_dim=out_dim,
            dropout=params["dropout"]
        ).to(device)
        checkpoint_path = config.CKPT_PATH_SCRATCH

    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    mae_metric = MeanAbsoluteError().to(device)

    # 5. Vòng lặp huấn luyện
    best_loss, bad_epochs = float("inf"), 0
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for ep in range(1, config.EPOCHS + 1):
        tr_loss, _ = run_epoch(train_loader, model, crit, opt, device, True, mae_metric)
        va_loss, va_mae = run_epoch(valid_loader, model, crit, None, device, False, mae_metric)
        sched.step(va_loss)
        print(f"Epoch {ep:02d} | Train Loss {tr_loss:.6f} | Valid Loss {va_loss:.6f} | Valid MAE {va_mae:.6f}")

        if va_loss < best_loss - 1e-12:
            best_loss, bad_epochs = va_loss, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_features": INPUT_FEATURES,
                "target_features": TARGET_FEATURES,
                "in_dim": in_dim,
                "out_dim": out_dim,
                "config": params,
            }, checkpoint_path)
            print(" -> Đã lưu model tốt nhất.")
        else:
            bad_epochs += 1
            if bad_epochs >= config.PATIENCE:
                print("Early stopping do không cải thiện trên tập validation.")
                break

    print(f"--- Hoàn tất huấn luyện model: {model_type.upper()} ---")